#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YuE2 音乐生成器 —— 一键安装脚本

做四件事：
  1. 在程序目录里建一个独立的 Python 环境（不污染系统）
  2. 装 PyTorch（CUDA 版）+ YuE2 推理包
  3. 从国内镜像下载模型权重（约 7.8 GB）
  4. 写一个安装完成标记

重复运行是安全的，已完成的步骤会自动跳过。
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENV = ROOT / ".venv"
MODELS = ROOT / "models"
WHEEL_NAME = "yue2_infer-0.1.5-py3-none-any.whl"
MIRROR = "https://hf-mirror.com"
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
MARKER = ROOT / ".installed"

IS_WINDOWS = os.name == "nt"

# PyTorch CUDA 轮子的下载源，按速度排序实测过：
# 官方 CDN 国内约 0.17 MB/s，交大约 56 MB/s，阿里约 0.1 MB/s
TORCH_MIRRORS = [
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128",
    "https://mirrors.aliyun.com/pytorch-wheels/cu128",
    "https://download.pytorch.org/whl/cu128",
]

MOT_REPO = "m-a-p/YuE2-3B"
VAE_REPO = "m-a-p/YuE2-Vae"

# 模型目录里实际需要的文件（对照 yue2 包的 MODEL_FILES 清单核过）。
# assets/ 和 examples/ 是演示素材，推理用不到，不下载。
MODEL_MANIFEST = {
    MOT_REPO: [
        "LICENSE", "THIRD_PARTY_NOTICES.md",
        "config.json", "generation_config.json", "yue2_generation_config.json",
        "weights_manifest.json", "model.safetensors",
        "modeling_yue2.py", "qwen.tiktoken",
        "licenses/SnakeBeta-NVIDIA-MIT.txt",
        "licenses/stable-audio-tools-MIT.txt",
    ],
    VAE_REPO: [
        "LICENSE", "THIRD_PARTY_NOTICES.md",
        "config.json", "weights_manifest.json", "model.safetensors",
        "modeling_vae.py",
        "licenses/SnakeBeta-NVIDIA-MIT.txt",
        "licenses/stable-audio-tools-MIT.txt",
    ],
}
MODEL_DIRS = {MOT_REPO: "YuE2-3B", VAE_REPO: "YuE2-Vae"}


def log(msg=""):
    print(msg, flush=True)


def force_utf8():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def banner(title):
    log()
    log("=" * 64)
    log(f"  {title}")
    log("=" * 64)


def run(cmd, **kw):
    """执行命令并实时透传输出；失败时抛出带命令内容的异常。"""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], **kw)
    if result.returncode != 0:
        raise RuntimeError(f"命令失败（退出码 {result.returncode}）：{' '.join(str(c) for c in cmd)}")
    return result


def venv_python():
    """Windows 在 Scripts\\ 下，Linux/macOS 在 bin/ 下。"""
    if IS_WINDOWS:
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def check_base_python():
    banner("步骤 1/4 · 检查基础 Python")
    if sys.version_info < (3, 10):
        log(f"当前 Python {sys.version.split()[0]} 太旧，YuE2 需要 3.10 或更高。")
        log("请到 https://www.python.org/downloads/ 装一个 3.10+ 后重试。")
        return False
    log(f"基础 Python：{sys.version.split()[0]}  ({sys.executable})")
    log("通过。")
    return True


def create_venv():
    banner("步骤 2/4 · 创建独立运行环境")
    if venv_python().exists():
        log(f"已存在，跳过：{VENV}")
    else:
        log(f"正在创建：{VENV}")
        run([sys.executable, "-m", "venv", str(VENV)])
        log("创建完成。")

    log("升级 pip ...")
    run([venv_python(), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])

    # 国内装包走清华源，快很多。Windows 读 pip.ini，Linux 读 pip.conf。
    cfg = VENV / ("pip.ini" if IS_WINDOWS else "pip.conf")
    cfg.write_text(
        "[global]\n"
        f"index-url = {PYPI_MIRROR}\n"
        "timeout = 120\n"
        "[install]\n"
        "trusted-host = pypi.tuna.tsinghua.edu.cn\n",
        encoding="utf-8",
    )
    log("已把 pip 源切到清华镜像。")
    return True


def torch_wheel_name(py):
    """按 venv 的 Python 版本 + 当前平台推出对应的 wheel 文件名。"""
    tag = subprocess.check_output(
        [py, "-c",
         "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True).strip()

    if IS_WINDOWS:
        plat = "win_amd64"
    else:
        machine = platform.machine().lower()
        plat = ("manylinux_2_28_aarch64" if machine in ("aarch64", "arm64")
                else "manylinux_2_28_x86_64")
    return f"torch-2.10.0%2Bcu128-{tag}-{tag}-{plat}.whl"


def download_with_progress(url, dest):
    """带进度输出的下载。返回 (已下载字节, 服务端声明总字节)。"""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 3:
                    last = now
                    if total:
                        log(f"      已下载 {done / 1e6:.0f} / {total / 1e6:.0f} MB"
                            f"  ({done / total * 100:.0f}%)")
                    else:
                        log(f"      已下载 {done / 1e6:.0f} MB")
    return done, total


def wheel_is_complete(path, expected_total=0):
    """轮子必须是完整可解的 zip —— 光比体积会被截断的下载骗过去。"""
    import zipfile

    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    if expected_total and path.stat().st_size != expected_total:
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def install_torch():
    """下载 PyTorch 的 CUDA 版本。

    官方 CDN 在国内实测只有 0.17 MB/s（要 3 小时），
    上海交大镜像实测 56 MB/s（约 1 分钟），所以按顺序试镜像。
    """
    cache = ROOT / "_cache"
    cache.mkdir(exist_ok=True)
    name = torch_wheel_name(venv_python())
    local = cache / name.replace("%2B", "+")

    if wheel_is_complete(local):
        log(f"  已有完整缓存，跳过下载：{local.name}"
            f"  ({local.stat().st_size / 1e9:.2f} GB)")
    else:
        errors = []
        for base in TORCH_MIRRORS:
            log(f"  尝试镜像：{base}")
            try:
                done, total = download_with_progress(f"{base}/{name}", local)
                if total and done < total:
                    errors.append(f"{base} 只下到 {done / 1e6:.0f}/"
                                  f"{total / 1e6:.0f} MB")
                    continue
                if wheel_is_complete(local, total):
                    break
                errors.append(f"{base} 下载的文件损坏")
            except Exception as exc:
                errors.append(f"{base} -> {exc}")
                log(f"      失败：{exc}")
        else:
            log()
            log("  所有镜像都失败了：")
            for e in errors:
                log(f"    - {e}")
            raise RuntimeError("PyTorch 下载失败，检查网络后重新运行本脚本")

    size = local.stat().st_size / 1e9
    log(f"  下载完成并校验通过：{local.name}  ({size:.2f} GB)")
    log()
    log("  正在安装（解包约需 1-3 分钟）…")
    run([venv_python(), "-m", "pip", "install", str(local)])


def install_packages():
    banner("步骤 3/4 · 安装 PyTorch 与 YuE2 推理包")

    log()
    log("--- PyTorch 2.10.0 (CUDA 12.8)，约 2.9 GB ---")
    install_torch()

    log()
    log("--- YuE2 推理包及其依赖 ---")
    wheel_path = ROOT / WHEEL_NAME
    if not wheel_path.exists() or wheel_path.stat().st_size < 50_000:
        log(f"  下载推理包：{WHEEL_NAME}")
        # 注意：镜像会拒掉 urllib 的默认 UA（403），必须伪装成浏览器
        download_with_progress(
            f"{MIRROR}/{MOT_REPO}/resolve/main/{WHEEL_NAME}", wheel_path)
    run([venv_python(), "-m", "pip", "install", str(wheel_path)])

    log()
    log("--- 界面所需的附加依赖 ---")
    run([venv_python(), "-m", "pip", "install", "huggingface-hub==0.36.2"])
    return True


def hardware_report():
    """用 venv 里的 torch 读显卡信息。"""
    code = (
        "import torch,json;"
        "ok=torch.cuda.is_available();"
        "d={'cuda':ok};"
        "p=torch.cuda.get_device_properties(0) if ok else None;"
        "d.update(name=(p.name if ok else ''),"
        " vram=(p.total_memory/1024**3 if ok else 0),"
        " cc=(list(torch.cuda.get_device_capability(0)) if ok else [0,0]),"
        " bf16=(torch.cuda.is_bf16_supported() if ok else False));"
        "print('HWJSON'+json.dumps(d))"
    )
    try:
        out = subprocess.run([str(venv_python()), "-c", code],
                             capture_output=True, text=True, timeout=300)
    except Exception:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("HWJSON"):
            return json.loads(line[len("HWJSON"):])
    return None


def check_hardware():
    """在下载 7.8 GB 模型之前先判断这台机器能不能跑，避免白下。"""
    banner("硬件检查")

    hw = hardware_report()
    if hw is None:
        log("  读不到显卡信息，跳过检查。")
        return True

    if not hw.get("cuda"):
        log("  没有检测到可用的 NVIDIA 显卡。")
        log("  YuE2 必须要 NVIDIA GPU，这台机器跑不了推理。")
        log("  可以继续安装，但装完只能把整个文件夹搬到有 N 卡的机器上用。")
        return True

    vram = float(hw.get("vram") or 0)
    cc = hw.get("cc") or [0, 0]
    log(f"  显卡：{hw.get('name')}")
    log(f"  显存：{vram:.1f} GB")
    log(f"  算力：CC {cc[0]}.{cc[1]}    BF16：{'支持' if hw.get('bf16') else '不支持'}")

    try:
        import ram_mode
        mem = ram_mode.system_memory()
        if mem:
            log(f"  内存：{mem[0]:.1f} GB（当前可用 {mem[1]:.1f} GB）")
    except Exception:
        mem = None
    log()

    # 分档依据是实测，不是照抄官方文档：
    # 官方要求 24 GB，但权重实际只有 6.763 GiB（AR 2.625 + NAR 2.625 + 词表 1.513）。
    # 官方实现要求两条路径同时驻留显存，所以 8 GB 卡必然溢出；
    # ram_mode 按阶段把闲置的那条路径换回内存，峰值显存降到约 5.7 GiB。
    if vram >= 20:
        log("  ✓ 显存充足，将以完整模式全速运行。")
    elif vram >= 14:
        log("  ✓ 显存够用，将以标准模式运行。")
    elif vram >= 8:
        log("  ✓ 将启用内存扩展模式：闲置权重随阶段在内存和显存之间搬运。")
        log("    实测峰值显存约 5.7 GiB，8 GB 卡可以生成完整歌曲。")
        if mem and mem[1] < 4.0:
            log(f"  ⚠ 当前可用内存只有 {mem[1]:.1f} GB，偏紧。")
            log("    生成前关掉浏览器、聊天软件会稳很多。")
    elif vram >= 6:
        log("  ⚠ 显存偏小，会启用内存扩展模式并自动切到旋律模式，建议 2 分钟以内。")
    else:
        log("  ⚠ 显存很小，只能试很短的片段，而且可能失败。")

    log()
    log("  继续下载模型。装好后本机跑不动也可以整体搬到别的机器。")
    return True


def download_models():
    banner("步骤 4/4 · 下载模型权重（约 7.8 GB）")

    log(f"使用镜像：{MIRROR}")
    log("（本机直连 huggingface.co 不通，必须走镜像）")
    log()
    log("说明：这里不用 huggingface_hub 的整仓下载，而是按清单逐个拉。")
    log("      整仓下载会创建并删除 .lock 临时文件，容易被安全策略拦截。")

    total = 0
    for repo, files in MODEL_MANIFEST.items():
        target = MODELS / MODEL_DIRS[repo]
        log()
        log(f"--> {repo}")
        for name in files:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                log(f"    已有，跳过：{name}")
            else:
                log(f"    下载：{name}")
                download_with_progress(
                    f"{MIRROR}/{repo}/resolve/main/{name}", dest)
            total += dest.stat().st_size

    log()
    log(f"  合计：{total / 1e9:.2f} GB")

    need = [
        MODELS / "YuE2-3B" / "model.safetensors",
        MODELS / "YuE2-3B" / "config.json",
        MODELS / "YuE2-3B" / "qwen.tiktoken",
        MODELS / "YuE2-3B" / "weights_manifest.json",
        MODELS / "YuE2-Vae" / "model.safetensors",
        MODELS / "YuE2-Vae" / "config.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if missing:
        log()
        log("下载不完整，缺少以下文件：")
        for m in missing:
            log(f"  - {m}")
        log("重新运行本脚本即可续传。")
        return False

    log()
    log("模型文件校验通过。")
    return True


def final_report():
    banner("安装完成")
    log("双击 start.bat 就能打开生成界面。")
    log()
    log(f"  程序目录：{ROOT}")
    log(f"  运行环境：{VENV}")
    log(f"  模型文件：{MODELS}")
    log()
    MARKER.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def main():
    force_utf8()
    log()
    log("  YuE2 音乐生成器 · 安装程序")
    log("  " + "-" * 60)
    log("  全程大约需要 10-20 分钟，取决于网速。中途别关窗口。")
    log()

    if MARKER.exists() and "--force" not in sys.argv:
        log("检测到已安装过。")
        log("如果要重新安装，加 --force 参数运行；否则直接双击 start.bat 即可。")
        return 0

    try:
        if not check_base_python():
            return 1
        if not create_venv():
            return 1
        if not install_packages():
            return 1
        check_hardware()
        if not download_models():
            return 1
        final_report()
        return 0
    except KeyboardInterrupt:
        log()
        log("已中断。重新运行会从断点继续。")
        return 130
    except Exception as exc:
        log()
        log("=" * 64)
        log("  安装失败")
        log("=" * 64)
        log(str(exc))
        log()
        log("常见原因：")
        log("  - 网络不通：本机直连 huggingface.co 不通，本脚本已自动走 hf-mirror.com")
        log("  - 磁盘不够：需要约 15 GB 空闲空间")
        log("  - 装包被中断：重新运行本脚本，已完成的步骤会自动跳过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
