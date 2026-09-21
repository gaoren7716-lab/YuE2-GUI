#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YuE2 音乐生成器 —— 本地图形界面

双击 start.bat 启动，浏览器会自动打开。界面只有三步：选风格、写歌词、点生成。

设计原则：
  - 显存自动识别，自动选最合适的配置档，用户不用懂任何参数
  - 显存不够时自动启用「内存扩展模式」，把当前用不到的权重放回内存
  - 只暴露「时长」一个技术相关选项，且按硬件能力自动限制
  - 生成过程有实时进度，结果直接在线播放 + 一键下载
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 必须在 import torch 之前设置：expandable_segments 能明显减少
# "反复申请/释放大块权重"产生的显存碎片，是内存扩展模式的前提。
# 但它只在 Linux 上被 PyTorch 支持；Windows 上设了会被忽略，还会多打印一行
# "expandable_segments not supported on this platform" 警告，所以按平台来。
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).parent.resolve()
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
GIB = 1024 ** 3

# 每次生成会往 outputs/ 里留一个 6-10 MB 的目录。界面不读历史目录
# （结果是通过 /api/audio?dir=<目录名> 直接取的），所以只保留最近若干首，
# 更早的自动删掉，免得长期用下来把磁盘塞满。
# 想长期保存某一首，用界面上的「下载」按钮存到自己想要的位置。
KEEP_DIRS = 20

# 只认 <日期>-<时间>-<种子> 这种自己生成的目录名，
# 不碰用户手动放进 outputs/ 的其它东西。
GENERATED_DIR = re.compile(r"^\d{8}-\d{6}-\d+$")

# 生成出来的是无损 flac，音质最好，但微信和部分手机播放器不认。
# 机器上有 ffmpeg 就额外提供一个 MP3 下载口，方便分享；没有就只给 flac。
FFMPEG = shutil.which("ffmpeg")

# --------------------------------------------------------------------------
# 创作选项：把底层推理参数夹到合法区间
# --------------------------------------------------------------------------
# 范围来自 yue2.protocol.Sampling 的校验规则。底层是 frozen dataclass +
# 严格校验，一个越界值会让整次生成直接失败，所以必须在入口就把关，
# 不能让用户输入原样捅到推理层。
SAMPLING_LIMITS = {
    "temperature": (0.0, 5.0),        # 越高越发散，创造性来源
    "top_p": (0.01, 1.0),
    "top_k": (1, 500),                # 每步只从概率最高的前 N 个音里挑
    "penalty_window": (1, 100),       # 重复抑制往前看多少个音
    "repetition_penalty": (1.0, 2.0),
}
DEFAULT_SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 100,
    "penalty_window": 50, "repetition_penalty": 1.2,
}
ODE_STEPS_CHOICES = (16, 32, 48)
COT_CHOICES = ("off", "melody", "full")

# 一次最多出几首。出得多挑得爽，但总耗时是线性翻倍的，4 首是甜点
MAX_BATCH = 4


def sanitize_options(raw):
    """把前端传来的创作选项夹到合法区间，坏值一律退回默认。"""
    raw = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULT_SAMPLING)

    for key, (lo, hi) in SAMPLING_LIMITS.items():
        try:
            val = float(raw.get(key))
        except (TypeError, ValueError):
            continue
        if val != val:                      # NaN
            continue
        out[key] = min(hi, max(lo, val))
    out["top_k"] = int(out["top_k"])
    out["penalty_window"] = int(out["penalty_window"])

    try:
        steps = int(raw.get("ode_steps") or 32)
    except (TypeError, ValueError):
        steps = 32
    out["ode_steps"] = steps if steps in ODE_STEPS_CHOICES else 32

    # cfg_scale 传 None 表示"用官方默认"（cot=off 时 1.01，否则 1.0）
    cfg = raw.get("cfg_scale")
    out["cfg_scale"] = None
    if cfg not in (None, ""):
        try:
            cfg = float(cfg)
            if cfg == cfg:
                out["cfg_scale"] = min(20.0, max(0.0, cfg))
        except (TypeError, ValueError):
            pass

    abc = raw.get("abc")
    out["abc"] = abc.strip() if isinstance(abc, str) and abc.strip() else None

    cot = raw.get("cot")
    out["cot"] = cot if cot in COT_CHOICES else None

    # 外部乐谱要求 cot 不是 off，冲突时自动抬到 full（否则底层会抛错）
    if out["abc"] and (out["cot"] or PROFILE.get("cot")) == "off":
        out["cot"] = "full"
    return out


def prune_outputs(keep=KEEP_DIRS):
    """保留最近 keep 个生成目录，返回删掉的数量。"""
    try:
        dirs = [d for d in OUTPUTS.iterdir()
                if d.is_dir() and GENERATED_DIR.match(d.name)]
    except OSError:
        return 0
    # 目录名以时间戳开头，字典序就是时间序，倒序后前 keep 个是最新的
    dirs.sort(key=lambda d: d.name, reverse=True)
    removed = 0
    for old in dirs[keep:]:
        try:
            shutil.rmtree(old, ignore_errors=True)
            removed += 1
        except Exception:
            pass
    return removed


def list_history(limit=8):
    """列出最近生成的作品，给界面上的「历史作品」用。

    界面不扫描目录是为了启动快，但用户关掉程序之后总得有个地方找回
    自己生成的歌——让他去 outputs/ 里翻文件夹名不叫傻瓜式。
    """
    items = []
    try:
        dirs = [d for d in OUTPUTS.iterdir()
                if d.is_dir() and GENERATED_DIR.match(d.name)]
    except OSError:
        return items
    dirs.sort(key=lambda d: d.name, reverse=True)
    for d in dirs:
        if not (d / "audio.flac").exists():
            continue
        seconds = None
        try:
            meta = json.loads((d / "result.json").read_text(encoding="utf-8"))
            seconds = meta.get("audio_seconds")
        except Exception:
            pass
        try:
            mtime = d.stat().st_mtime
        except OSError:
            continue
        items.append({"dir": d.name, "seconds": seconds, "mtime": mtime,
                      "has_score": (d / "score.abc").exists()})
        if len(items) >= limit:
            break
    return items


# 界面监听地址与端口。云 GPU 上部署时设 YUE2_BIND=0.0.0.0 才能从外部访问。
BIND = os.environ.get("YUE2_BIND", "127.0.0.1")
PORT = int(os.environ.get("YUE2_PORT", "7861"))

# 实测反推（2026-09-20，420 令牌 → 16.80 秒音频，latent 帧数与令牌数严格 1:1）：
# VAE 是 25 帧/秒，所以 1 个语义令牌 = 1 帧 = 0.04 秒音频。
# 官方 max_tokens=9000 的上限因此对应 360 秒 = 6 分钟音频。
FRAMES_PER_SECOND = 25.0
TOKENS_PER_SECOND = FRAMES_PER_SECOND
MIN_TOKENS = 200
MAX_TOKENS = 9000


def force_utf8():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# --------------------------------------------------------------------------
# 硬件识别与自动配置
# --------------------------------------------------------------------------

def detect_hardware():
    hw = {"cuda": False, "torch": None, "error": None}
    try:
        import torch
        hw["torch"] = torch.__version__
        hw["cuda"] = torch.cuda.is_available()
        if hw["cuda"]:
            props = torch.cuda.get_device_properties(0)
            hw["name"] = props.name
            hw["vram_gib"] = props.total_memory / GIB
            hw["cc"] = torch.cuda.get_device_capability(0)
            hw["bf16"] = torch.cuda.is_bf16_supported()
            free, _ = torch.cuda.mem_get_info()
            hw["vram_free_gib"] = free / GIB
    except Exception as exc:
        hw["error"] = str(exc)

    # 内存扩展模式要靠内存兜底，所以内存也要体检
    try:
        import ram_mode
        info = ram_mode.system_memory()
        if info:
            hw["ram_total_gb"], hw["ram_free_gb"] = info[0], info[1]
    except Exception:
        pass
    return hw


def pick_profile(hw):
    """按显存自动选配置档。返回的 dict 直接喂给 pipeline。

    YuE2 是 MoT 双路径模型：AR 路径 2.625 GiB + NAR 路径 2.625 GiB +
    词表权重 1.41 GiB = 6.763 GiB，官方实现要求它们同时驻留显存，
    所以 8 GB 卡必然溢出。内存扩展模式（ram_mode）按阶段把闲置的那条
    路径挪回内存，峰值显存降到约 5.7 GiB，8 GB 卡就能跑完整歌曲。
    """
    if not hw.get("cuda"):
        return {
            "label": "无法运行",
            "note": "没有检测到可用的 NVIDIA 显卡。YuE2 需要 NVIDIA GPU。",
            "max_seconds": 0,
            "disabled": True,
        }

    if not hw.get("bf16", False):
        return {
            "label": "无法运行",
            "note": "这张显卡不支持 BF16 精度，YuE2 无法在其上运行。",
            "max_seconds": 0,
            "disabled": True,
        }

    # 显存容量要先修约再比档位。实测 RTX 4070 Laptop 报告 7.9956 GiB，
    # 直接和 8 比会掉进下一档，白白降级成旋律模式。
    vram = round(hw.get("vram_gib", 0), 1)
    base = dict(device="cuda", disabled=False, quantization="none",
                backend="torch-eager", offload_ar=True, cot="full",
                memory_budget_gib=max(6, int(vram)))

    if vram >= 20:
        return {**base,
                "label": "完整模式",
                "note": "显存充足，直接全速运行，可以生成完整长度的歌曲并输出乐谱。",
                "ram_mode": False, "backend": "torch", "offload_ar": False,
                "memory_budget_gib": 24,
                "max_seconds": 240}

    if vram >= 14:
        return {**base,
                "label": "标准模式",
                "note": "显存够用，可以生成完整长度的歌曲。",
                "ram_mode": True,
                "max_seconds": 240}

    if vram >= 8:
        return {**base,
                "label": "内存扩展模式",
                "note": ("显存不够时自动把当前用不到的权重放回内存，"
                         "可以生成完整歌曲。速度比大显存慢一些，慢一点没关系。"),
                "ram_mode": True,
                # 实测（2026-09-20）：8GB 卡跑 221 秒完整歌曲，
                # 峰值显存 5.54 GiB、物理内存最低可用 0.47 GB，全程无 OOM。
                # 所以 4 分钟以内都放得开；超过 3 分钟内存会很紧，
                # 界面上的 durhint 会提醒用户先关掉别的程序。
                "max_seconds": 240}

    if vram >= 6:
        return {**base,
                "label": "内存扩展模式（紧凑）",
                "note": "显存偏小，已切换成旋律模式进一步降低占用，建议 2 分钟以内。",
                "ram_mode": True, "cot": "melody",
                "max_seconds": 120}

    return {**base,
            "label": "内存扩展模式（极限）",
            "note": "显存很小，只能试短片段，而且很可能失败。",
            "ram_mode": True, "cot": "off",
            "max_seconds": 60}


def models_ready():
    need = [
        MODELS / "YuE2-3B" / "model.safetensors",
        MODELS / "YuE2-3B" / "config.json",
        MODELS / "YuE2-3B" / "qwen.tiktoken",
        MODELS / "YuE2-Vae" / "model.safetensors",
    ]
    return all(p.exists() for p in need)


# --------------------------------------------------------------------------
# 全局状态
# --------------------------------------------------------------------------

HW = {}
PROFILE = {}
PIPE = None
PIPE_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()

JOB = {
    "status": "idle",          # idle | running | done | error
    "progress": 0,
    "stage": "",
    "message": "",
    "audio_url": None,
    "audio_name": None,
    "error": None,
    "seconds": None,
    "elapsed": None,
    "peak_vram": None,
    "seed": None,
    "truncated": None,
    "dir": None,
    "has_score": False,
    "batch_total": 0,          # 这次要出几首，1 = 单首
    "batch_index": 0,          # 正在做第几首（从 1 开始）
    "batch_done": [],          # 已完成的那几首，前端一次列出来挑
}


def set_job(**kw):
    with JOB_LOCK:
        JOB.update(kw)


def get_job():
    with JOB_LOCK:
        return dict(JOB)


def get_pipeline():
    """惰性加载并缓存 pipeline。加载一次要读 7 GB，不能每次生成都重来。"""
    global PIPE
    with PIPE_LOCK:
        if PIPE is not None:
            return PIPE
        from yue2 import YuE2Pipeline

        if PROFILE.get("ram_mode"):
            # 必须在构造 pipeline 之前打补丁：官方实现会在 __init__ 里
            # 把显存分配上限锁死，补丁晚了就不生效。
            import ram_mode
            fraction = ram_mode.install(device=PROFILE.get("device", "cuda"))
            print(f"  内存扩展模式：显存分配上限放宽到 {fraction:.0%}，"
                  "用不到的权重会随阶段在内存和显存之间搬运。")

        set_job(stage="加载模型",
                message="首次生成要把 7 GB 模型读进内存，约 1-3 分钟…",
                progress=2)
        PIPE = YuE2Pipeline.from_pretrained(
            str(MODELS / "YuE2-3B"),
            vae=str(MODELS / "YuE2-Vae"),
            local_files_only=True,
            progress=False,
            device=PROFILE.get("device", "cuda"),
            memory_budget_gib=PROFILE["memory_budget_gib"],
            backend=PROFILE["backend"],
            quantization=PROFILE["quantization"],
            offload_ar=PROFILE["offload_ar"],
            verify_hashes=False,
        )
        return PIPE


# --------------------------------------------------------------------------
# 生成任务
# --------------------------------------------------------------------------

def run_generation(style, lyrics, seconds, seed, options=None, batch=None):
    """生成一首。

    batch=(第几首, 共几首)，None 表示单首。批量时中间几首**不把 status 设成 done**，
    否则前端每秒轮询会以为整批做完了，提前收摊只显示第一首。
    """
    import torch

    opts = options or {}
    cot = opts.get("cot") or PROFILE["cot"]
    started = time.perf_counter()
    max_tokens = max(MIN_TOKENS, min(MAX_TOKENS, int(round(seconds * TOKENS_PER_SECOND))))

    # max_memory_allocated 是进程级的累计高水位，不重置的话第二首会
    # 直接显示第一首的峰值（实测两首都是 4.87 GiB，看不出差别）。
    # 这里按次清零，报出来的才是这首歌自己的峰值。
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    try:
        set_job(status="running", progress=2, stage="加载模型",
                message="正在准备模型，首次需要把 7 GB 权重读进内存…", error=None,
                audio_url=None, audio_name=None, seconds=None, elapsed=None,
                peak_vram=None, seed=seed, truncated=None,
                dir=None, has_score=False)

        pipe = get_pipeline()

        # 进度分两段：ABC 规划占 5-25%，语义生成占 25-88%。
        # ABC 长度波动很大（几百到几千令牌），先按 900 估，之后自然过渡。
        # 回调签名在不同版本间可能微调，所以参数全部给默认值。
        state = {"abc": 0, "semantic": 0, "last": time.perf_counter()}
        abc_estimate = 900 if cot != "off" else 1
        stop = {"flag": False}

        def on_token(phase=None, token=None):
            state["last"] = time.perf_counter()
            if phase == "abc":
                state["abc"] += 1
                done, total, lo, hi = state["abc"], abc_estimate, 5, 25
            else:
                state["semantic"] += 1
                done, total, lo, hi = state["semantic"], max_tokens, 25, 88
            pct = lo + int((hi - lo) * min(1.0, done / max(1, total)))
            with JOB_LOCK:
                JOB["progress"] = max(JOB["progress"], pct)
                if phase == "abc":
                    JOB["stage"] = "规划乐谱"
                    JOB["message"] = f"正在规划乐谱，已生成 {state['abc']} 个令牌…"
                else:
                    JOB["stage"] = "生成中"
                    JOB["message"] = (f"已生成 {state['semantic']} / {max_tokens} "
                                      f"个音乐令牌…")

        def watchdog():
            """令牌出完后还有 NAR 合成和 VAE 解码，两段都没有回调。

            实测 3 分钟的歌这两段要 100 秒左右，不提示的话界面会一直停在
            「已生成 N 个令牌」上，看着像卡死。
            """
            while not stop["flag"]:
                time.sleep(2)
                if time.perf_counter() - state["last"] < 20:
                    continue
                with JOB_LOCK:
                    if JOB["status"] != "running":
                        return
                    if JOB["progress"] < 88:
                        JOB["progress"] = 88
                    JOB["stage"] = "合成音频"
                    JOB["message"] = "令牌已生成完毕，正在把它们还原成声音…"

        threading.Thread(target=watchdog, daemon=True).start()
        set_job(progress=5, stage="生成中", message="开始生成…")

        # ode_steps 决定 NAR 合成的精细度，只能在 pipeline 加载后改。
        # 它是 frozen dataclass，改一个字段要重建实例。
        steps = int(opts.get("ode_steps") or 32)
        if steps != getattr(pipe.generation_config, "ode_steps", 32):
            from yue2.protocol import GenerationConfig
            pipe.generation_config = GenerationConfig(
                abc=pipe.generation_config.abc,
                semantic=pipe.generation_config.semantic,
                ode_steps=steps)

        sampling = {
            "max_tokens": max_tokens,
            "min_tokens": MIN_TOKENS,
            "temperature": float(opts.get("temperature", 1.0)),
            "top_p": float(opts.get("top_p", 0.95)),
            "top_k": int(opts.get("top_k", 100)),
            "penalty_window": int(opts.get("penalty_window", 50)),
            "repetition_penalty": float(opts.get("repetition_penalty", 1.2)),
        }
        call = {
            "style": style, "lyrics": lyrics, "cot": cot, "seed": seed,
            "semantic_sampling": sampling, "on_token": on_token,
        }
        if opts.get("cfg_scale") is not None:
            call["cfg_scale"] = float(opts["cfg_scale"])
        if opts.get("abc"):
            call["abc"] = opts["abc"]

        song = pipe(**call)

        set_job(progress=90, stage="保存", message="正在写入音频文件…")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = OUTPUTS / f"{stamp}-{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            song.save_artifacts(out_dir)
        except Exception:
            # 元数据保存失败不影响出音频
            song.save(out_dir / "audio.flac")

        audio_path = out_dir / "audio.flac"
        if not audio_path.exists():
            raise RuntimeError("生成结束但没有找到音频文件")

        audio = getattr(song, "audio", None)
        rate = getattr(song, "sample_rate", 48000)
        length = len(audio) / rate if audio is not None else None

        peak = None
        try:
            peak = torch.cuda.max_memory_allocated() / GIB
        except Exception:
            pass

        # 批量时**一律**保持 running，整批做完由 run_batch 统一收尾。
        # 若最后一首就设成 done，前端会在 batch_done 还没凑齐时提前收摊。
        in_batch = bool(batch)
        last_one = in_batch and batch[0] >= batch[1]
        set_job(
            status="running" if in_batch else "done",
            progress=100 if last_one else (96 if in_batch else 100),
            stage="完成" if not in_batch else f"第 {batch[0]}/{batch[1]} 首好了",
            message=("生成完成，可以试听了。" if not in_batch
                     else ("正在汇总这一批…" if last_one
                           else f"接着做第 {batch[0] + 1} 首，稍等…")),
            audio_url=f"/api/audio?dir={out_dir.name}",
            audio_name=f"yue2-{stamp}.flac",
            seconds=length,
            elapsed=time.perf_counter() - started,
            peak_vram=peak,
            truncated=getattr(song, "truncated", None),
            dir=out_dir.name,
            has_score=(out_dir / "score.abc").exists(),
            seed=seed,
            batch_index=batch[0] if in_batch else 0,
            batch_total=batch[1] if in_batch else 0,
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        text = str(exc).lower()
        if "out of memory" in text:
            hint = ("内存/显存同时吃紧。先关掉浏览器、聊天软件、游戏，"
                    "再把时长调短一档重试。程序已经把用不到的权重放回内存了，"
                    "剩下的就是给它腾地方。")
        elif "cuda error" in text:
            hint = ("显卡驱动报了错。重启一下程序通常就好；"
                    "如果反复出现，把下面的完整报错发出来。")
        elif "compute capability" in text:
            hint = "这张显卡不支持 FP8 量化。内存扩展模式不需要量化，请把量化关掉。"
        else:
            hint = "把下面的完整报错发出来，我来判断原因。"
        set_job(status="error", progress=0, stage="失败",
                message=hint, error=detail)
        traceback.print_exc()
    finally:
        try:
            stop["flag"] = True
        except NameError:
            pass
        # 释放中间显存，给下一次生成留空间
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        # 顺手清掉过老的生成目录，只留最近 KEEP_DIRS 首
        prune_outputs()
        # 把工作集还给系统。mmap 回填的权重页是文件支撑的干净页，丢掉零成本
        # （下次用到时从磁盘重读），但生成完成后桌面就不会一直卡着。
        # 批量中间不还：反正还要接着生成，还了下一首又得硬缺页读回来（每首 +8 秒）。
        if (not batch) or batch[0] >= batch[1]:
            try:
                import ram_mode
                model = getattr(PIPE, "_model", None)
                planner = ram_mode.planner_of(model) if model is not None else None
                ram_mode.trim_memory(getattr(planner, "weight_file", None))
            except Exception:
                pass


def run_batch(style, lyrics, seconds, seed, opts, count):
    """一次出 count 首，参数完全一样，只有种子逐个递增。

    采样随机性大，抽卡是提升产出质量最划算的手段：等差不多的时间，
    从「1 首」变成「4 首挑 1 首」。中途某首失败不丢已完成的。
    """
    results = []
    for i in range(count):
        set_job(
            status="running",
            progress=1,
            stage=f"第 {i + 1}/{count} 首",
            message="准备中…" if i == 0 else "接着来，别急…",
            error=None,
            batch_index=i + 1,
            batch_total=count,
            batch_done=list(results),
        )
        run_generation(style, lyrics, seconds, seed + i, opts, batch=(i + 1, count))

        job = get_job()
        if job.get("status") == "error":
            # 中途炸了：把已经做好的交出去，别让前面的白跑
            if results:
                set_job(batch_done=list(results),
                        message=(f"第 {i + 1} 首失败了，但前面 {len(results)} 首"
                                 "已经做好了，可以先用。"))
            return
        results.append({
            k: job.get(k) for k in
            ("audio_url", "audio_name", "seconds", "elapsed",
             "seed", "dir", "has_score", "peak_vram")
        })

    set_job(
        status="done",
        progress=100,
        stage="完成",
        message=f"{count} 首都好了，挑一首满意的吧。",
        batch_index=count,
        batch_total=count,
        batch_done=results,
    )


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------

PRESETS = [
    {
        "name": "中文流行",
        "style": ("Mandarin, contemporary pop ballad, warm expressive female vocal, "
                  "piano and strings, soft drums, emotional singable chorus, 82 BPM"),
        "lyrics": ("[Verse]\n路灯把影子拉得很长\n我数着回家的方向\n\n"
                   "[Chorus]\n如果风也会说话\n它会替我说想你啊"),
    },
    {
        "name": "英文摇滚",
        "style": ("English, alternative rock, gritty male lead vocal, distorted electric "
                  "guitars, driving bass, powerful live drums, anthemic chorus, 138 BPM"),
        "lyrics": ("[Verse]\nCity lights are burning low\nEvery street I used to know\n\n"
                   "[Chorus]\nWe are not afraid to fall\nWe are gonna take it all"),
    },
    {
        "name": "电子舞曲",
        "style": ("English, synthwave electronic, airy female vocal, analog synth pads, "
                  "punchy bass, four-on-the-floor drums, neon night mood, 118 BPM"),
        "lyrics": ("[Verse]\nMidnight running through the wire\nStatic dancing in the fire\n\n"
                   "[Chorus]\nTake me to the afterglow\nWhere the neon rivers flow"),
    },
    {
        "name": "纯音乐",
        "style": ("Instrumental, cinematic orchestral, sweeping strings, warm grand piano, "
                  "subtle percussion, uplifting and hopeful, 76 BPM"),
        "lyrics": "[Instrumental]",
    },
    {
        "name": "粤语抒情",
        "style": ("Cantonese, sentimental ballad, warm male vocal, grand piano and strings, "
                  "gentle brushed drums, late night Hong Kong mood, 72 BPM"),
        "lyrics": ("[Verse]\n旧照片翻到这一页\n街灯照住我嘅背影\n\n"
                   "[Chorus]\n如果时间可以倒流\n我想再听你讲一次"),
    },
    {
        "name": "日系 City Pop",
        "style": ("Japanese, city pop, bright female vocal, funky electric guitar, "
                  "slap bass, tight drums, 80s Tokyo neon night, 112 BPM"),
        "lyrics": ("[Verse]\n真夜中の高速を\n君と走り抜けて\n\n"
                   "[Chorus]\n夏の終わりに　また会えるよ\nネオンの海を泳いで"),
    },
    {
        "name": "爵士小酒馆",
        "style": ("English, cool jazz, smoky female vocal, tenor saxophone, upright bass, "
                  "brushed drums, intimate late-night club mood, 88 BPM"),
        "lyrics": ("[Verse]\nThe rain is tapping on the glass\nAnother hour, another glass\n\n"
                   "[Chorus]\nDon't tell me that the night is over\nWe still have one more song"),
    },
    {
        "name": "民谣吉他",
        "style": ("Mandarin, acoustic folk, intimate male vocal, fingerpicked acoustic guitar, "
                  "light harmonica, warm and honest, 92 BPM"),
        "lyrics": ("[Verse]\n背起包就走了很多年\n路上的人来了又走\n\n"
                   "[Chorus]\n我还是要回到那条河边\n看看当年写的字"),
    },
    {
        "name": "R&B 慢摇",
        "style": ("English, contemporary R&B, smooth female vocal, electric piano, "
                  "deep sub bass, laid-back groove, sensual night mood, 84 BPM"),
        "lyrics": ("[Verse]\nSlow down, the city's out of sight\nJust us and the dashboard light\n\n"
                   "[Chorus]\nStay with me until the morning\nWe don't need a reason"),
    },
    {
        "name": "中国风",
        "style": ("Mandarin, Chinese traditional style, ethereal female vocal, guzheng, "
                  "dizi flute, erhu, modern orchestral backing, ancient poetic mood, 78 BPM"),
        "lyrics": ("[Verse]\n一盏灯照尽长安雪\n旧年书信未寄出\n\n"
                   "[Chorus]\n山河仍在 故人已远\n只剩这一轮明月"),
    },
    {
        "name": "Lo-fi 放松",
        "style": ("Instrumental, lo-fi hip hop, warm vinyl crackle, mellow electric piano, "
                  "soft boom bap drums, relaxed study and chill mood, 74 BPM"),
        "lyrics": "[Instrumental]",
    },
    {
        "name": "复古迪斯科",
        "style": ("English, 80s disco, powerful female vocal, funky rhythm guitar, "
                  "lush strings, punchy horn section, four-on-the-floor drums, 122 BPM"),
        "lyrics": ("[Verse]\nMirror ball is spinning slow\nEverybody's letting go\n\n"
                   "[Chorus]\nDance with me under the light\nWe can make it through the night"),
    },
    {
        "name": "摇滚民谣",
        "style": ("Mandarin, indie folk rock, raspy male vocal, strummed acoustic guitar, "
                  "building drums, organ, anthemic singalong chorus, 104 BPM"),
        "lyrics": ("[Verse]\n二十岁那年我离开家\n以为世界就在前方\n\n"
                   "[Chorus]\n唱吧 趁我们还年轻\n唱到天亮也不停"),
    },
]


def build_html():
    presets_json = json.dumps(PRESETS, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__PRESETS__", presets_json).replace(
        "__PORT__", str(PORT))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YuE2 音乐生成器</title>
<!-- 内联空图标：不然浏览器会去请求 /favicon.ico，控制台留一条 404 -->
<link rel="icon" href="data:,">
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 28px 20px 60px;
    background: #14161a; color: #e6e8ec;
    font-family: system-ui, -apple-system, "Microsoft YaHei", sans-serif;
    font-size: 15px; line-height: 1.6;
  }
  .wrap { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 6px; }
  .sub { color: #9aa3b2; font-size: 13px; margin-bottom: 22px; }

  .status {
    background: #1c1f26; border: 1px solid #2a2f3a; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 22px; font-size: 13px;
  }
  .status .row { display: flex; gap: 10px; padding: 3px 0; }
  .status .k { color: #9aa3b2; min-width: 74px; }
  .status .v { color: #e6e8ec; }
  .badge {
    display: inline-block; padding: 1px 9px; border-radius: 20px;
    font-size: 12px; font-weight: 500;
  }
  .badge.ok { background: #14361f; color: #5fd08a; }
  .badge.warn { background: #3a2c12; color: #e0a44a; }
  .badge.bad { background: #3d1a18; color: #e87a72; }

  .card {
    background: #1c1f26; border: 1px solid #2a2f3a; border-radius: 10px;
    padding: 18px; margin-bottom: 16px;
  }
  .label { font-size: 14px; font-weight: 500; margin-bottom: 4px; }
  .label .num {
    display: inline-flex; width: 20px; height: 20px; border-radius: 50%;
    background: #4f8cff; color: #fff; font-size: 12px; align-items: center;
    justify-content: center; margin-right: 8px; vertical-align: 1px;
  }
  .hint { color: #9aa3b2; font-size: 12px; margin-bottom: 12px; }

  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .chip {
    padding: 7px 15px; border-radius: 20px; border: 1px solid #2a2f3a;
    background: #22262f; color: #c8ced8; cursor: pointer; font-size: 13px;
    font-family: inherit; transition: all .12s;
  }
  .chip:hover { border-color: #4f8cff; color: #e6e8ec; }
  .chip.on { background: #1b3559; border-color: #4f8cff; color: #9dc2ff; }

  input[type=text], textarea {
    width: 100%; background: #14161a; border: 1px solid #2a2f3a;
    border-radius: 8px; padding: 10px 12px; color: #e6e8ec;
    font-family: inherit; font-size: 14px; resize: vertical;
  }
  input[type=text]:focus, textarea:focus {
    outline: none; border-color: #4f8cff;
  }
  textarea { min-height: 130px; line-height: 1.7; }

  .durations { display: flex; flex-wrap: wrap; gap: 10px; }
  .dur {
    flex: 1 1 90px; padding: 14px 8px; border-radius: 9px; border: 1px solid #2a2f3a;
    background: #22262f; color: #c8ced8; cursor: pointer; font-size: 14px;
    font-family: inherit; transition: all .12s; text-align: center;
  }
  .dur small { display: block; color: #9aa3b2; font-size: 11px; margin-top: 2px; }
  .dur:hover:not(:disabled) { border-color: #4f8cff; }
  .dur.on { background: #1b3559; border-color: #4f8cff; color: #9dc2ff; }
  .dur.on small { color: #7ba6e8; }
  .dur:disabled { opacity: .32; cursor: not-allowed; }

  .batchrow { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    margin: 16px 0 12px; }
  .batchrow .durations { flex: 0 0 auto; }
  .batchrow .dur { flex: 0 0 76px; padding: 9px 6px; font-size: 13px; }
  .batchk { font-size: 13px; color: #9aa3b2; }
  .batchhint { font-size: 12px; color: #6d7787; flex: 1 1 220px; }
  .bitem { border: 1px solid #2a2f3a; border-radius: 10px; padding: 10px 12px;
    margin-bottom: 10px; background: #22262f; }
  .bhead { font-size: 13px; color: #9aa3b2; margin-bottom: 8px; }
  .bitem audio { width: 100%; }

  .go {
    width: 100%; padding: 16px; border: none; border-radius: 10px;
    background: #4f8cff; color: #fff; font-size: 16px; font-weight: 600;
    cursor: pointer; font-family: inherit; transition: background .12s;
  }
  .go:hover:not(:disabled) { background: #3d7ce8; }
  .go:disabled { background: #2a3346; color: #6d7787; cursor: not-allowed; }

  .progress-wrap { display: none; margin-top: 18px; }
  .bar {
    height: 7px; background: #22262f; border-radius: 4px; overflow: hidden;
  }
  .bar > i {
    display: block; height: 100%; width: 0; background: #4f8cff;
    border-radius: 4px; transition: width .35s ease;
  }
  .pmsg { margin-top: 9px; font-size: 13px; color: #9aa3b2; }
  .pmsg b { color: #e6e8ec; font-weight: 500; }

  .result { display: none; margin-top: 18px; }
  audio { width: 100%; margin-top: 10px; }
  .dl {
    display: inline-block; margin-top: 12px; padding: 9px 18px;
    background: #22262f; border: 1px solid #2a2f3a; border-radius: 8px;
    color: #c8ced8; text-decoration: none; font-size: 13px;
  }
  .dl:hover { border-color: #4f8cff; color: #e6e8ec; }

  .err {
    display: none; margin-top: 16px; padding: 14px 16px;
    background: #2a1a19; border: 1px solid #4a2523; border-radius: 9px;
    color: #e8a29a; font-size: 13px;
  }
  .err code {
    display: block; margin-top: 9px; padding: 9px; background: #14161a;
    border-radius: 6px; color: #c8ced8; font-size: 12px;
    white-space: pre-wrap; word-break: break-all;
    font-family: ui-monospace, Consolas, monospace;
  }
  .meta { color: #9aa3b2; font-size: 12px; margin-top: 10px; }
  .keepnote { color: #7d8694; font-size: 11px; margin-top: 8px; line-height: 1.6; }
  .scorebox { margin-top: 14px; border-top: 1px solid #262b33; padding-top: 12px; }
  pre.abc { background: #15181e; border: 1px solid #262b33; border-radius: 6px;
            color: #9dc2ff; font-size: 12px; line-height: 1.6; padding: 10px 12px;
            margin: 10px 0 8px; max-height: 260px; overflow: auto;
            white-space: pre-wrap; word-break: break-all; font-family: ui-monospace,
            SFMono-Regular, Consolas, monospace; }
  .hrow { display: flex; align-items: center; gap: 10px; padding: 9px 0;
          border-bottom: 1px solid #262b33; font-size: 13px; }
  .hrow:last-child { border-bottom: none; }
  .htime { color: #e6e8ec; min-width: 84px; }
  .hwhen { color: #7d8694; flex: 1; font-size: 12px; }
  .hbtn { background: none; border: 1px solid #3a4049; color: #9aa3b2;
          border-radius: 6px; padding: 4px 11px; font-size: 12px;
          cursor: pointer; text-decoration: none; font-family: inherit; }
  .hbtn:hover { border-color: #4f8cff; color: #e6e8ec; }

  /* 风格组合器 */
  .builder { margin-top: 14px; border-top: 1px solid #262b33; padding-top: 14px; }
  .brow { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
  .bk { color: #9aa3b2; font-size: 12px; min-width: 58px; flex-shrink: 0; }
  .bchips { display: flex; flex-wrap: wrap; gap: 6px; flex: 1; }
  .bchip { background: #1c2028; border: 1px solid #2a2f3a; color: #9aa3b2;
           border-radius: 6px; padding: 4px 10px; font-size: 12px;
           cursor: pointer; font-family: inherit; }
  .bchip:hover { border-color: #4f8cff; color: #e6e8ec; }
  .bchip.on { background: #1b3559; border-color: #4f8cff; color: #9dc2ff; }
  .bval { color: #7d8694; font-size: 12px; min-width: 62px; text-align: right; }
  select { background: #1c2028; border: 1px solid #2a2f3a; color: #c8ced8;
           border-radius: 6px; padding: 5px 8px; font-size: 12px;
           font-family: inherit; flex: 1; }
  input[type=range] { flex: 1; accent-color: #4f8cff; min-width: 80px; }
  .stylefoot { display: flex; align-items: center; gap: 10px; margin: 14px 0 6px; }
  .mini { background: none; border: 1px solid #3a4049; color: #9aa3b2;
          border-radius: 5px; padding: 3px 9px; font-size: 11px;
          cursor: pointer; font-family: inherit; }
  .mini:hover { border-color: #4f8cff; color: #e6e8ec; }

  /* 高级设置 */
  .advtoggle { background: none; border: none; color: #9aa3b2; font-size: 13px;
               cursor: pointer; padding: 0; font-family: inherit; }
  .advtoggle:hover { color: #e6e8ec; }
  .arow { margin-top: 16px; }
  .ak { color: #c8ced8; font-size: 12px; display: block; margin-bottom: 7px; }
  .ak b { color: #9dc2ff; font-weight: 500; }
  .anote { color: #6b7482; font-size: 11px; margin-top: 6px; line-height: 1.55; }
</style>
</head>
<body>
<div class="wrap">
  <h1>YuE2 音乐生成器</h1>
  <div class="sub">输入风格和歌词，一键生成带人声的歌曲</div>

  <div class="status" id="status"></div>

  <div class="card">
    <div class="label"><span class="num">1</span>选择风格</div>
    <div class="hint">点预设一键填入，或者用下面的积木自己搭一套。英文描述效果更稳。</div>
    <div class="chips" id="presets"></div>

    <div class="builder">
      <div class="brow"><span class="bk">语言</span><div class="bchips" id="b-lang"></div></div>
      <div class="brow"><span class="bk">流派</span><select id="b-genre"></select></div>
      <div class="brow"><span class="bk">人声</span><div class="bchips" id="b-vocal"></div></div>
      <div class="brow"><span class="bk">情绪</span><div class="bchips" id="b-mood"></div></div>
<div class="brow"><span class="bk">乐器</span><div class="bchips" id="b-inst"></div></div>
<div class="brow"><span class="bk">合奏编制</span><select id="b-ensemble"></select></div>
<div class="brow"><span class="bk">调性</span><select id="b-key"></select></div>
<div class="brow"><span class="bk">拍号</span><select id="b-meter"></select></div>
      <div class="brow">
        <span class="bk">速度</span>
        <input type="range" id="b-bpm" min="60" max="180" step="2" value="90">
        <span class="bval" id="b-bpmval">90 BPM</span>
      </div>
    </div>

    <div class="stylefoot">
      <span class="bk">最终描述</span>
      <button class="mini" id="b-reset">清空重来</button>
    </div>
    <textarea id="style" rows="2" style="min-height:58px"
      placeholder="上面的选择会自动拼到这里，也可以直接手改"></textarea>
  </div>

  <div class="card">
    <div class="label"><span class="num">2</span>填写歌词</div>
    <div class="hint">用 [Verse] [Chorus] [Bridge] 标记段落，换行会被当作断句。
      <b>歌词写几段，歌就唱多长</b>——一段大约 15 秒，想要 3 分钟的歌就写十来段。
      下面的时长只是上限，歌词短了会提前收尾。</div>
    <textarea id="lyrics"></textarea>
    <div class="meta" id="lyricinfo"></div>
  </div>

  <div class="card">
    <div class="label"><span class="num">3</span>规划方式</div>
    <div class="hint">决定模型是「先写谱再唱」还是「直接开唱」。这是官方演示页里的
      With symbolic planning / Direct generation 那一栏。</div>
    <div class="chips" id="plan"></div>
    <div class="meta" id="planhint"></div>
  </div>

  <div class="card">
    <div class="label"><span class="num">4</span>创作风格</div>
    <div class="hint">决定旋律有多「敢想」。稳妥适合抒情，大胆容易出惊喜，狂野可能跑偏。</div>
    <div class="chips" id="creative"></div>
    <div class="meta" id="creativehint"></div>
  </div>

  <div class="card">
    <div class="label"><span class="num">5</span>选择时长上限</div>
    <div class="hint" id="durhint"></div>
    <div class="durations" id="durations"></div>
  </div>

  <div class="card">
    <button class="advtoggle" id="advtoggle">▸ 高级设置（想精调再展开）</button>
    <div id="adv" style="display:none">
      <div class="arow">
        <span class="ak">风格贴合度 · <b id="a-cfgval">自动</b></span>
        <input type="range" id="a-cfg" min="0" max="20" step="0.5" value="0">
        <div class="anote">越大越贴着风格描述走，越小越自由发挥。拉到最左是「自动」，用官方默认值。</div>
      </div>
      <div class="arow">
        <span class="ak">重复抑制 · <b id="a-repval">1.20</b></span>
        <input type="range" id="a-rep" min="1" max="1.6" step="0.01" value="1.2">
        <div class="anote">旋律老是重复就调大一点；调太大会让旋律变生硬。</div>
      </div>
      <div class="arow">
        <span class="ak">候选范围 top_k · <b id="a-topkval">100</b></span>
        <input type="range" id="a-topk" min="1" max="300" step="1" value="100">
        <div class="anote">每步只从概率最高的前 N 个音里挑。调小更聚拢、更稳，调大更发散。</div>
      </div>
      <div class="arow">
        <span class="ak">重复回头看 penalty_window · <b id="a-pwinval">50</b></span>
        <input type="range" id="a-pwin" min="1" max="100" step="1" value="50">
        <div class="anote">重复抑制往前看多少个音。调小只管最近几句，调大管整首。</div>
      </div>
      <div class="arow">
        <span class="ak">合成精细度</span>
        <div class="bchips" id="a-ode"></div>
        <div class="anote">32 是官方默认。48 更细腻但更慢，16 更快但粗糙。</div>
      </div>
      <div class="arow">
        <span class="ak">随机种子</span>
        <input type="text" id="a-seed" placeholder="留空 = 每次随机">
        <div class="anote">填一个数字就能复现同一首歌（其它设置也要保持一致）。</div>
      </div>
      <div class="arow">
        <span class="ak">外部乐谱（ABC 记谱）</span>
        <textarea id="a-abc" rows="3" style="min-height:76px"
          placeholder="粘贴 ABC 记谱，模型会照它编曲。不懂就别填。"></textarea>
        <div class="anote">需要懂 ABC 记谱法。填了之后会自动切到「完整」规划模式。</div>
      </div>
    </div>
  </div>

  <div class="batchrow">
    <span class="batchk">一次出</span>
    <div class="durations" id="batch"></div>
    <span class="batchhint" id="batchhint"></span>
  </div>

  <button class="go" id="go">生成歌曲</button>

  <div class="progress-wrap" id="pw">
    <div class="bar"><i id="bar"></i></div>
    <div class="pmsg" id="pmsg"></div>
  </div>

  <div class="err" id="err"></div>

  <div class="result" id="result">
    <div class="label" id="resultlabel">生成结果</div>
    <div id="singlewrap">
      <audio id="player" controls></audio>
      <a class="dl" id="dl" download>下载音频文件</a>
      <a class="dl" id="dlmp3" download style="display:none;margin-left:8px">下载 MP3（方便分享）</a>
      <div class="meta" id="meta"></div>
    </div>
    <div id="batchlist" style="display:none"></div>
    <div class="keepnote">想长期保存就点上面的「下载音频文件」。程序只保留最近 20 首，
      更早的会自动清理，免得一直占硬盘。</div>

    <div class="scorebox" id="scorebox" style="display:none">
      <button class="advtoggle" id="scoretoggle">▸ 看看模型给这首歌写的谱</button>
      <div id="scorewrap" style="display:none">
        <pre class="abc" id="scoretext"></pre>
        <div class="stylefoot">
          <button class="mini" id="scorecopy">复制乐谱</button>
          <button class="mini" id="scoreuse">拿去当外部乐谱</button>
        </div>
        <div class="anote">改几个音、或者只留喜欢的乐句，再填进高级设置里的
          「外部乐谱」，就能在同一个骨架上反复打磨。</div>
      </div>
    </div>
  </div>

  <div class="card" id="historycard" style="display:none">
    <div class="label">历史作品</div>
    <div class="hint">最近生成的作品。点「播放」直接重听，点「下载」存到自己电脑上。</div>
    <div id="history"></div>
  </div>
</div>

<script>
const PRESETS = __PRESETS__;
let hw = {}, prof = {}, durations = [], picked = null, polling = null;
let mp3ok = false;

// ---- 风格组合器的积木 ----
// 流派取自官方演示页枚举出的标签（map-yue2.github.io），再补几个日常常用的。
const GENRES = ['', 'Pop', 'Ballad', 'Rock', 'Indie Rock', 'Folk', 'Country',
  'Hip Hop', 'R&B', 'Soul', 'Funk', 'Lo-fi', 'City Pop', 'J-Pop',
  'Ambient', 'Dark Ambient', 'Electropop', 'Dance-Pop', 'Electronic Dance Music',
  'Alternative Dance', 'Eurobeat', 'Italo-Disco', 'Nu-Disco', 'Synthwave',
  'Cool Jazz', 'Afro-Jazz', 'Ethio-Jazz', 'Jazz-Funk', 'Big Band', 'Swing',
  'Boogie Woogie', 'Jump Blues', 'Dixieland', 'Soul Blues', 'Bossa Nova',
  'Metal', 'Glam Metal', 'Cyber Metal', 'Industrial Metal', 'Emo',
  'Classical Music', 'Easy Listening', 'Barbershop', 'Flamenco', 'Bachata',
  'Bluegrass', 'Southern Gospel', 'Southern Soul', 'Grime', 'Folktronica'];

const LANGS = [
  { v: 'Mandarin', label: '中文' },
  { v: 'Cantonese', label: '粤语' },
  { v: 'English', label: '英文' },
  { v: 'Japanese', label: '日文' },
  { v: 'Korean', label: '韩文' },
  { v: 'Instrumental', label: '纯音乐' },
];
const VOCALS = [
  { v: 'warm expressive female vocal', label: '女声' },
  { v: 'gritty male lead vocal', label: '男声' },
  { v: 'layered choir harmonies', label: '合唱' },
  { v: 'no vocals, purely instrumental', label: '纯器乐' },
];
const MOODS = [
  { v: 'emotional and warm', label: '温暖' },
  { v: 'melancholic and wistful', label: '忧郁' },
  { v: 'upbeat and joyful', label: '欢快' },
  { v: 'powerful and anthemic', label: '激昂' },
  { v: 'dreamy and atmospheric', label: '梦幻' },
  { v: 'nostalgic and bittersweet', label: '怀旧' },
  { v: 'dark and intense', label: '暗黑' },
];
const MAX_INST = 4;   // 最多叠几件乐器，再多描述会糊
const INSTS = [
  { v: 'piano', label: '钢琴' },
  { v: 'acoustic guitar', label: '木吉他' },
  { v: 'electric guitar', label: '电吉他' },
  { v: 'strings', label: '弦乐' },
  { v: 'synthesizer', label: '合成器' },
  { v: 'tenor saxophone', label: '萨克斯' },
  { v: 'trumpet', label: '小号' },
  { v: 'flute', label: '长笛' },
  { v: 'drum machine', label: '鼓机' },
  { v: 'bass', label: '贝斯' },
  { v: 'violin', label: '小提琴' },
  { v: 'cello', label: '大提琴' },
  { v: 'harp', label: '竖琴' },
  { v: 'organ', label: '管风琴' },
  { v: 'electric piano', label: '电钢琴' },
  { v: 'acoustic drums', label: '原声鼓' },
  { v: 'percussion', label: '打击乐' },
  { v: 'banjo', label: '班卓琴' },
  { v: 'guqin', label: '古琴' },
  { v: 'guzheng', label: '古筝' },
  { v: 'pipa', label: '琵琶' },
  { v: 'erhu', label: '二胡' },
  { v: 'dizi', label: '竹笛' },
  { v: 'xiao', label: '洞箫' },
  { v: 'sheng', label: '笙' },
  { v: 'ruan', label: '阮' },
];

// 调性 / 拍号：演示页的 ABC 里就有 K（调性）和 M（拍号），这俩最影响音乐性格。
// 不指定就交给模型自己定；指定了会写进风格描述去偏置生成。
const KEYS = ['', 'C major', 'G major', 'D major', 'A major', 'E major', 'F major',
  'A minor', 'E minor', 'D minor', 'B minor', 'C minor', 'F# minor'];
const METERS = ['', '4/4', '3/4', '6/8', '2/4', '5/4', '7/8'];

// 合奏编制：一键选择常见中国风合奏组合
const ENSEMBLES = [
  { v: '', label: '（不指定）' },
  { v: 'guqin+xiao', label: '琴箫合奏' },
  { v: 'guzheng+dizi', label: '筝笛合奏' },
  { v: 'guzheng+pipa', label: '筝琵合奏' },
  { v: 'guqin+guzheng+pipa+dizi', label: '丝竹乐合奏' },
];

// 创作风格：把采样温度包装成人话。temperature 越高越发散。
const CREATIVE = [
  { v: 'safe', label: '稳妥', temp: 0.8, top_p: 0.90,
    note: '旋律保守、不跑调，适合抒情和翻唱向的作品。' },
  { v: 'normal', label: '均衡', temp: 1.0, top_p: 0.95,
    note: '官方默认档，各方面比较平衡。' },
  { v: 'bold', label: '大胆', temp: 1.25, top_p: 0.97,
    note: '旋律更跳、更有惊喜，偶尔会有点意外。' },
  { v: 'wild', label: '狂野', temp: 1.5, top_p: 0.99,
    note: '高度发散，适合找灵感，也更容易跑偏。' },
];
const ODE = [
  { v: 16, label: '16 · 快' },
  { v: 32, label: '32 · 默认' },
  { v: 48, label: '48 · 精细' },
];

// 规划方式 = 官方的 cot（chain-of-thought）。先写谱再唱，谱会约束旋律和和弦。
const PLANS = [
  { v: 'full', label: '完整规划', cot: 'full',
    note: '先生成带和弦的乐谱，再照着唱。结构最完整、最贴合风格，但最慢。' },
  { v: 'melody', label: '旋律规划', cot: 'melody',
    note: '只规划旋律线、不带和弦。速度居中，旋律更连贯。' },
  { v: 'off', label: '直接生成', cot: 'off',
    note: '跳过写谱，模型直接出音乐令牌。最快，结构最松散，适合找灵感。' },
];

let sel = { lang: 'Mandarin', genre: '', vocal: '', mood: '', inst: [], bpm: 90,
            key: '', meter: '', ensemble: '' };
let creative = 'normal';
let planMode = 'profile';   // 'profile' = 跟随硬件档位默认
let odeSteps = 32;
let styleAuto = true;      // 最终描述是否由组合器自动生成

function el(id) { return document.getElementById(id); }

async function boot() {
  const r = await fetch('/api/status');
  const d = await r.json();
  hw = d.hardware; prof = d.profile; mp3ok = !!d.mp3;
  renderStatus(d);
  renderPresets();
  renderBuilder();
  renderCreative();
  renderPlan();
  renderAdvanced();
  renderDurations();
  renderBatchChoices();
  setupScoreUI();
  applyPreset(0);
  if (d.job && d.job.status === 'running') startPolling();
  if (d.job && d.job.status === 'done') showResult(d.job);
  loadHistory();
}

function renderStatus(d) {
  const s = el('status');
  if (!d.installed) {
    s.innerHTML = '<div class="row"><span class="k">状态</span>'
      + '<span class="v"><span class="badge bad">未安装</span> '
      + '请先双击运行 install.bat 安装环境和模型</span></div>';
    el('go').disabled = true;
    return;
  }
  const vram = hw.vram_gib ? hw.vram_gib.toFixed(1) + ' GB' : '未知';
  const cls = prof.disabled ? 'bad' : 'ok';
  let rows =
    '<div class="row"><span class="k">显卡</span><span class="v">'
      + (hw.name || '未检测到') + '</span></div>'
    + '<div class="row"><span class="k">显存</span><span class="v">' + vram
      + '</span></div>';
  if (hw.ram_total_gb)
    rows += '<div class="row"><span class="k">内存</span><span class="v">'
      + hw.ram_total_gb.toFixed(1) + ' GB，当前可用 '
      + hw.ram_free_gb.toFixed(1) + ' GB</span></div>';
  rows +=
    '<div class="row"><span class="k">运行模式</span><span class="v">'
      + '<span class="badge ' + cls + '">' + prof.label + '</span></span></div>'
    + '<div class="row"><span class="k">说明</span><span class="v">'
      + prof.note + '</span></div>';
  if (hw.ram_free_gb && hw.ram_free_gb < 3.5)
    rows += '<div class="row"><span class="k">提示</span><span class="v">'
      + '内存偏紧，生成前关掉浏览器和聊天软件会稳很多。</span></div>';
  s.innerHTML = rows;
  if (prof.disabled) el('go').disabled = true;
}

function renderPresets() {
  const box = el('presets');
  box.innerHTML = '';
  PRESETS.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = p.name;
    b.onclick = () => applyPreset(i);
    box.appendChild(b);
  });
}

// ---- 风格组合器 ----
function renderChips(id, list, key, multi) {
  const box = el(id);
  box.innerHTML = '';
  list.forEach(item => {
    const b = document.createElement('button');
    b.className = 'bchip';
    b.textContent = item.label;
    b.dataset.v = item.v;
    b.onclick = () => {
      if (multi) {
        const i = sel[key].indexOf(item.v);
        if (i >= 0) sel[key].splice(i, 1);
        else if (sel[key].length < MAX_INST) sel[key].push(item.v);
        else return;                       // 最多 MAX_INST 件乐器，再多描述会糊
      } else {
        sel[key] = (sel[key] === item.v) ? '' : item.v;
      }
      markChips(id, multi ? sel[key] : [sel[key]]);
      styleAuto = true;
      syncStyle();
    };
    box.appendChild(b);
  });
}

function markChips(id, values) {
  el(id).querySelectorAll('.bchip').forEach(b =>
    b.classList.toggle('on', values.indexOf(b.dataset.v) >= 0));
}

function renderBuilder() {
  renderChips('b-lang', LANGS, 'lang', false);
  renderChips('b-vocal', VOCALS, 'vocal', false);
  renderChips('b-mood', MOODS, 'mood', false);
  renderChips('b-inst', INSTS, 'inst', true);

  const g = el('b-genre');
  g.innerHTML = GENRES.map(x => '<option value="' + x + '">'
    + (x || '（不指定）') + '</option>').join('');
  g.onchange = () => { sel.genre = g.value; styleAuto = true; syncStyle(); };

  const bpm = el('b-bpm');
  bpm.oninput = () => {
    sel.bpm = +bpm.value;
    el('b-bpmval').textContent = sel.bpm + ' BPM';
    styleAuto = true;
    syncStyle();
  };

  el('b-reset').onclick = () => {
    sel = { lang: 'Mandarin', genre: '', vocal: '', mood: '', inst: [], bpm: 90,
            key: '', meter: '' };
    g.value = '';
    bpm.value = 90;
    el('b-bpmval').textContent = '90 BPM';
    el('b-key').value = '';
    el('b-meter').value = '';
    markChips('b-lang', [sel.lang]);
    markChips('b-vocal', []);
    markChips('b-mood', []);
    markChips('b-inst', []);
    styleAuto = true;
    syncStyle();
  };

const ksel = el('b-key');
ksel.innerHTML = KEYS.map(x => '<option value="' + x + '">'
  + (x || '（不指定）') + '</option>').join('');
ksel.onchange = () => { sel.key = ksel.value; styleAuto = true; syncStyle(); };
const msel = el('b-meter');
msel.innerHTML = METERS.map(x => '<option value="' + x + '">'
  + (x || '（不指定）') + '</option>').join('');
msel.onchange = () => { sel.meter = msel.value; styleAuto = true; syncStyle(); };

// 合奏编制下拉框
const esel = el('b-ensemble');
esel.innerHTML = ENSEMBLES.map(x => '<option value="' + x.v + '">'
  + x.label + '</option>').join('');
esel.onchange = () => {
  sel.ensemble = esel.value;
  // 选择合奏时自动填充乐器
  if (sel.ensemble) {
    // 清空现有乐器选择
    sel.inst = [];
    // 根据合奏类型添加乐器
    if (sel.ensemble === 'guqin+xiao') {
      sel.inst = ['guqin', 'xiao'];
    } else if (sel.ensemble === 'guzheng+dizi') {
      sel.inst = ['guzheng', 'dizi'];
    } else if (sel.ensemble === 'guzheng+pipa') {
      sel.inst = ['guzheng', 'pipa'];
    } else if (sel.ensemble === 'guqin+guzheng+pipa+dizi') {
      sel.inst = ['guqin', 'guzheng', 'pipa', 'dizi'];
    }
    // 只需更新选中态——DOM 已经渲染过了，用 renderChips 重建反而会丢选中
    markChips('b-inst', sel.inst);
  }
  styleAuto = true;
  syncStyle();
};

  markChips('b-lang', [sel.lang]);
  bpm.value = sel.bpm;
  el('b-bpmval').textContent = sel.bpm + ' BPM';
}

function buildStyle() {
  const p = [];
  if (sel.lang) p.push(sel.lang);
  if (sel.genre) p.push(sel.genre);
  if (sel.mood) p.push(sel.mood);
  if (sel.vocal) p.push(sel.vocal);
  if (sel.inst.length) p.push(sel.inst.join(', '));
if (sel.bpm) p.push(sel.bpm + ' BPM');
if (sel.key) p.push('in the key of ' + sel.key);
if (sel.meter) p.push('in ' + sel.meter + ' time');
if (sel.ensemble) p.push('with ' + sel.ensemble.split('+').join(' and '));
return p.join(', ');
}

// 只在用户没手动改过描述时才自动覆盖，免得把他写的冲掉
function syncStyle() {
  if (styleAuto) el('style').value = buildStyle();
}

// ---- 创作风格 ----
function renderCreative() {
  const box = el('creative');
  box.innerHTML = '';
  CREATIVE.forEach(c => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = c.label;
    b.onclick = () => { creative = c.v; markCreative(); showCreativeNote(); };
    box.appendChild(b);
  });
  markCreative();
  showCreativeNote();
}

function markCreative() {
  el('creative').querySelectorAll('.chip').forEach((b, i) =>
    b.classList.toggle('on', CREATIVE[i].v === creative));
}

function showCreativeNote() {
  const c = CREATIVE.find(x => x.v === creative) || CREATIVE[1];
  el('creativehint').textContent =
    c.note + '（temperature ' + c.temp + ' / top_p ' + c.top_p + '）';
}

// ---- 规划方式 ----
function renderPlan() {
  const box = el('plan');
  box.innerHTML = '';

  // 第一项是「跟随默认」，选中时用硬件档位自带的 cot
  const list = [{ v: 'profile', label: '跟随默认' }].concat(PLANS);
  list.forEach(p => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = p.label;
    b.dataset.v = p.v;
    b.onclick = () => { planMode = p.v; markPlan(); showPlanNote(); };
    box.appendChild(b);
  });
  markPlan();
  showPlanNote();
}

function markPlan() {
  el('plan').querySelectorAll('.chip').forEach(b =>
    b.classList.toggle('on', b.dataset.v === planMode));
}

function showPlanNote() {
  const hint = el('planhint');
  const abc = el('a-abc') ? el('a-abc').value.trim() : '';
  // 有外部乐谱就不能「直接生成」，否则模型没地方放那段谱
  const effective = planMode === 'profile' ? (prof.cot || 'full') : planMode;
  const forced = abc && effective === 'off';

  if (forced) {
    hint.textContent = '你填了外部乐谱，所以这次会自动按「完整规划」跑'
      + '（直接生成没法读谱）。';
    return;
  }
  if (planMode === 'profile') {
    const names = { full: '完整规划', melody: '旋律规划', off: '直接生成' };
    hint.textContent = '按这台机器的档位自动选（当前是「'
      + (names[prof.cot] || '完整规划') + '」）。不折腾就选这个。';
    return;
  }
  const p = PLANS.find(x => x.v === planMode);
  hint.textContent = p.note;
}

// ---- 高级设置 ----
function renderAdvanced() {
  const box = el('a-ode');
  box.innerHTML = '';
  ODE.forEach(o => {
    const b = document.createElement('button');
    b.className = 'bchip';
    b.textContent = o.label;
    b.dataset.v = o.v;
    b.onclick = () => {
      odeSteps = o.v;
      box.querySelectorAll('.bchip').forEach(x =>
        x.classList.toggle('on', +x.dataset.v === odeSteps));
    };
    box.appendChild(b);
  });
  box.querySelectorAll('.bchip').forEach(x =>
    x.classList.toggle('on', +x.dataset.v === odeSteps));

  el('advtoggle').onclick = () => {
    const open = el('adv').style.display === 'block';
    el('adv').style.display = open ? 'none' : 'block';
    el('advtoggle').textContent = (open ? '▸' : '▾')
      + ' 高级设置（想精调再展开）';
  };

  const cfg = el('a-cfg');
  cfg.oninput = () => {
    const v = +cfg.value;
    el('a-cfgval').textContent = v > 0 ? v.toFixed(1) : '自动';
  };
  const rep = el('a-rep');
  rep.oninput = () => { el('a-repval').textContent = (+rep.value).toFixed(2); };
  const topk = el('a-topk');
  topk.oninput = () => { el('a-topkval').textContent = (+topk.value); };
  const pwin = el('a-pwin');
  pwin.oninput = () => { el('a-pwinval').textContent = (+pwin.value); };
}

function collectOptions() {
  const c = CREATIVE.find(x => x.v === creative) || CREATIVE[1];
  const cfg = +el('a-cfg').value;
  const abc = el('a-abc').value.trim() || null;

  // cot 交给后端裁决：选了「跟随默认」就传 null，
  // 填了外部乐谱又不许写谱时后端会自动抬到 full
  let cot = null;
  if (planMode !== 'profile') {
    const p = PLANS.find(x => x.v === planMode);
    cot = p ? p.cot : null;
  }
  if (abc && cot === 'off') cot = 'full';   // 有谱就没法直接生成

  return {
    temperature: c.temp,
    top_p: c.top_p,
    repetition_penalty: +el('a-rep').value,
    top_k: +el('a-topk').value,
    penalty_window: +el('a-pwin').value,
    cfg_scale: cfg > 0 ? cfg : null,
    ode_steps: odeSteps,
    cot: cot,
    abc: abc,
  };
}

function applyPreset(i) {
  document.querySelectorAll('#presets .chip').forEach((c, j) =>
    c.classList.toggle('on', i === j));
  styleAuto = false;        // 预设是完整描述，别被组合器覆盖
  el('style').value = PRESETS[i].style;
  el('lyrics').value = PRESETS[i].lyrics;
el('b-key').value = '';
el('b-meter').value = '';
el('b-ensemble').value = '';
sel.key = '';
sel.meter = '';
sel.ensemble = '';
  updateLyricInfo();
}

// 实测：一段歌词大约唱 15 秒，歌曲长度由歌词长度决定，时长选项只是上限
function updateLyricInfo() {
  const text = el('lyrics').value;
  const sections = (text.match(/^\s*\[[^\]]+\]\s*$/gm) || []).length;
  const lines = text.split('\n').filter(l => l.trim() && !/^\s*\[/.test(l)).length;
  if (sections) {
    el('lyricinfo').textContent =
      '识别到 ' + sections + ' 个段落，预计能唱约 ' + (sections * 15) + ' 秒';
  } else if (lines) {
    el('lyricinfo').textContent =
      '没有段落标记，按 ' + lines + ' 行估算约 ' + (lines * 5) + ' 秒';
  } else {
    el('lyricinfo').textContent = '';
  }
}

function renderDurations() {
  const cap = prof.max_seconds || 0;
  durations = [
    { s: 10, label: '试跑', sub: '约 10 秒' },
    { s: 30, label: '短', sub: '约 30 秒' },
    { s: 60, label: '中', sub: '约 1 分钟' },
    { s: 120, label: '长', sub: '约 2 分钟' },
    { s: 180, label: '完整', sub: '约 3 分钟' },
    { s: 240, label: '超长', sub: '约 4 分钟' },
  ];
  const box = el('durations');
  box.innerHTML = '';
  durations.forEach((d, i) => {
    const b = document.createElement('button');
    b.className = 'dur';
    b.innerHTML = d.label + '<small>' + d.sub + '</small>';
    b.disabled = d.s > cap;
    b.onclick = () => { picked = d.s; markDur(i); };
    box.appendChild(b);
  });
  const first = durations.findIndex(d => d.s <= cap);
  if (first >= 0) { picked = durations[first].s; markDur(first); }
  el('durhint').textContent = cap
    ? '这台机器最长支持约 ' + cap + ' 秒。灰色选项超出硬件能力。'
      + '时长越长等待越久，第一次建议先用「试跑」确认能出声。'
      + '超过 3 分钟的歌内存会很紧，生成前先关掉浏览器和聊天软件。'
    : '当前硬件无法生成。';
}

function markDur(i) {
  document.querySelectorAll('#durations .dur').forEach((b, j) =>
    b.classList.toggle('on', i === j));
}

// 一次出几首。采样随机性大，出 4 首挑 1 首比反复重抽划算得多
const BATCHES = [
  { v: 1, label: '1 首', sub: '标准' },
  { v: 2, label: '2 首', sub: '比一比' },
  { v: 4, label: '4 首', sub: '慢慢挑' },
];
let batchCount = 1;

function renderBatchChoices() {
  const box = el('batch');
  box.innerHTML = '';
  BATCHES.forEach((b, i) => {
    const btn = document.createElement('button');
    btn.className = 'dur';
    btn.innerHTML = b.label + '<small>' + b.sub + '</small>';
    btn.onclick = () => markBatch(i);
    box.appendChild(btn);
  });
  markBatch(0);
}

function markBatch(i) {
  document.querySelectorAll('#batch .dur').forEach((b, j) =>
    b.classList.toggle('on', i === j));
  batchCount = BATCHES[i].v;
  el('batchhint').textContent = batchCount === 1
    ? '出 1 首。想多试几个方向就往上调。'
    : '一次出 ' + batchCount + ' 首：参数完全一样，只有种子不同。'
      + '总耗时大约是单首的 ' + batchCount + ' 倍。';
}

el('lyrics').addEventListener('input', updateLyricInfo);
el('style').addEventListener('input', () => { styleAuto = false; });
el('a-abc').addEventListener('input', showPlanNote);

el('go').onclick = async () => {
  const style = el('style').value.trim();
  const lyrics = el('lyrics').value.trim();
  if (!style || !lyrics) { alert('请先填写风格和歌词'); return; }
  if (!picked) { alert('请选择时长'); return; }

  // 种子留空 = 后端每次随机；填了就必须是个像样的整数
  const seedRaw = el('a-seed').value.trim();
  let seed = null;
  if (seedRaw) {
    seed = Number(seedRaw);
    if (!Number.isInteger(seed) || seed < 0) {
      alert('随机种子要填一个非负整数，或者留空');
      return;
    }
  }

  el('go').disabled = true;
  el('go').textContent = '生成中…';
  el('err').style.display = 'none';
  el('result').style.display = 'none';
  el('pw').style.display = 'block';
  el('bar').style.width = '1%';
  el('pmsg').innerHTML = '<b>准备中</b>';

  await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ style, lyrics, seconds: picked, count: batchCount,
                           seed, options: collectOptions() })
  });
  startPolling();
};

function startPolling() {
  if (polling) return;
  polling = setInterval(tick, 1000);
  tick();
}

async function tick() {
  const r = await fetch('/api/job');
  const j = await r.json();
  el('pw').style.display = 'block';
  el('bar').style.width = Math.max(1, j.progress) + '%';
  el('pmsg').innerHTML = '<b>' + (j.stage || '') + '</b> ' + (j.message || '');

  if (j.status === 'done') {
    clearInterval(polling); polling = null;
    el('go').disabled = false;
    el('go').textContent = '再生成一首';
    showResult(j);
  } else if (j.status === 'error') {
    clearInterval(polling); polling = null;
    el('go').disabled = false;
    el('go').textContent = '重试';
    el('pw').style.display = 'none';
    el('err').style.display = 'block';
    el('err').innerHTML = j.message
      + '<code>' + (j.error || '') + '</code>';
  }
}

function showResult(j) {
  el('pw').style.display = 'none';
  el('result').style.display = 'block';
  const list = (j.batch_done && j.batch_done.length > 1) ? j.batch_done : null;
  if (list) renderBatchResult(list);
  else renderSingleResult(j);
  setupScore(j);
  loadHistory();
}

function renderSingleResult(j) {
  el('singlewrap').style.display = 'block';
  el('batchlist').style.display = 'none';
  el('batchlist').innerHTML = '';   // 清掉上一批的播放器，别留在 DOM 里
  el('resultlabel').textContent = '生成结果';
  const p = el('player');
  p.src = j.audio_url + '&t=' + Date.now();
  el('dl').href = j.audio_url + '&download=1';
  el('dl').download = j.audio_name || 'yue2.flac';
  if (mp3ok) {
    el('dlmp3').style.display = 'inline-block';
    el('dlmp3').href = j.audio_url + '&download=1&format=mp3';
    el('dlmp3').download = (j.audio_name || 'yue2.flac').replace('.flac', '.mp3');
  }
  let m = [];
  if (j.seconds) m.push('时长 ' + j.seconds.toFixed(1) + ' 秒');
  if (j.elapsed) m.push('耗时 ' + j.elapsed.toFixed(0) + ' 秒');
  if (j.peak_vram) m.push('峰值显存 ' + j.peak_vram.toFixed(2) + ' GB');
  if (j.seed !== null && j.seed !== undefined) m.push('种子 ' + j.seed);
  if (j.truncated && (j.truncated.abc || j.truncated.semantic))
    m.push('注意：达到长度上限，歌曲可能被截断');
  el('meta').textContent = m.join(' · ');
}

// 批量结果：一次列全，挨个听、挑一首。音频懒加载，不点播放不占带宽
function renderBatchResult(list) {
  el('singlewrap').style.display = 'none';
  const box = el('batchlist');
  box.style.display = 'block';
  el('resultlabel').textContent = '生成结果 · ' + list.length + ' 首，挑一首满意的';
  box.innerHTML = '';
  list.forEach((it, i) => {
    const d = document.createElement('div');
    d.className = 'bitem';
    const head = document.createElement('div');
    head.className = 'bhead';
    let t = '第 ' + (i + 1) + ' 首';
    if (it.seconds) t += ' · ' + it.seconds.toFixed(1) + ' 秒';
    if (it.elapsed) t += ' · 耗时 ' + it.elapsed.toFixed(0) + ' 秒';
    if (it.seed !== null && it.seed !== undefined) t += ' · 种子 ' + it.seed;
    head.textContent = t;
    const au = document.createElement('audio');
    au.controls = true;
    au.preload = 'none';
    au.src = it.audio_url + '&t=' + Date.now();
    const a = document.createElement('a');
    a.className = 'dl';
    a.download = it.audio_name || 'yue2.flac';
    a.href = it.audio_url + '&download=1';
    a.textContent = '下载这首';
    d.appendChild(head);
    d.appendChild(au);
    d.appendChild(a);
    box.appendChild(d);
  });
}

// ---- 生成结果里的乐谱 ----
let scoreDir = null, scoreLoaded = null, scoreText = '';

function setupScore(j) {
  const box = el('scorebox'), wrap = el('scorewrap'), pre = el('scoretext');
  if (!j.has_score || !j.dir) {
    box.style.display = 'none';
    scoreDir = null; scoreLoaded = null; scoreText = '';
    return;
  }
  box.style.display = 'block';
  scoreDir = j.dir;
  scoreLoaded = null;
  scoreText = '';
  pre.textContent = '';
  wrap.style.display = 'none';
  el('scoretoggle').textContent = '▸ 看看模型给这首歌写的谱';
}

async function loadScore(dir) {
  if (scoreLoaded === dir) return;
  el('scoretext').textContent = '读取中…';
  try {
    const r = await fetch('/api/score?dir=' + encodeURIComponent(dir));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    scoreText = d.abc || '';
    scoreLoaded = dir;
    el('scoretext').textContent = scoreText || '（这首没有乐谱）';
  } catch (e) {
    el('scoretext').textContent = '读取乐谱失败：' + e.message;
  }
}

// 历史作品里的「乐谱」按钮走这里
function viewScore(dir) {
  scoreDir = dir;
  scoreLoaded = null;
  scoreText = '';
  el('scorebox').style.display = 'block';
  el('scorewrap').style.display = 'block';
  el('scoretoggle').textContent = '▾ 收起乐谱';
  loadScore(dir);
  el('scorebox').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function toggleScore() {
  const wrap = el('scorewrap'), btn = el('scoretoggle');
  const open = wrap.style.display === 'block';
  if (open) {
    wrap.style.display = 'none';
    btn.textContent = '▸ 看看模型给这首歌写的谱';
    return;
  }
  wrap.style.display = 'block';
  btn.textContent = '▾ 收起乐谱';
  if (scoreDir) loadScore(scoreDir);
}

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText)
    return navigator.clipboard.writeText(text);
  // 老浏览器兜底
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); } catch (e) { /* 忽略 */ }
  document.body.removeChild(ta);
  return Promise.resolve();
}

function setupScoreUI() {
  el('scoretoggle').onclick = toggleScore;

  el('scorecopy').onclick = async () => {
    if (!scoreText) { alert('乐谱还没读出来，先展开看一眼'); return; }
    const b = el('scorecopy'), old = b.textContent;
    await copyText(scoreText);
    b.textContent = '已复制';
    setTimeout(() => { b.textContent = old; }, 1400);
  };

  el('scoreuse').onclick = () => {
    if (!scoreText) { alert('乐谱还没读出来，先展开看一眼'); return; }
    el('a-abc').value = scoreText;
    // 有外部乐谱就必须写谱，把规划方式切到完整，免得用户以为是「直接生成」
    planMode = 'full';
    markPlan();
    showPlanNote();
    el('adv').style.display = 'block';
    el('advtoggle').textContent = '▾ 高级设置（想精调再展开）';
    el('a-abc').scrollIntoView({ behavior: 'smooth', block: 'center' });
    el('a-abc').focus();
  };
}

function fmtDur(sec) {
  if (!sec) return '未知长度';
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return m ? m + ' 分 ' + s + ' 秒' : s + ' 秒';
}

function fmtWhen(ts) {
  const d = new Date(ts * 1000), p = n => (n < 10 ? '0' : '') + n;
  return p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' '
       + p(d.getHours()) + ':' + p(d.getMinutes());
}

async function loadHistory() {
  let d;
  try { d = await (await fetch('/api/history')).json(); } catch (e) { return; }
  const card = el('historycard'), box = el('history');
  if (!d.items || !d.items.length) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  box.innerHTML = '';
  d.items.forEach(it => {
    const url = '/api/audio?dir=' + encodeURIComponent(it.dir);
    const row = document.createElement('div');
    row.className = 'hrow';

    const t = document.createElement('span');
    t.className = 'htime';
    t.textContent = fmtDur(it.seconds);

    const w = document.createElement('span');
    w.className = 'hwhen';
    w.textContent = fmtWhen(it.mtime);

    const play = document.createElement('button');
    play.className = 'hbtn';
    play.textContent = '播放';
    play.onclick = () => {
      const p = el('player');
      p.src = url + '&t=' + Date.now();
      el('dl').href = url + '&download=1';
      el('dl').download = 'yue2-' + it.dir + '.flac';
      el('result').style.display = 'block';
      el('meta').textContent =
        '正在播放历史作品 · ' + fmtDur(it.seconds) + ' · ' + fmtWhen(it.mtime);
      p.play();
      el('result').scrollIntoView({ behavior: 'smooth', block: 'center' });
    };

    const dl = document.createElement('a');
    dl.className = 'hbtn';
    dl.textContent = '下载';
    dl.href = url + '&download=1';
    dl.download = 'yue2-' + it.dir + '.flac';

    const btns = [t, w, play, dl];
    if (it.has_score) {
      const sc = document.createElement('button');
      sc.className = 'hbtn';
      sc.textContent = '乐谱';
      sc.onclick = () => viewScore(it.dir);
      btns.push(sc);
    }

    if (mp3ok) {
      const m3 = document.createElement('a');
      m3.className = 'hbtn';
      m3.textContent = 'MP3';
      m3.href = url + '&download=1&format=mp3';
      m3.download = 'yue2-' + it.dir + '.mp3';
      btns.push(m3);
    }
    row.append(...btns);
    box.appendChild(row);
  });
}

boot();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# HTTP 服务
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # 静音访问日志，控制台只留有用信息

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, build_html(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(200, json.dumps({
                "hardware": HW, "profile": PROFILE,
                "installed": models_ready(), "job": get_job(),
                "mp3": FFMPEG is not None,
            }, ensure_ascii=False))
        elif path == "/api/job":
            self._send(200, json.dumps(get_job(), ensure_ascii=False))
        elif path == "/api/history":
            self._send(200, json.dumps({"items": list_history()},
                                       ensure_ascii=False))
        elif path == "/api/audio":
            self._serve_audio()
        elif path == "/api/score":
            self._serve_score()
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _safe_dir_name(self):
        """从查询串里取输出目录名，挡掉路径穿越。"""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        name = (q.get("dir") or [""])[0]
        if not name or "/" in name or "\\" in name or ".." in name:
            return None
        return name

    def _serve_score(self):
        """把模型规划出来的 ABC 乐谱发回去。

        直接生成模式（cot=off）没有乐谱，这时候返回 404，
        前端会自己把乐谱那一块藏起来。
        """
        name = self._safe_dir_name()
        if name is None:
            self._send(400, json.dumps({"error": "bad dir"}))
            return
        target = OUTPUTS / name / "score.abc"
        if not target.exists():
            self._send(404, json.dumps({"error": "no score"}))
            return
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            self._send(500, json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
            return
        self._send(200, json.dumps({"abc": text}, ensure_ascii=False))

    def _serve_audio(self):
        name = self._safe_dir_name()
        if name is None:
            self._send(400, json.dumps({"error": "bad dir"}))
            return
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        if (q.get("format") or [""])[0].lower() == "mp3":
            self._serve_mp3(name)
            return
        target = OUTPUTS / name / "audio.flac"
        if not target.exists():
            self._send(404, json.dumps({"error": "audio not found"}))
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/flac")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_mp3(self, name):
        """按需把 flac 转成 MP3，结果缓存在同目录，同一首只转一次。"""
        src = OUTPUTS / name / "audio.flac"
        if not src.exists():
            self._send(404, json.dumps({"error": "audio not found"}))
            return
        if FFMPEG is None:
            self._send(503, json.dumps({"error": "ffmpeg not available"}))
            return
        dst = OUTPUTS / name / "audio.mp3"
        try:
            stale = not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime
        except OSError:
            stale = True
        if stale:
            try:
                subprocess.run(
                    [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                     "-codec:a", "libmp3lame", "-b:a", "192k", str(dst)],
                    check=True, timeout=300, capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception as exc:
                self._send(500, json.dumps(
                    {"error": f"MP3 转码失败：{type(exc).__name__}: {exc}"},
                    ensure_ascii=False))
                return
        if not dst.exists():
            self._send(500, json.dumps({"error": "MP3 转码没有产出文件"}))
            return
        data = dst.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/api/generate":
            self._send(404, json.dumps({"error": "not found"}))
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send(400, json.dumps({"error": "bad json"}))
            return

        if PROFILE.get("disabled") or not models_ready():
            self._send(400, json.dumps(
                {"error": "当前环境无法生成"}, ensure_ascii=False))
            return

        job = get_job()
        if job["status"] == "running":
            self._send(409, json.dumps(
                {"error": "已有任务在生成中"}, ensure_ascii=False))
            return

        style = str(payload.get("style", "")).strip()
        lyrics = str(payload.get("lyrics", "")).strip()
        try:
            seconds = float(payload.get("seconds") or 20)
        except Exception:
            seconds = 20.0

        if not style or not lyrics:
            self._send(400, json.dumps({"error": "风格和歌词不能为空"}))
            return

        cap = PROFILE.get("max_seconds", 0)
        if cap and seconds > cap:
            seconds = float(cap)

        opts = sanitize_options(payload.get("options"))

        # 用户指定了种子就用它（可复现），否则每首换一个，
        # 保证「再生成一首」不会得到完全一样的结果
        try:
            seed = int(payload.get("seed"))
            if not 0 <= seed < 2 ** 63:
                raise ValueError
        except (TypeError, ValueError):
            seed = random.randint(1, 2 ** 31 - 1)

        # 一次出几首。采样随机性大，出 4 首挑 1 首比反复重抽划算得多
        try:
            count = int(payload.get("count") or 1)
        except (TypeError, ValueError):
            count = 1
        count = max(1, min(MAX_BATCH, count))

        if count > 1:
            threading.Thread(
                target=run_batch,
                args=(style, lyrics, seconds, seed, opts, count),
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=run_generation, args=(style, lyrics, seconds, seed, opts),
                daemon=True,
            ).start()
        self._send(200, json.dumps({"ok": True, "seed": seed, "count": count}))


def main():
    force_utf8()
    global HW, PROFILE

    print()
    print("=" * 62)
    print("  YuE2 音乐生成器")
    print("=" * 62)

    if not models_ready():
        print()
        print("  模型文件不完整，请先双击运行 install.bat 完成安装。")
        print(f"  模型目录：{MODELS}")
        print()
        try:
            input("  按回车键退出…")
        except EOFError:
            pass
        return 1

    OUTPUTS.mkdir(exist_ok=True)

    print()
    print("  正在检测显卡…")
    HW = detect_hardware()
    PROFILE = pick_profile(HW)

    if HW.get("cuda"):
        print(f"  显卡：{HW.get('name')}")
        print(f"  显存：{HW.get('vram_gib', 0):.1f} GB")
    else:
        print("  未检测到可用的 NVIDIA 显卡。")
    print(f"  运行模式：{PROFILE['label']}")
    print(f"  说明：{PROFILE['note']}")

    if PROFILE.get("disabled"):
        print()
        print("  当前硬件无法运行 YuE2。程序仍会启动，但无法生成音乐。")

    url = f"http://{'127.0.0.1' if BIND == '0.0.0.0' else BIND}:{PORT}/"
    print()
    print(f"  监听地址：{BIND}:{PORT}")
    print(f"  本机访问：{url}")
    if BIND == "0.0.0.0":
        print("  外部访问：用云主机的公网 IP 或平台提供的端口转发地址")
    print("  浏览器没自动打开的话，手动复制上面这个地址访问。")
    print()
    print("  关闭这个窗口即可退出程序。")
    print("=" * 62)
    print()

    def open_browser():
        # 云主机上没有图形界面，webbrowser 会报错或卡住，静默跳过即可
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if os.environ.get("YUE2_NO_BROWSER") != "1":
        threading.Timer(1.2, open_browser).start()

    server = ThreadingHTTPServer((BIND, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  已退出。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
