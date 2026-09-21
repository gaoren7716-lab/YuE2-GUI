#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YuE2 环境自检：确认这台机器能不能生成音乐。

不走界面，直接跑一条最短的完整链路（加载 → 生成 → 保存），
顺便把硬件、峰值显存、内存余量、各阶段耗时打出来。
出问题时先跑这个，比在界面里试快得多。

用法：
    .venv\\Scripts\\python.exe selftest.py          # 默认 20 秒片段
    .venv\\Scripts\\python.exe selftest.py 60       # 指定秒数
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
OUT = ROOT / "outputs" / "selftest"
GIB = 1024 ** 3
FPS = 25.0          # 实测：VAE 25 帧/秒，1 个语义令牌 = 1 帧 = 0.04 秒音频

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0


def line(label, value):
    print(f"  {label:<12}{value}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print()
    print("=" * 60)
    print("  YuE2 环境自检")
    print("=" * 60)

    # ---- 1. 硬件 ----
    print("\n[1/4] 硬件")
    try:
        import torch
    except Exception as exc:
        line("失败", f"读不到 torch：{exc}")
        line("", "请先运行 install.bat / install.sh 完成安装。")
        return 1

    line("torch", torch.__version__)
    if not torch.cuda.is_available():
        line("CUDA", "不可用")
        line("", "YuE2 需要 NVIDIA 显卡，这台机器跑不了推理。")
        return 1

    props = torch.cuda.get_device_properties(0)
    line("显卡", props.name)
    line("显存", f"{props.total_memory / GIB:.2f} GiB")
    line("算力", f"CC {props.major}.{props.minor}")
    line("BF16", "支持" if torch.cuda.is_bf16_supported() else "不支持")

    import ram_mode
    mem = ram_mode.system_memory()
    if mem:
        line("内存", f"{mem[0]:.1f} GB（可用 {mem[1]:.1f} GB，页面文件可用 {mem[2]:.1f} GB）")
        advice = ram_mode.memory_advice()
        if not advice["ok"]:
            print()
            line("提醒", advice["message"])

    # ---- 2. 模型文件 ----
    print("\n[2/4] 模型文件")
    needed = [MODELS / "YuE2-3B" / "model.safetensors",
              MODELS / "YuE2-3B" / "config.json",
              MODELS / "YuE2-3B" / "qwen.tiktoken",
              MODELS / "YuE2-Vae" / "model.safetensors"]
    missing = [p for p in needed if not p.exists()]
    if missing:
        for p in missing:
            line("缺失", str(p))
        line("", "请先运行 install.bat / install.sh 下载模型。")
        return 1
    total = sum(p.stat().st_size for p in needed) / GIB
    line("状态", f"齐全（{total:.2f} GiB）")

    # ---- 3. 加载 ----
    print("\n[3/4] 加载模型（启用内存扩展模式）")
    fraction = ram_mode.install(device="cuda")
    line("显存上限", f"{fraction:.0%}（官方默认会预留 2 GiB）")

    from yue2 import YuE2Pipeline

    # 内存采样：峰值显存好看，但内存最低可用才是崩溃的真正前兆
    samples = {"min": 1e9, "stop": False}

    def sampler():
        while not samples["stop"]:
            info = ram_mode.system_memory()
            if info:
                samples["min"] = min(samples["min"], info[1])
            time.sleep(0.5)

    threading.Thread(target=sampler, daemon=True).start()

    t0 = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained(
        str(MODELS / "YuE2-3B"), vae=str(MODELS / "YuE2-Vae"),
        local_files_only=True, progress=False, device="cuda",
        memory_budget_gib=max(6, int(round(props.total_memory / GIB))),
        backend="torch-eager", quantization="none", offload_ar=True,
        verify_hashes=False,
    )
    line("加载耗时", f"{time.perf_counter() - t0:.1f} 秒")

    # ---- 4. 生成 ----
    tokens = max(200, int(round(SECONDS * FPS)))
    print(f"\n[4/4] 生成 {SECONDS:.0f} 秒片段（{tokens} 个语义令牌）")

    t0 = time.perf_counter()
    song = pipe(
        style=("Mandarin, simple acoustic pop, warm female lead vocal, "
               "clean piano, soft bass and light drums, 96 BPM"),
        lyrics=("[Verse]\n路灯亮起的时候\n我们还在路上\n\n"
                "[Chorus]\n再走一会儿\n天就亮了"),
        cot="full",
        seed=831001,
        semantic_sampling={"max_tokens": tokens, "min_tokens": 200},
    )
    elapsed = time.perf_counter() - t0
    samples["stop"] = True

    OUT.mkdir(parents=True, exist_ok=True)
    path = song.save(OUT / "audio.flac")
    secs = len(song.audio) / song.sample_rate
    peak = torch.cuda.max_memory_allocated() / GIB
    timing = song.timing

    print()
    print("=" * 60)
    line("音频长度", f"{secs:.1f} 秒")
    line("总耗时", f"{elapsed:.0f} 秒")
    line("  ABC 规划", f"{timing['abc']['seconds']:.0f} 秒 / {timing['abc']['output_tokens']} 令牌")
    line("  语义生成", f"{timing['semantic']['seconds']:.0f} 秒 / "
                      f"{timing['semantic']['output_tokens']} 令牌 "
                      f"({timing['semantic']['output_tps']:.1f} tok/s)")
    line("  NAR 合成", f"{timing['nar_seconds']:.0f} 秒")
    line("  VAE 解码", f"{timing['vae_seconds']:.0f} 秒")
    line("峰值显存", f"{peak:.2f} GiB / {props.total_memory / GIB:.2f} GiB")
    line("内存最低可用", f"{samples['min']:.2f} GB")
    line("是否截断", str(song.truncated))
    line("输出文件", str(path))
    print("=" * 60)

    if song.truncated["semantic"]:
        print()
        print("  提示：语义生成撞到了 max_tokens 上限，歌曲被截断。")
        print("  歌曲长度主要由歌词长度决定（一段约 15 秒），")
        print("  想要更长的歌请多写几段歌词。")
    print()
    print("  自检通过。双击 start.bat 打开图形界面。")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
