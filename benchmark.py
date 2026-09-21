# -*- coding: utf-8 -*-
"""完整歌曲压测：跑一首歌并记录内存/显存时间线。

不走界面，直接调用推理库，用来判断「这台机器到底能不能跑完整歌曲」，
以及找出耗时结构和内存瓶颈在哪。

用法：
    .venv\\Scripts\\python.exe benchmark.py          # 默认 180 秒
    .venv\\Scripts\\python.exe benchmark.py 60       # 跑 60 秒

输出会存到 outputs/ramtest-<秒数>/，可以直接听。
"""
import sys, time, threading, torch
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
import ram_mode

GIB = 1024 ** 3
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
FPS = 25.0

# 歌曲长度由歌词长度决定（一段约 15 秒），max_tokens 只是上限。
# 所以压测长歌必须把歌词也铺够，否则会提前收尾，量到的不是目标时长。
VERSE_POOL = [
    "[Verse]\n路灯把影子拉得很长\n我数着回家的方向\n口袋里那张旧车票\n还留着你的名字啊",
    "[Verse]\n站台的风吹过旧时光\n广播里念着远方\n我把围巾裹得更紧\n好像还能闻到你的香",
    "[Verse]\n今天的雨下得很轻\n像你从前说话的声音\n我终于学会了不回头\n可还是会在夜里想起",
    "[Verse]\n窗外的树又绿了一遍\n时间走得比想象中快\n我学会了笑着说起你\n只是不再说后来",
]
CHORUS = ("[Chorus]\n如果风会说话\n它会替我说想你啊\n"
          "如果云也会停下\n它会替我看你一眼")
BRIDGE = "[Bridge]\n一年又一年\n我把心事折成纸船\n放进这条没有尽头的河"
OUTRO = "[Outro]\n风还在吹\n我还在这里"


def build_lyrics(seconds):
    """按目标时长拼出足够长的歌词。实测一段约唱 15 秒。"""
    need = max(4, int(seconds / 15) + 2)
    parts = []
    while len(parts) < need - 1:
        parts.append(VERSE_POOL[len(parts) % len(VERSE_POOL)])
        if len(parts) < need - 1:
            parts.append(CHORUS)
        if len(parts) >= need - 3 and BRIDGE not in parts:
            parts.append(BRIDGE)
    parts.append(OUTRO)
    return "\n\n".join(parts)


LYRICS = build_lyrics(SECONDS)

STATE = {"phase": "init", "samples": [], "stop": False}


def sampler():
    while not STATE["stop"]:
        mem = ram_mode.system_memory()
        free, _ = torch.cuda.mem_get_info()
        STATE["samples"].append((
            time.perf_counter() - T0,
            STATE["phase"],
            mem[1] if mem else -1,
            mem[2] if mem else -1,
            free / GIB,
            torch.cuda.memory_allocated() / GIB,
        ))
        time.sleep(1.0)


T0 = time.perf_counter()
threading.Thread(target=sampler, daemon=True).start()

tokens = int(SECONDS * FPS)
print(f"[目标] {SECONDS:.0f} 秒完整歌曲 = {tokens} 个语义令牌")
ram_mode.install()
from yue2 import YuE2Pipeline

STATE["phase"] = "加载"
t0 = time.perf_counter()
pipe = YuE2Pipeline.from_pretrained(
    str(ROOT / "models" / "YuE2-3B"), vae=str(ROOT / "models" / "YuE2-Vae"),
    local_files_only=True, progress=False, device="cuda",
    memory_budget_gib=8, backend="torch-eager",
    quantization="none", offload_ar=True, verify_hashes=False,
)
print(f"[加载] {time.perf_counter() - t0:.1f} 秒", flush=True)

counter = {"n": 0}
last = {"t": time.perf_counter()}


def on_token(phase, token):
    counter["n"] += 1
    STATE["phase"] = f"{phase}生成"
    now = time.perf_counter()
    if now - last["t"] >= 60:
        last["t"] = now
        mem = ram_mode.system_memory()
        free, _ = torch.cuda.mem_get_info()
        print(f"      累计 {counter['n']} 令牌 | 用时 {now - T0:.0f}s | "
              f"内存可用 {mem[1]:.2f} GB | 显存分配 {torch.cuda.memory_allocated() / GIB:.2f} GiB",
              flush=True)


STATE["phase"] = "生成"
t1 = time.perf_counter()
song = pipe(
    style="Mandarin, contemporary pop ballad, warm expressive female vocal, piano and strings, soft drums, emotional singable chorus, 82 BPM",
    lyrics=LYRICS,
    cot="full",
    seed=20260920,
    semantic_sampling={"max_tokens": tokens, "min_tokens": 200},
    on_token=on_token,
)
total = time.perf_counter() - t1
STATE["stop"] = True
time.sleep(0.2)

audio = song.audio
peak_vram = torch.cuda.max_memory_allocated() / GIB
report = pipe._model._yue2_ram.report()

print()
print("=" * 62)
print(f"音频长度      {len(audio) / 48000:.1f} 秒  ({len(audio) / 48000 / 60:.2f} 分钟)")
print(f"总耗时        {total:.0f} 秒  ({total / 60:.1f} 分钟)")
print(f"  ABC 规划    {song.timing['abc']['seconds']:.0f} 秒 / {song.timing['abc']['output_tokens']} 令牌")
print(f"  语义生成    {song.timing['semantic']['seconds']:.0f} 秒 / "
      f"{song.timing['semantic']['output_tokens']} 令牌 ({song.timing['semantic']['output_tps']:.2f} tok/s)")
print(f"  NAR 合成    {song.timing['nar_seconds']:.0f} 秒")
print(f"  VAE 解码    {song.timing['vae_seconds']:.0f} 秒")
print(f"峰值显存      {peak_vram:.2f} GiB / 8.00 GiB")
print(f"阶段调度      {report}")
print(f"是否截断      {song.truncated}")

print()
print("内存时间线（每 10 秒抽 1 点，只显示低于 3 GB 的告急时刻）")
print(f"{'时刻':>8} {'阶段':<12} {'物理可用':>10} {'提交可用':>10} {'显存空闲':>10} {'显存分配':>10}")
low = [s for s in STATE["samples"] if 0 <= s[2] < 3.0]
if not low:
    print("      没有低于 3 GB 的时刻")
else:
    step = max(1, len(low) // 25)
    for s in low[::step]:
        print(f"{s[0]:8.0f} {s[1]:<12} {s[2]:9.2f}G {s[3]:9.1f}G {s[4]:9.2f}G {s[5]:9.2f}G")

valid = [s for s in STATE["samples"] if s[2] >= 0]
print()
print(f"物理内存最低可用   {min(s[2] for s in valid):.2f} GB  "
      f"（出现在 {min(valid, key=lambda s: s[2])[0]:.0f} 秒，阶段 {min(valid, key=lambda s: s[2])[1]}）")
print(f"显存最低空闲       {min(s[4] for s in valid):.2f} GiB")

out = ROOT / "outputs" / f"ramtest-{int(SECONDS)}s"
song.save_artifacts(out)
print(f"已保存        {out}")
print("=" * 62)
