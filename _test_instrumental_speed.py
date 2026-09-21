"""复测：纯器乐 vs 带人声，同样 60 秒，看耗时到底差多少。

上次纯器乐 60 秒跑了 13.2 分钟（1:13.2），但当时并发跑了前端验证脚本
（Chrome 抢内存），结论不可靠。这次串行跑两个样本，中途不做任何别的事，
把「纯器乐是不是真的慢 5 倍」这件事钉死。

样本 A = 带人声中文歌，样本 B = 纯器乐。其余参数完全一致（cot=melody / ode 32 / 60 秒）。
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7861"
GUI = Path(__file__).parent.resolve()
OUTPUTS = GUI / "outputs"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# VAE 25 帧/秒，1 语义令牌 = 1 帧
FPS = 25.0

SAMPLES = [
    {
        "name": "A 带人声",
        "style": ("Mandarin, pop ballad, warm expressive female vocal, grand piano, "
                  "strings, 72 BPM"),
        "lyrics": ("[Verse]\n夜色落在窗台上\n一盏灯陪我发呆\n\n"
                   "[Chorus]\n就这样慢慢地走"),
    },
    {
        "name": "B 纯器乐",
        "style": ("Instrumental, Chinese traditional, guqin and dizi ensemble, "
                  "pentatonic scale, meditative and serene, silk and bamboo, "
                  "no vocals, purely instrumental, 72 BPM"),
        "lyrics": "[Instrumental]",
    },
]


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def get(path):
    with OPENER.open(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_done(timeout=2400):
    t0 = time.perf_counter()
    last_pct = -1
    while time.perf_counter() - t0 < timeout:
        job = get("/api/job")
        pct = job.get("progress") or 0
        if pct != last_pct and pct % 10 == 0:
            print(f"      [{time.perf_counter() - t0:7.1f}s] {pct:>3}% "
                  f"{job.get('stage')} {(job.get('message') or '')[:48]}", flush=True)
            last_pct = pct
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(4)
    return {"status": "timeout"}


def run_one(sample):
    before = {p.name for p in OUTPUTS.iterdir() if p.is_dir()} if OUTPUTS.exists() else set()
    payload = {
        "style": sample["style"],
        "lyrics": sample["lyrics"],
        "seconds": 60,
        "seed": 831001,
        "options": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 100,
            "penalty_window": 50, "repetition_penalty": 1.2,
            "ode_steps": 32, "cfg_scale": None, "abc": None,
            "cot": "melody",
        },
    }
    print(f"\n>>> {sample['name']}  提交中…", flush=True)
    t0 = time.perf_counter()
    code, body = post("/api/generate", payload)
    print(f"    HTTP {code} {body}", flush=True)
    job = wait_done()
    elapsed = time.perf_counter() - t0
    if job.get("status") != "done":
        print(f"    失败：{job.get('status')} {job.get('message')} {job.get('error')}")
        return None
    secs = job.get("seconds") or 0
    tokens = secs * FPS
    return {
        "name": sample["name"],
        "seconds": secs,
        "elapsed": elapsed,
        "tokens": tokens,
        "sec_per_token": elapsed / tokens if tokens else None,
        "ratio": elapsed / secs if secs else None,
        "peak_vram": job.get("peak_vram_gb"),
        "min_avail": job.get("min_avail_gb"),
        "truncated": job.get("truncated"),
    }


def main():
    st = get("/api/status")
    if not st.get("installed"):
        print("模型未就绪，退出")
        return 1

    rows = []
    for s in SAMPLES:
        r = run_one(s)
        if r:
            rows.append(r)
            print(f"    完成：{r['seconds']:.1f}s 音频 / 耗时 {r['elapsed']:.1f}s "
                  f"/ {r['sec_per_token']:.3f} 秒每令牌", flush=True)

    if not rows:
        return 2

    print("\n================ 复测对照 ================")
    print(f"{'样本':<10}{'音频':>9}{'总耗时':>11}{'秒/令牌':>10}{'音频:耗时':>12}")
    for r in rows:
        print(f"{r['name']:<10}{r['seconds']:>8.1f}s{r['elapsed']:>10.1f}s"
              f"{r['sec_per_token']:>10.3f}{'1:' + format(r['ratio'], '.2f'):>12}")
    if len(rows) == 2:
        a, b = rows
        if a["sec_per_token"] and b["sec_per_token"]:
            print(f"\n纯器乐 / 带人声 耗时比 = "
                  f"{b['sec_per_token'] / a['sec_per_token']:.2f} 倍")
    return 0


if __name__ == "__main__":
    sys.exit(main())
