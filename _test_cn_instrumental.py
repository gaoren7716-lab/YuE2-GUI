"""实测：中国风纯器乐到底能不能出、出成什么样。

用途不是回归测试，是给老板一份真实样本 + 客观数据。
跑完把输出目录路径打出来，老板自己点播放听。
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


def wait_done(timeout=1500):
    t0 = time.perf_counter()
    last = ""
    while time.perf_counter() - t0 < timeout:
        job = get("/api/job")
        line = f"{job.get('status')}|{job.get('progress')}"
        if line != last:
            print(f"    [{time.perf_counter() - t0:6.1f}s] {job.get('progress'):>3}% "
                  f"{job.get('stage')} {(job.get('message') or '')[:56]}", flush=True)
            last = line
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(3)
    return {"status": "timeout"}


def main():
    st = get("/api/status")
    print("服务状态：", json.dumps(st, ensure_ascii=False)[:200])
    if not st.get("installed"):
        print("模型未就绪，退出")
        return 1

    before = {p.name for p in OUTPUTS.iterdir() if p.is_dir()} if OUTPUTS.exists() else set()

    # 老板的真实目标：中国风 + 纯器乐。用组合器能拼出来的写法。
    payload = {
        "style": ("Instrumental, Chinese traditional, guqin and dizi ensemble, "
                  "pentatonic scale, meditative and serene, silk and bamboo, "
                  "no vocals, purely instrumental, 72 BPM"),
        "lyrics": "[Instrumental]",
        "seconds": 60,
        "seed": 831001,
        "options": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 100,
            "penalty_window": 50, "repetition_penalty": 1.2,
            "ode_steps": 32, "cfg_scale": None, "abc": None,
            "cot": "melody",
        },
    }
    t0 = time.perf_counter()
    code, body = post("/api/generate", payload)
    print(f"提交 -> HTTP {code} {body}")
    job = wait_done()
    if job.get("status") != "done":
        print("失败：", job.get("status"), job.get("message"), job.get("error"))
        return 2

    elapsed = time.perf_counter() - t0
    after = {p.name for p in OUTPUTS.iterdir() if p.is_dir()}
    new = sorted(after - before)
    if not new:
        print("没找到新输出目录")
        return 3
    d = OUTPUTS / new[-1]

    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    gen = cfg.get("generation", {})
    print("\n================ 实测结果 ================")
    print(f"输出目录     : {d}")
    print(f"目标时长     : 60s")
    print(f"实际产出     : {job.get('seconds')}s")
    print(f"总耗时       : {elapsed:.1f}s  (音频:耗时 = 1:{elapsed / max(job.get('seconds') or 1, 1):.2f})")
    print(f"峰值显存     : {job.get('peak_vram_gb')} GiB")
    print(f"最低可用内存 : {job.get('min_avail_gb')} GB")
    print(f"cot 档位     : {gen.get('cot')}")
    print(f"ode_steps    : {gen.get('ode_steps')}")
    print(f"是否截断     : {gen.get('truncated')}")
    print("落盘目录内容 :")
    for f in sorted(d.iterdir()):
        print(f"    {f.name:<18} {f.stat().st_size / 1024:>9.1f} KB")
    abc = d / "score.abc"
    if abc.exists():
        txt = abc.read_text(encoding="utf-8", errors="replace")
        print(f"\n乐谱前 12 行（共 {len(txt.splitlines())} 行）：")
        for ln in txt.splitlines()[:12]:
            print("    " + ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
