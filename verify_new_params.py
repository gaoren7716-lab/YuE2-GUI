"""聚焦验证：新增的 top_k / penalty_window 是否真的传到推理层并落盘。

做法：POST /api/generate 带上显式 top_k=40 / penalty_window=20 / ode_steps=16，
轮询到结束，读 outputs/<dir>/config.json 的 semantic 采样核对落盘值。
"""
import json
import sys
import time
import urllib.error
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


def wait_done(timeout=1200):
    t0 = time.perf_counter()
    last = ""
    while time.perf_counter() - t0 < timeout:
        job = get("/api/job")
        line = f"{job.get('status')}|{job.get('progress')}|{job.get('stage')}"
        if line != last:
            print(f"    [{time.perf_counter() - t0:6.1f}s] {job.get('progress'):>3}% "
                  f"{job.get('stage')} {(job.get('message') or '')[:52]}")
            last = line
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(3)
    return {"status": "timeout"}


def main():
    st = get("/api/status")
    if not st.get("installed"):
        print("模型未就绪，退出")
        return 1

    before = {p.name for p in OUTPUTS.iterdir() if p.is_dir()} if OUTPUTS.exists() else set()
    payload = {
        "style": "Mandarin, Lo-fi, dreamy and atmospheric, warm expressive female vocal, "
                 "piano, drum machine, 78 BPM",
        "lyrics": "[Verse]\n夜色落在窗台上\n一盏灯陪我发呆\n\n[Chorus]\n就这样慢慢地走",
        "seconds": 10,
        "seed": 424242,
        "options": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 40,
            "penalty_window": 20, "repetition_penalty": 1.2,
            "ode_steps": 16, "cfg_scale": None, "abc": None,
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
    print(f"完成：{job.get('seconds'):.1f}s 音频 / 总耗时 {elapsed:.1f}s")

    after = {p.name for p in OUTPUTS.iterdir() if p.is_dir()}
    new = sorted(after - before)
    d = OUTPUTS / new[-1]
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    gen = cfg.get("generation", {})
    sem = gen.get("semantic", {})

    checks = [
        ("ode_steps", gen.get("ode_steps"), 16),
        ("semantic.top_k", sem.get("top_k"), 40),
        ("semantic.penalty_window", sem.get("penalty_window"), 20),
        ("semantic.temperature", sem.get("temperature"), 1.0),
        ("semantic.repetition_penalty", sem.get("repetition_penalty"), 1.2),
    ]
    ok = True
    for label, got, want in checks:
        good = got == want
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {label}: {got!r}"
              + ("" if good else f"  (期望 {want!r})"))
    print("\n快照 semantic =", json.dumps(sem, ensure_ascii=False))
    print("结果：", "通过" if ok else "未通过")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
