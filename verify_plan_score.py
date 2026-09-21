"""第二轮端到端：验证新增的「规划方式」选择器与 /api/score 乐谱接口。

要验证的：
  1. options.cot="off"   -> config.cot 落盘为 off，且没有 score.abc
  2. options.cot="melody"-> config.cot 落盘为 melody，score.abc 存在
  3. /api/score 能把乐谱读出来，且挡得住路径穿越
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

OK = 0
BAD = 0


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with OPENER.open(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def get(path):
    with OPENER.open(BASE + path, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def wait_done(timeout=1800):
    t0 = time.perf_counter()
    last = ""
    while time.perf_counter() - t0 < timeout:
        _, job = get("/api/job")
        line = f"{job.get('status')}|{job.get('progress')}|{job.get('stage')}"
        if line != last:
            print(f"    [{time.perf_counter() - t0:6.1f}s] {job.get('progress'):>3}% "
                  f"{job.get('stage')} {(job.get('message') or '')[:48]}")
            last = line
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(3)
    return {"status": "timeout"}


def check(label, got, want):
    global OK, BAD
    ok = got == want
    if ok:
        OK += 1
    else:
        BAD += 1
    print(f"    {'OK ' if ok else 'BAD'} {label}: {got!r}"
          + ("" if ok else f"  (期望 {want!r})"))


def run_cot_case(title, cot, want_score, seed):
    print(f"\n{'-' * 66}\n{title}\n{'-' * 66}")
    before = {p.name for p in OUTPUTS.iterdir() if p.is_dir()}
    code, body = post("/api/generate", {
        "style": "Mandarin, Ambient, dreamy and atmospheric, "
                 "no vocals, purely instrumental, synthesizer, 70 BPM",
        "lyrics": "[Verse]\n(器乐)\n\n[Chorus]\n(器乐)",
        "seconds": 10,
        "seed": seed,
        "options": {"temperature": 1.0, "top_p": 0.95, "top_k": 100,
                    "repetition_penalty": 1.2, "ode_steps": 16,
                    "cfg_scale": None, "cot": cot, "abc": None},
    })
    print(f"  提交 -> HTTP {code} {body}")
    job = wait_done()
    if job.get("status") != "done":
        print(f"  失败 status={job.get('status')} msg={job.get('message')}")
        print(f"  err={job.get('error')}")
        return None

    print(f"  完成：{job.get('seconds'):.1f}s 音频 / 峰值显存 "
          f"{job.get('peak_vram'):.2f} GiB")
    new = sorted({p.name for p in OUTPUTS.iterdir() if p.is_dir()} - before)
    d = new[-1] if new else None
    if not d:
        print("  没找到输出目录")
        return None

    cfg = json.loads((OUTPUTS / d / "config.json").read_text(encoding="utf-8"))
    print(f"  输出目录：{d}")
    check("config.cot", cfg.get("cot"), cot)
    check("job.has_score", job.get("has_score"), want_score)
    check("磁盘上 score.abc 存在", (OUTPUTS / d / "score.abc").exists(), want_score)
    return d


def main():
    _, st = get("/api/status")
    print(f"环境：{st['hardware']['name']} / 可用内存 "
          f"{st['hardware']['ram_free_gb']:.2f} GB")
    print(f"模式：{st['profile'].get('label')} / 默认 cot={st['profile'].get('cot')} / "
          f"上限 {st['profile'].get('max_seconds')}s")
    if not st.get("installed"):
        print("模型未就绪")
        return 1

    # 直接生成：不该有乐谱
    d_off = run_cot_case("用例 A：规划方式 = 直接生成（cot=off）", "off", False, 444444)
    if d_off:
        try:
            code, _ = get("/api/score?dir=" + d_off)
            check("/api/score 对无谱曲目返回 404", code, 404)
        except urllib.error.HTTPError as e:
            check("/api/score 对无谱曲目返回 404", e.code, 404)

    # 旋律规划：应该有乐谱
    d_mel = run_cot_case("用例 B：规划方式 = 旋律规划（cot=melody）", "melody", True, 555555)
    if d_mel:
        code, body = get("/api/score?dir=" + d_mel)
        check("/api/score 返回 200", code, 200)
        abc = (body or {}).get("abc") or ""
        print(f"    乐谱长度 {len(abc)} 字符，开头：{abc[:70]!r}")
        check("乐谱非空", len(abc) > 0, True)
        check("乐谱像 ABC（含 X: 头）", abc.lstrip().startswith("X:"), True)

    # 路径穿越防护
    print(f"\n{'-' * 66}\n安全：路径穿越\n{'-' * 66}")
    for bad in ["../models", "..%2f..", "a/b", "..\\x"]:
        try:
            code, _ = get("/api/score?dir=" + bad)
        except urllib.error.HTTPError as e:
            code = e.code
        check(f"挡掉 dir={bad!r}", code in (400, 404), True)

    print(f"\n{'=' * 66}\n通过 {OK} 项，失败 {BAD} 项\n{'=' * 66}")
    return 0 if BAD == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
