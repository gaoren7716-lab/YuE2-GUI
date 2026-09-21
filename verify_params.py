"""端到端验证：新创作参数是否真的传到推理层。

做法：POST /api/generate 带上 options，轮询 /api/job 直到结束，
然后读输出目录里的 config.json / request.json 核对参数落盘值。
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

# 绕过系统代理，本地回环直连
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


def wait_done(timeout=1800):
    """轮询直到任务结束，返回 job dict。"""
    t0 = time.perf_counter()
    last = ""
    while time.perf_counter() - t0 < timeout:
        job = get("/api/job")
        line = f"{job.get('status')}|{job.get('progress')}|{job.get('stage')}"
        if line != last:
            print(f"    [{time.perf_counter() - t0:6.1f}s] "
                  f"{job.get('progress'):>3}% {job.get('stage')} "
                  f"{(job.get('message') or '')[:52]}")
            last = line
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(3)
    return {"status": "timeout"}


def find_new_dir(before):
    after = {p.name for p in OUTPUTS.iterdir() if p.is_dir()}
    new = sorted(after - before)
    return OUTPUTS / new[-1] if new else None


def load_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def check(label, got, want):
    ok = got == want
    print(f"    {'OK ' if ok else 'BAD'} {label}: {got!r}"
          + ("" if ok else f"  (期望 {want!r})"))
    return ok


def run_case(idx, title, payload, expects, timeout=1800):
    print(f"\n{'=' * 66}\n用例 {idx}：{title}\n{'=' * 66}")
    before = {p.name for p in OUTPUTS.iterdir() if p.is_dir()} if OUTPUTS.exists() else set()

    t0 = time.perf_counter()
    try:
        code, body = post("/api/generate", payload)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
        return False, {}
    print(f"  提交 -> HTTP {code} {body}")

    job = wait_done(timeout)
    elapsed = time.perf_counter() - t0

    if job.get("status") != "done":
        print(f"  失败：status={job.get('status')}")
        print(f"  message={job.get('message')}")
        print(f"  error={job.get('error')}")
        return False, {}

    print(f"  完成：{job.get('seconds'):.1f}s 音频 / "
          f"总耗时 {elapsed:.1f}s / 峰值显存 {job.get('peak_vram')} GiB / "
          f"truncated={job.get('truncated')}")

    d = find_new_dir(before)
    if not d:
        print("  警告：没找到新输出目录，无法核对参数")
        return True, {}

    cfg = load_json(d / "config.json") or {}
    req = load_json(d / "request.json") or {}
    gen = cfg.get("generation", {})
    sem = gen.get("semantic", {})

    print(f"  输出目录：{d.name}")
    print("  --- 落盘参数核对 ---")
    allok = True
    for label, getter, want in expects:
        allok &= check(label, getter(cfg, req, gen, sem), want)

    print("  --- 全量快照 ---")
    print("    generation.ode_steps      =", gen.get("ode_steps"))
    print("    generation.abc            =", gen.get("abc"))
    print("    generation.semantic       =", sem)
    print("    cfg_scale / overrides     =", cfg.get("cfg_scale"),
          "/", cfg.get("overrides"))
    print("    cot                       =", cfg.get("cot"))
    print("    request.seed              =", req.get("seed"))
    print("    request.cfg_scale         =", req.get("cfg_scale"))
    print("    request.cot               =", req.get("cot"))
    print("    request.abc 长度          =",
          len(req.get("abc") or "") if req.get("abc") else None)
    return allok, {"elapsed": elapsed, "seconds": job.get("seconds")}


def main():
    st = get("/api/status")
    print(f"环境：{st['hardware']['name']} / "
          f"{st['hardware']['vram_gib']:.1f} GiB / "
          f"可用内存 {st['hardware']['ram_free_gb']:.2f} GB")
    print(f"模式：{'内存扩展' if st['profile'].get('ram_offload') else '标准'} / "
          f"上限 {st['profile'].get('max_seconds')}s")
    if not st.get("installed"):
        print("模型未就绪，退出")
        return 1

    results = {}

    # 用例 1：低开销跑通新代码路径，验证 ode_steps 与基础采样参数
    results["1"] = run_case(
        1, "短曲 + ode_steps=16 + 自定义采样（验证重建 GenerationConfig）",
        {
            "style": "Mandarin, Lo-fi, dreamy and atmospheric, warm expressive female "
                     "vocal, piano, drum machine, 78 BPM",
            "lyrics": "[Verse]\n夜色落在窗台上\n一盏灯陪我发呆\n\n[Chorus]\n就这样慢慢地走",
            "seconds": 10,
            "seed": 111111,
            "options": {
                "temperature": 1.0, "top_p": 0.95, "top_k": 100,
                "repetition_penalty": 1.2, "ode_steps": 16,
                "cfg_scale": None, "abc": None,
            },
        },
        [
            ("ode_steps", lambda c, r, g, s: g.get("ode_steps"), 16),
            ("semantic.temperature", lambda c, r, g, s: s.get("temperature"), 1.0),
            ("semantic.top_p", lambda c, r, g, s: s.get("top_p"), 0.95),
            ("semantic.repetition_penalty",
             lambda c, r, g, s: s.get("repetition_penalty"), 1.2),
            ("request.seed", lambda c, r, g, s: r.get("seed"), 111111),
            ("cot（本机默认档）", lambda c, r, g, s: c.get("cot"), "full"),
        ],
    )

    # 用例 2：高风险路径 —— 显式 cfg_scale 会走 CFG 双分支，显存翻倍
    results["2"] = run_case(
        2, "「大胆」+ 显式 cfg_scale=1.5 + ode_steps=32（验证 CFG 双分支不 OOM）",
        {
            "style": "Mandarin, City Pop, upbeat and joyful, warm expressive female "
                     "vocal, synthesizer, electric guitar, bass, 118 BPM",
            "lyrics": "[Verse]\n海风把霓虹吹散\n我们骑车穿过夏天\n\n"
                      "[Chorus]\n别问终点在哪边\n此刻就是永远",
            "seconds": 30,
            "seed": 222222,
            "options": {
                "temperature": 1.25, "top_p": 0.97, "top_k": 120,
                "repetition_penalty": 1.05, "ode_steps": 32,
                "cfg_scale": 1.5, "abc": None,
            },
        },
        [
            ("ode_steps", lambda c, r, g, s: g.get("ode_steps"), 32),
            ("semantic.temperature", lambda c, r, g, s: s.get("temperature"), 1.25),
            ("semantic.top_p", lambda c, r, g, s: s.get("top_p"), 0.97),
            ("semantic.top_k", lambda c, r, g, s: s.get("top_k"), 120),
            ("semantic.repetition_penalty",
             lambda c, r, g, s: s.get("repetition_penalty"), 1.05),
            ("cfg_scale", lambda c, r, g, s: c.get("cfg_scale"), 1.5),
            ("overrides.cfg_scale",
             lambda c, r, g, s: (c.get("overrides") or {}).get("cfg_scale"), 1.5),
            ("request.seed", lambda c, r, g, s: r.get("seed"), 222222),
        ],
    )

    # 用例 3：外部乐谱 —— 8GB 档默认 cot=off，填了 abc 必须自动抬到 full
    results["3"] = run_case(
        3, "外部 ABC 乐谱 + cot 自动抬升（验证冲突自动修正真的生效）",
        {
            "style": "Mandarin, Ballad, melancholic and wistful, warm expressive female "
                     "vocal, piano, strings, 72 BPM",
            "lyrics": "[Verse]\n旧照片泛了黄\n那年夏天还在吗\n\n[Chorus]\n我们都没说再见",
            "seconds": 20,
            "seed": 333333,
            "options": {
                "temperature": 0.9, "top_p": 0.92, "top_k": 80,
                "repetition_penalty": 1.15, "ode_steps": 32,
                "cfg_scale": None, "cot": "off",
                "abc": ("X:1\nT:Test Ballad\nM:4/4\nL:1/4\nK:C\n"
                        "| C E G c | B G E C | F A c f | e c A F |"
                        "\n| C E G c | B G E C | D F A d | c4 |"),
            },
        },
        [
            ("cot 自动抬到 full", lambda c, r, g, s: c.get("cot"), "full"),
            ("request.cot", lambda c, r, g, s: r.get("cot"), "full"),
            ("request.abc 已传入",
             lambda c, r, g, s: bool(r.get("abc")), True),
            ("ode_steps", lambda c, r, g, s: g.get("ode_steps"), 32),
            ("semantic.temperature", lambda c, r, g, s: s.get("temperature"), 0.9),
            ("request.seed", lambda c, r, g, s: r.get("seed"), 333333),
        ],
    )

    print(f"\n{'=' * 66}\n汇总\n{'=' * 66}")
    for k in sorted(results):
        ok, info = results[k]
        extra = f"  {info.get('elapsed', 0):.0f}s 出 {info.get('seconds', 0):.1f}s 音频" \
            if info else ""
        print(f"  用例 {k}: {'通过' if ok else '未通过'}{extra}")
    return 0 if all(v[0] for v in results.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
