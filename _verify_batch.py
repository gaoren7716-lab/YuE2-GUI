"""验证批量生成：一次出 2 首，检查 job 状态机、种子递增、产出目录。

关注三件事：
1. count 参数真的被后端接住了
2. 批量过程中 status 始终是 running（不会中途跳 done 让前端提前收摊）
3. 两首的种子不同、各自落了目录
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


def main():
    st = get("/api/status")
    if not st.get("installed"):
        print("模型未就绪，退出")
        return 1

    before = {p.name for p in OUTPUTS.iterdir() if p.is_dir()} if OUTPUTS.exists() else set()

    payload = {
        "style": "Mandarin, Lo-fi, warm expressive female vocal, piano, 80 BPM",
        "lyrics": "[Verse]\n夜色落在窗台上\n一盏灯陪我发呆",
        "seconds": 10,
        "seed": 555001,
        "count": 2,
        "options": {"cot": "off", "ode_steps": 16},
    }
    code, body = post("/api/generate", payload)
    print("提交 ->", code, body)
    if body.get("count") != 2:
        print("BAD：count 没传到后端，实得", body.get("count"))
        return 2

    t0 = time.perf_counter()
    last = None
    saw_running_with_done = False
    while time.perf_counter() - t0 < 900:
        j = get("/api/job")
        key = (j.get("status"), j.get("batch_index"), j.get("batch_total"),
               len(j.get("batch_done") or []))
        if key != last:
            print(f"  [{time.perf_counter() - t0:6.1f}s] status={key[0]} "
                  f"第 {key[1]}/{key[2]} 首  已完成 {key[3]}  "
                  f"{(j.get('message') or '')[:38]}", flush=True)
            last = key
        # 第 1 首做完时，status 必须还是 running，否则前端会提前收摊
        if (j.get("status") == "running" and (j.get("batch_done") or [])
                and j.get("batch_index") == 2):
            saw_running_with_done = True
        if j.get("status") in ("done", "error"):
            break
        time.sleep(3)

    j = get("/api/job")
    if j.get("status") != "done":
        print("失败：", j.get("status"), j.get("message"), j.get("error"))
        return 3

    done = j.get("batch_done") or []
    after = {p.name for p in OUTPUTS.iterdir() if p.is_dir()}
    new = sorted(after - before)

    ok = True

    def chk(label, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {label}  {extra}")

    print()
    chk("batch_done 有 2 首", len(done) == 2, f"{len(done)} 首")
    chk("batch_total = 2", j.get("batch_total") == 2, str(j.get("batch_total")))
    chk("新增 2 个输出目录", len(new) == 2, str(new))
    seeds = [d.get("seed") for d in done]
    chk("两首种子不同", len(set(seeds)) == 2, str(seeds))
    chk("每首都有音频链接", all(d.get("audio_url") for d in done))
    chk("中间态保持 running（前端不会提前收摊）", saw_running_with_done,
        "已观测到" if saw_running_with_done else "未观测到")
    print("\n批量结果：", json.dumps(
        [{k: d.get(k) for k in ("seed", "seconds", "audio_url")} for d in done],
        ensure_ascii=False, indent=2))
    print("结果：", "通过" if ok else "未通过")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
