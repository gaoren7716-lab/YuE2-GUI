#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 YuE2-GUI 打包成一个精简 zip，用于上传到云 GPU。

只打包源码和启动脚本（约 200 KB）。
虚拟环境、模型权重、下载缓存都不打 —— 那些在云上重新生成更快，
而且模型走镜像下载只要约 2 分钟，比上传 7.3 GB 快得多。

用法：
    python pack.py                生成 YuE2-GUI-upload.zip
    python pack.py --out D:\\xx.zip  指定输出路径
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# 这些是要带上的
INCLUDE = [
    "app.py",
    "ram_mode.py",
    "install.py",
    "selftest.py",
    "benchmark.py",
    "install.bat",
    "start.bat",
    "start_cloud.bat",
    "install.sh",
    "start.sh",
]

# 这些绝对不能打进去
EXCLUDE_DIRS = {".venv", "models", "_cache", "outputs", "__pycache__", ".git"}
EXCLUDE_FILES = {".installed", "pack.py"}


def collect():
    items = []
    for name in INCLUDE:
        path = ROOT / name
        if path.exists():
            items.append(path)
        else:
            print(f"  警告：缺少 {name}")
    # 推理包也带上，云上就不用再下一次
    for whl in sorted(ROOT.glob("yue2_infer-*.whl")):
        items.append(whl)
    return items


def main():
    ap = argparse.ArgumentParser(description="打包用于上传云 GPU 的精简 zip")
    ap.add_argument("--out", default=str(ROOT / "YuE2-GUI-upload.zip"))
    args = ap.parse_args()

    items = collect()
    if not items:
        print("没有可打包的文件。")
        return 1

    out = Path(args.out)
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in items:
            arc = f"YuE2-GUI/{path.name}"
            z.write(path, arc)
            total += path.stat().st_size
            print(f"  + {arc}  ({path.stat().st_size / 1024:.1f} KB)")

    size = out.stat().st_size
    print()
    print(f"打包完成：{out}")
    print(f"  原始合计：{total / 1024:.1f} KB")
    print(f"  压缩后  ：{size / 1024:.1f} KB")
    print()
    print("上传后，在云主机上执行：")
    print("  unzip YuE2-GUI-upload.zip")
    print("  cd YuE2-GUI")
    print("  chmod +x install.sh start.sh")
    print("  ./install.sh")
    print("  ./start.sh --cloud")
    return 0


if __name__ == "__main__":
    sys.exit(main())
