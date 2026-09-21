#!/usr/bin/env bash
# YuE2 音乐生成器 —— Linux 启动入口
#   ./start.sh            本机使用，只监听 127.0.0.1
#   ./start.sh --cloud    云 GPU 使用，监听 0.0.0.0，可从外部访问
set -u
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo
  echo "  Not installed yet. Run ./install.sh first."
  echo
  exit 1
fi

if [ "${1:-}" = "--cloud" ]; then
  export YUE2_BIND=0.0.0.0
  export YUE2_NO_BROWSER=1
  echo
  echo "  Cloud mode: listening on 0.0.0.0"
  echo "  Open it via this machine's public IP or your provider's"
  echo "  port-forwarding address. Port: ${YUE2_PORT:-7861}"
  echo
  echo "  Tip: prefer an SSH tunnel over exposing the port publicly:"
  echo "      ssh -L 7861:127.0.0.1:7861 <user>@<host>"
  echo
fi

export YUE2_PORT="${YUE2_PORT:-7861}"

exec .venv/bin/python app.py
