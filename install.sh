#!/usr/bin/env bash
# YuE2 音乐生成器 —— Linux 安装入口（云 GPU 上用这个）
set -u
cd "$(dirname "$0")"

echo
echo "  ============================================================"
echo "    YuE2 Music Generator  -  Setup  (Linux)"
echo "  ============================================================"
echo

PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$c"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "  [ERROR] Python 3.10+ not found."
  echo
  echo "  Install it first. On Debian/Ubuntu:"
  echo "      apt update && apt install -y python3 python3-venv python3-pip"
  echo
  exit 1
fi

echo "  Using Python: $PY  ($("$PY" -V 2>&1))"
echo

"$PY" install.py "$@"
RC=$?

echo
if [ "$RC" -ne 0 ]; then
  echo "  Setup did NOT finish. Read the messages above."
  echo "  Re-running this script will resume from where it stopped."
else
  echo "  Setup finished."
  echo "    Local machine : ./start.sh"
  echo "    Cloud GPU     : ./start.sh --cloud"
fi
exit "$RC"
