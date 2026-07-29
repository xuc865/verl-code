#!/usr/bin/env bash
# Find free TCP ports on the current host (for vLLM / eval API).
#
# Usage (train host):
#   bash /mnt/z4/solariewang/verl-swe/scripts/find_free_port.sh
#   bash .../find_free_port.sh 8000 8001 8080        # check specific ports
#   bash .../find_free_port.sh --find 3               # pick 3 free ports
#   bash .../find_free_port.sh --find 1 --start 8000  # from 8000 upward
#   PORT=$(bash .../find_free_port.sh --one)          # print one free port only
#
# Env:
#   START=8000 END=9000 COUNT=1

set -euo pipefail

START=${START:-8000}
END=${END:-9000}
COUNT=${COUNT:-1}
MODE=auto          # auto | check | find | one
PORTS=()

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

is_free() {
  local p=$1
  # Prefer ss; fall back to python bind test (works without root / ss).
  if command -v ss >/dev/null 2>&1; then
    if ss -lntu 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${p}$"; then
      return 1
    fi
    return 0
  fi
  python3 - "$p" <<'PY'
import socket, sys
p = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", p))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

who_holds() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ss -lntp 2>/dev/null | awk -v p=":$p" '$4 ~ p"$" {print; found=1} END{exit !found}' \
      || ss -lnt 2>/dev/null | awk -v p=":$p" '$4 ~ p"$"' || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$p" -sTCP:LISTEN 2>/dev/null || true
  else
    echo "(no ss/lsof; port busy)"
  fi
}

find_ports() {
  local need=$1 start=$2 end=$3
  local found=() p
  for ((p=start; p<=end; p++)); do
    if is_free "$p"; then
      found+=("$p")
      (( ${#found[@]} >= need )) && break
    fi
  done
  if (( ${#found[@]} < need )); then
    echo "ERROR: only found ${#found[@]}/$need free ports in [$start,$end]" >&2
    return 1
  fi
  printf '%s\n' "${found[@]}"
}

# ---- args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --find) MODE=find; COUNT=${2:?}; shift 2 ;;
    --one) MODE=one; COUNT=1; shift ;;
    --start) START=${2:?}; shift 2 ;;
    --end) END=${2:?}; shift 2 ;;
    --check) MODE=check; shift ;;
    [0-9]*) PORTS+=("$1"); MODE=check; shift ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

hostname_s=$(hostname 2>/dev/null || echo unknown)

if [[ "$MODE" == "one" ]]; then
  find_ports 1 "$START" "$END"
  exit $?
fi

if [[ "$MODE" == "find" || ( "$MODE" == "auto" && ${#PORTS[@]} -eq 0 ) ]]; then
  echo "==> host=$hostname_s  scan=[$START,$END]  need=$COUNT"
  mapfile -t found < <(find_ports "$COUNT" "$START" "$END")
  echo "FREE:"
  printf '  %s\n' "${found[@]}"
  echo
  echo "export PORT=${found[0]}"
  if (( ${#found[@]} > 1 )); then
    echo "# others: ${found[*]}"
  fi
  exit 0
fi

# MODE=check specific ports
echo "==> host=$hostname_s  check ports: ${PORTS[*]}"
any_busy=0
for p in "${PORTS[@]}"; do
  if is_free "$p"; then
    echo "  FREE  $p"
  else
    echo "  BUSY  $p"
    who_holds "$p" | sed 's/^/         /'
    any_busy=1
  fi
done
if (( any_busy == 0 )); then
  echo
  echo "All listed ports are free."
  echo "export PORT=${PORTS[0]}"
else
  echo
  echo "Some ports busy — pick a free one:"
  mapfile -t found < <(find_ports 1 "$START" "$END")
  echo "  suggest: ${found[0]}"
  echo "  export PORT=${found[0]}"
  exit 1
fi
