# Source before training so Ray workers inherit SwanLab credentials.
# Usage:
#   source /mnt/z4/solariewang/verl-swe/scripts/swanlab_env.sh
#
# Put your key in ONE of:
#   export SWANLAB_API_KEY='...'
#   echo '...' > /mnt/z4/solariewang/.swanlab_api_key
#   export SWANLAB_API_KEY_FILE=/path/to/keyfile

ROOT=${ROOT:-/mnt/z4/solariewang}
if [ ! -f "$ROOT/verl-swe/scripts/swanlab_env.sh" ] && [ -f "/apdcephfs/z4/solariewang/verl-swe/scripts/swanlab_env.sh" ]; then
    ROOT=/apdcephfs/z4/solariewang
fi
REPO=${REPO:-$ROOT/verl-swe}

export SWANLAB_MODE=${SWANLAB_MODE:-local}
# SwanLab SDK uses "local" for offline-only logging; accept "offline" as alias.
if [ "${SWANLAB_MODE}" = "offline" ]; then
    SWANLAB_MODE=local
fi
export SWANLAB_MODE
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$REPO/swanlog}

# Many training nodes set a global HTTP(S) proxy that breaks SwanLab uploads
# (ProxyError / RemoteDisconnected on api.swanlab.cn). Bypass proxy for SwanLab.
_swans=("api.swanlab.cn" "swanlab.cn" ".swanlab.cn" "localhost" "127.0.0.1")
for _h in "${_swans[@]}"; do
    if [[ ":${NO_PROXY:-}:${no_proxy:-}:" != *":${_h}:"* ]]; then
        NO_PROXY="${NO_PROXY:+$NO_PROXY,}${_h}"
        no_proxy="${no_proxy:+$no_proxy,}${_h}"
    fi
done
export NO_PROXY no_proxy

# Set SWANLAB_DISABLE_PROXY=1 to drop http(s)_proxy entirely for this shell.
if [ "${SWANLAB_DISABLE_PROXY:-0}" = "1" ]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
fi

_key_file="${SWANLAB_API_KEY_FILE:-$ROOT/.swanlab_api_key}"
if [ -z "${SWANLAB_API_KEY:-}" ] && [ -f "$_key_file" ]; then
    SWANLAB_API_KEY="$(tr -d '[:space:]' < "$_key_file")"
    export SWANLAB_API_KEY
fi

if [ "${SWANLAB_MODE}" = "cloud" ] || [ "${SWANLAB_MODE}" = "online" ]; then
    if [ -z "${SWANLAB_API_KEY:-}" ]; then
        echo "ERROR: SWANLAB_API_KEY is empty (cloud mode requires a key)." >&2
        echo "  1) export SWANLAB_API_KEY='your-key'" >&2
        echo "  2) echo 'your-key' > $_key_file" >&2
        echo "  Get key: https://swanlab.cn/settings" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
