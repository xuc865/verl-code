# Source before GPT / xmcode.shop API evals.
# Usage:
#   source /mnt/z4/solariewang/verl-swe/scripts/xmcode_env.sh
#
# Key file (do not commit):
#   /mnt/z4/solariewang/.xmcode_api_key

ROOT=${ROOT:-/mnt/z4/solariewang}
_key_file="${XMCODE_API_KEY_FILE:-$ROOT/.xmcode_api_key}"
if [ -z "${API_KEY:-}" ] && [ -f "$_key_file" ]; then
  API_KEY="$(tr -d '[:space:]' < "$_key_file")"
  export API_KEY
fi
export API_BASE="${API_BASE:-https://xmcode.shop/v1}"
# Cloudflare on xmcode rejects empty/default UA in some paths; eval client sets UA too.
export OPENAI_API_KEY="${OPENAI_API_KEY:-$API_KEY}"
