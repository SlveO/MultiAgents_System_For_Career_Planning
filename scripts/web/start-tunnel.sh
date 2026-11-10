#!/usr/bin/env bash
set -euo pipefail
task_cloudflared="${CAREER_CLOUDFLARED:-cloudflared}"
if ! command -v "$task_cloudflared" >/dev/null 2>&1; then
    printf '%s\n' '未找到 cloudflared。安装官方客户端或设置 CAREER_CLOUDFLARED 后重试。' >&2
    exit 1
fi
# A random HTTPS URL is printed by cloudflared. No service token is passed here.
exec "$task_cloudflared" tunnel --no-autoupdate --url http://127.0.0.1:8787
