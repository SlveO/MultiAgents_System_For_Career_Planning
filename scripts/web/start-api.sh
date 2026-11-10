#!/usr/bin/env bash
set -euo pipefail
task_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$task_repo_dir"
task_python="${CAREER_PYTHON:-$task_repo_dir/.venv-l20/bin/python}"
if [[ ! -x "$task_python" ]]; then
    printf '%s\n' '设置 CAREER_PYTHON 为已有、经过验证的 GPU Python 环境。' >&2
    exit 1
fi
exec "$task_python" -m project.web
