"""Run the actual API and local single-user webpage, without Sites or downloads."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def prepare_environment(node_path=None, data_dir=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    env = os.environ.copy()
    candidates = [node_path, env.get("CAREER_NODE"), ROOT / "data/tools/node22/bin/node", "/tmp/career-node22/node_modules/node/bin/node", shutil.which("node")]
    node = None
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        result = subprocess.run([candidate, "--version"], capture_output=True, text=True, check=True)
        version = tuple(int(part) for part in result.stdout.strip().lstrip("v").split("."))
        if version >= (22, 13, 0):
            node = str(Path(candidate).resolve())
            break
        if node_path or env.get("CAREER_NODE"):
            raise RuntimeError("指定的 Node 版本过低，需要现有的 Node >=22.13；未下载安装。")
    if node is None:
        raise RuntimeError("未找到现有 Node >=22.13；未下载安装。请设置 CAREER_NODE。")
    for dependency in ("fastapi", "uvicorn", "multipart", "torch", "transformers"):
        if importlib.util.find_spec(dependency) is None:
            raise RuntimeError(f"现有 Python 环境缺少 {dependency}；未下载安装。")
    for dependency in ("vite/bin/vite.js", "vinext/dist/cli.js", "@cloudflare/vite-plugin/package.json"):
        if not (ROOT / "web/node_modules" / dependency).is_file():
            raise RuntimeError("网页依赖不完整；未下载安装。")
    known_guidance = ROOT.parents[1] / "models/Qwen3-4B-Instruct-2507"
    if not env.get("GUIDANCE_MODEL_PATH") and known_guidance.is_dir():
        env["GUIDANCE_MODEL_PATH"] = str(known_guidance)
    if not env.get("FEEDBACK_MODEL_PATH"):
        env["FEEDBACK_MODEL_PATH"] = env.get("GUIDANCE_MODEL_PATH", "")
    for name in ("GUIDANCE_MODEL_PATH", "FEEDBACK_MODEL_PATH", "VISION_MODEL_PATH"):
        value = env.get(name, "")
        path = Path(value) if value else None
        if path is not None and not path.is_absolute():
            path = ROOT / path
        if path is None or not path.is_dir():
            raise RuntimeError(f"{name} 未配置为已有权重目录；未下载模型。")
        env[name] = str(path.resolve())
    if not env.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("需要在服务器 .env 中配置 DEEPSEEK_API_KEY，启动检查不会调用 API。")
    root = Path(data_dir or ROOT / "data/local-demo").resolve()
    env.update(
        CAREER_SERVICE_TOKEN=secrets.token_urlsafe(48),
        CAREER_WEB_DATA_DIR=str(root), CAREER_WEB_DB=str(root / "web.sqlite3"),
        CAREER_WEB_UPLOADS=str(root / "uploads"), CAREER_KNOWLEDGE_DIR=str(root / "knowledge"),
        LOCAL_MODEL_DEVICE="cuda:1", BRAIN_RETRY_TIMES="0",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WRANGLER_SEND_METRICS="false",
    )
    return node, env


def check_port(port):
    with socket.socket() as listener:
        # A just-stopped HTTP server can leave TIME_WAIT connections. Like
        # Uvicorn/Vite, allow rebinding those while rejecting live listeners.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"端口 {port} 不可用，请换端口；没有终止已有进程。") from None


def wait_ready(url, processes, timeout=60):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("本地服务提前退出，请检查终端中的错误。")
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(.2)
    raise RuntimeError("本地服务启动超时；未提交或重试模型任务。")


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5180, help="Loopback webpage port")
    parser.add_argument("--api-port", type=int, default=8788, help="Loopback API port")
    parser.add_argument("--node", help="Existing Node >=22.13 executable")
    parser.add_argument("--data-dir", type=Path, help="Separate local demo data directory")
    parser.add_argument("--check", action="store_true", help="Check existing files/config only; no servers or models")
    args = parser.parse_args()
    processes = []
    try:
        os.umask(0o077)
        if args.port == args.api_port or any(not 1024 <= port <= 65535 for port in (args.port, args.api_port)):
            raise RuntimeError("网页与API必须使用不同的1024至65535端口。")
        node, env = prepare_environment(args.node, args.data_dir)
        if args.check:
            print(json.dumps({"status": "ready", "node": node, "device": "cuda:1", "data_dir": env["CAREER_WEB_DATA_DIR"], "downloads": False, "model_calls": 0}, ensure_ascii=False))
            return 0
        check_port(args.port)
        check_port(args.api_port)
        env["CAREER_WEB_PORT"] = str(args.api_port)
        api = subprocess.Popen([sys.executable, "-m", "project.web"], cwd=ROOT, env=env, start_new_session=True)
        processes.append(api)
        wait_ready(f"http://127.0.0.1:{args.api_port}/healthz", processes)
        web_env = env.copy()
        web_env.pop("DEEPSEEK_API_KEY", None)
        web_env.update(CAREER_LOCAL_DEMO="true", CAREER_LOCAL_API_URL=f"http://127.0.0.1:{args.api_port}")
        web_env["PATH"] = str(Path(node).parent) + os.pathsep + web_env.get("PATH", "")
        web = subprocess.Popen([node, "node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", str(args.port), "--strictPort"], cwd=ROOT / "web", env=web_env, start_new_session=True)
        processes.append(web)
        wait_ready(f"http://127.0.0.1:{args.port}/api/sessions", processes)
        print(f"\n本地 Demo 已就绪：http://127.0.0.1:{args.port}\n直接进入单用户工作台，无需 GPT 登录或 Sites。\n点击材料理解、回答、规划或完整阅读状态下的更详细会调用真实模型；简要切换、恢复完整阅读和这样刚好不调用；启动不会调用。\n数据：{env['CAREER_WEB_DATA_DIR']}\n按 Ctrl+C 停止本次启动的两个服务。", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(.5)
        raise RuntimeError("一个本地服务已退出，正在停止配套服务。")
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            # The API waits for its running model task; normal shutdown must
            # not turn a long accepted task into a forced interruption.
            process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
