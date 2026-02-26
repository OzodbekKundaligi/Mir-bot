import os
import signal
import subprocess
import sys
import time


def _log(message: str) -> None:
    print(message, flush=True)


def _public_url() -> str:
    explicit = os.getenv("WEBAPP_URL", "").strip()
    if explicit:
        return explicit
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
    if railway_domain:
        if railway_domain.startswith("http://") or railway_domain.startswith("https://"):
            return railway_domain
        return f"https://{railway_domain}"
    return ""


def _terminate_process(name: str, proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _log(f"[shutdown] stopping {name} (pid={proc.pid})")
    proc.terminate()
    try:
        proc.wait(timeout=12)
    except subprocess.TimeoutExpired:
        _log(f"[shutdown] force killing {name} (pid={proc.pid})")
        proc.kill()


def main() -> int:
    web_host = os.getenv("WEB_HOST", "0.0.0.0").strip() or "0.0.0.0"
    web_port = int((os.getenv("PORT") or os.getenv("WEB_PORT") or "8000").strip())
    public_url = _public_url()

    _log("==============================================================")
    _log("[boot] Mir-bot unified service starting")
    _log(f"[boot] Python: {sys.version.split()[0]}")
    _log(f"[boot] WEB API listen: {web_host}:{web_port}")
    if public_url:
        _log(f"[boot] Public URL: {public_url}")
        _log(f"[boot] Health URL: {public_url.rstrip('/')}/health")
    else:
        _log("[boot] Public URL: (not set)")
    _log("[boot] Starting: FastAPI + Telegram bot in one service")
    _log("==============================================================")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "webapp.server.app:app",
        "--host",
        web_host,
        "--port",
        str(web_port),
    ]
    bot_cmd = [sys.executable, "main.py"]

    api_proc = subprocess.Popen(api_cmd, env=env)
    bot_proc = subprocess.Popen(bot_cmd, env=env)
    _log(f"[boot] webapi pid={api_proc.pid}")
    _log(f"[boot] bot pid={bot_proc.pid}")

    stopped = {"done": False}

    def _graceful_exit(signum: int, _frame: object) -> None:
        if stopped["done"]:
            return
        stopped["done"] = True
        _log(f"[signal] received {signum}, stopping children...")
        _terminate_process("bot", bot_proc)
        _terminate_process("webapi", api_proc)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _graceful_exit)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _graceful_exit)

    while True:
        api_code = api_proc.poll()
        bot_code = bot_proc.poll()

        if api_code is not None:
            _log(f"[exit] webapi stopped with code={api_code}")
            _terminate_process("bot", bot_proc)
            return api_code
        if bot_code is not None:
            _log(f"[exit] bot stopped with code={bot_code}")
            _terminate_process("webapi", api_proc)
            return bot_code

        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
