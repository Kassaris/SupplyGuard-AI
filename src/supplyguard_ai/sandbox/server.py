#!/usr/bin/env python3
"""
SupplyGuard analysis sandbox - a tiny HTTP service that runs agent-written
shell / Python commands in an isolated container.

Deployed by docker-compose.yml (service `sandbox`):
    image python:3.12.10-slim, ONLY this file mounted read-only, run as
    `python /sandbox/server.py`.  So this file is SELF-CONTAINED: standard
    library only, no imports from the supplyguard_ai package.

WHY AN HTTP SERVICE (and not `docker exec` from the app)
--------------------------------------------------------
In compose the app runs in a container itself; giving it the Docker socket
to start sandboxes would hand it root on the host.  Instead the sandbox is a
long-running, locked-down container the app talks to over HTTP on an
INTERNAL network (`sandbox_net`, internal: true -> no route to the internet).

DEFENCE IN DEPTH (container + this server)
------------------------------------------
container (compose):  read-only root fs, writable /tmp only (tmpfs), user
                      nobody (65534), no Linux capabilities, no-new-privileges,
                      512 MB RAM, 128 processes, 1 CPU, no internet
this server:          per-command timeout, CPU-seconds / file-size / open-file
                      rlimits, output cap, request-size cap, max 4 commands
                      at once, path traversal checks on upload/download,
                      unguessable session ids in a non-listable (0311) directory,
                      sessions deleted on close or after 2 h idle,
                      empty environment for commands (no secrets to leak -
                      the container never receives any anyway)

API (JSON over HTTP, port 8200)
-------------------------------
  GET    /health
  POST   /sessions                         -> {"session_id", "workdir"}
  DELETE /sessions/<id>
  POST   /sessions/<id>/execute   {"command", "timeout"?}   -> {"output", "exit_code", "truncated"}
  POST   /sessions/<id>/upload    {"path", "content_b64"}   -> {"path", "error"}
  POST   /sessions/<id>/download  {"path"}                  -> {"path", "content_b64" | "error"}
Paths are absolute and must lie inside the session's workdir.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # Linux only (the container); lets unit tests import this file on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None

ROOT = os.environ.get("SANDBOX_ROOT", "/tmp/workspace")
PORT = int(os.environ.get("SANDBOX_PORT", "8200"))
MAX_TIMEOUT_S = 60
DEFAULT_TIMEOUT_S = 30
MAX_OUTPUT = 50_000           # chars returned per command
MAX_BODY = 6 * 1024 * 1024    # bytes per request (uploads are base64)
MAX_FILE = 20 * 1024 * 1024   # bytes a command may write per file (RLIMIT_FSIZE)
SESSION_TTL_S = 2 * 3600
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
SLOTS = threading.BoundedSemaphore(4)  # max concurrent commands

_last_used: dict[str, float] = {}
_lock = threading.Lock()


# =========================================================================== sessions & paths
def workdir(session_id: str) -> str:
    if not SESSION_RE.match(session_id or ""):
        raise ValueError("invalid session id")
    return os.path.join(ROOT, session_id)


def create_session() -> dict:
    _lock_root()
    sid = secrets.token_urlsafe(16)
    wd = workdir(sid)
    os.makedirs(wd, mode=0o700)
    with _lock:
        _last_used[sid] = time.time()
    return {"session_id": sid, "workdir": wd}


def _lock_root() -> None:
    """Make /tmp/workspace creatable and traversable but NOT listable.

    The server and the commands run as the same user (nobody), so ordinary
    "other users can't read" permissions don't separate sessions.  Mode 0o311
    removes READ even for the owner: nobody can list the folder, but can still
    create and enter a session folder whose (128-bit random) name it knows.
    The server never needs to list it - it tracks sessions in memory.
    (Found by the live probe: with 0o711 `ls /tmp/workspace` still worked.)
    Residual risk, documented: while two commands run at the same time, one
    could find the other's folder through /proc (same user).  Full isolation
    would need one container per session, i.e. Docker-socket access for the
    app - a far bigger risk than this one.
    """
    os.makedirs(ROOT, exist_ok=True)
    os.chmod(ROOT, 0o311)


def delete_session(session_id: str) -> None:
    shutil.rmtree(workdir(session_id), ignore_errors=True)
    with _lock:
        _last_used.pop(session_id, None)


def touch(session_id: str) -> str:
    wd = workdir(session_id)
    if not os.path.isdir(wd):
        raise FileNotFoundError("unknown session")
    with _lock:
        _last_used[session_id] = time.time()
    return wd


def reap_idle(now: float | None = None) -> list[str]:
    """Delete sessions idle for longer than SESSION_TTL_S (called on each request)."""
    now = now or time.time()
    with _lock:
        stale = [s for s, t in _last_used.items() if now - t > SESSION_TTL_S]
    for s in stale:
        delete_session(s)
    return stale


def safe_path(wd: str, path: str) -> str:
    """Resolve `path` and refuse anything outside the session workdir
    ('../../etc/passwd', symlink tricks, other sessions)."""
    candidate = path if os.path.isabs(path) else os.path.join(wd, path)
    real = os.path.realpath(candidate)
    root = os.path.realpath(wd)
    if real != root and not real.startswith(root + os.sep):
        raise PermissionError("path outside the session workdir")
    return real


# =========================================================================== execution
def _limits() -> None:  # pragma: no cover - runs in the child process on Linux
    """Per-command resource limits, applied in the child before exec."""
    if resource is None:
        return
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_TIMEOUT_S, MAX_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.setsid()  # own process group -> the whole tree can be killed on timeout


def execute(session_id: str, command: str, timeout: int | None = None) -> dict:
    wd = touch(session_id)
    limit = max(1, min(int(timeout or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": wd, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    if not SLOTS.acquire(timeout=10):
        return {"output": "sandbox busy, try again", "exit_code": 75, "truncated": False}
    try:
        proc = subprocess.Popen(["sh", "-c", command], cwd=wd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, preexec_fn=_limits if resource else None)
        try:
            out, _ = proc.communicate(timeout=limit)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            out, code = (proc.communicate()[0] or b"") + f"\ncommand timed out after {limit}s".encode(), 124
    finally:
        SLOTS.release()
    text = out.decode("utf-8", errors="replace")
    return {"output": text[:MAX_OUTPUT], "exit_code": code, "truncated": len(text) > MAX_OUTPUT}


def _kill_tree(proc) -> None:
    try:
        if resource is not None:
            os.killpg(proc.pid, 9)
        else:  # pragma: no cover
            proc.kill()
    except ProcessLookupError:
        pass


def upload(session_id: str, path: str, content_b64: str) -> dict:
    wd = touch(session_id)
    try:
        target = safe_path(wd, path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(base64.b64decode(content_b64))
        return {"path": path, "error": None}
    except PermissionError:
        return {"path": path, "error": "permission_denied"}
    except (ValueError, OSError):
        return {"path": path, "error": "invalid_path"}


def download(session_id: str, path: str) -> dict:
    wd = touch(session_id)
    try:
        target = safe_path(wd, path)
        with open(target, "rb") as fh:
            return {"path": path, "content_b64": base64.b64encode(fh.read(MAX_BODY)).decode()}
    except PermissionError:
        return {"path": path, "error": "permission_denied"}
    except IsADirectoryError:
        return {"path": path, "error": "is_directory"}
    except OSError:
        return {"path": path, "error": "file_not_found"}


# =========================================================================== HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "SupplyGuardSandbox/1.0"

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise OverflowError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _route(self, method: str) -> None:
        reap_idle()
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        try:
            if method == "GET" and parts == ["health"]:
                return self._send(200, {"status": "ok"})
            if method == "POST" and parts == ["sessions"]:
                return self._send(201, create_session())
            if len(parts) >= 2 and parts[0] == "sessions":
                sid = parts[1]
                if method == "DELETE" and len(parts) == 2:
                    delete_session(sid)
                    return self._send(200, {"deleted": True})
                if method == "POST" and len(parts) == 3:
                    body = self._body()
                    if parts[2] == "execute":
                        return self._send(200, execute(sid, str(body["command"]), body.get("timeout")))
                    if parts[2] == "upload":
                        return self._send(200, upload(sid, str(body["path"]), str(body["content_b64"])))
                    if parts[2] == "download":
                        return self._send(200, download(sid, str(body["path"])))
            self._send(404, {"error": "not found"})
        except FileNotFoundError:
            self._send(404, {"error": "unknown session"})
        except OverflowError as exc:
            self._send(413, {"error": str(exc)})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"bad request: {exc}"})

    def do_GET(self):  # noqa: N802
        self._route("GET")

    def do_POST(self):  # noqa: N802
        self._route("POST")

    def do_DELETE(self):  # noqa: N802
        self._route("DELETE")

    def log_message(self, fmt, *args):  # one compact line per request: method, route, status
        route = re.sub(r"/sessions/[^/]+", "/sessions/<id>", self.path)  # don't log session ids
        sys.stderr.write(f"sandbox {self.command} {route} {args[1] if len(args) > 1 else ''}\n")


def main() -> None:
    _lock_root()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    sys.stderr.write(f"sandbox listening on :{PORT}, workspaces in {ROOT}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
