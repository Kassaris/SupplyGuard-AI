"""
Docker sandbox backend for the Deep Agent (Week 3, day 9: "Sandboxing:
running code with sandbox backends").

WHY A SANDBOX
-------------
Some questions are easiest to answer by writing a short script, e.g.
"allocate the 3,200 usable units over 77 orders, tier-1 first, and show who
is late".  The Deep Agent gets an `execute` tool when its backend implements
`SandboxBackendProtocol`.  Running LLM-written code on the HOST would let a
prompt injection run anything on your PC, so we never use `LocalShellBackend`.

WHY DOCKER (not LangSmith / Modal / Daytona sandboxes)
------------------------------------------------------
Docker is already part of the project (Postgres, Neo4j), runs offline, costs
nothing and the hardening is visible in 10 lines below.  The hosted sandbox
backends implement the same `BaseSandbox` interface, so switching is a
one-line change.

HARDENING (each flag closes one attack path)
--------------------------------------------
  --network none              no exfiltration, no downloads
  --user 65534:65534          runs as 'nobody', not root
  --cap-drop ALL              no Linux capabilities
  --security-opt no-new-privileges   no setuid escalation
  --read-only + --tmpfs       image is immutable; only /workspace & /tmp writable (RAM, size-capped)
  --memory / --cpus / --pids-limit   no resource exhaustion / fork bombs
  per-command timeout         no infinite loops
  container removed on close  nothing survives the case

HOW BaseSandbox WORKS
---------------------
We only implement 4 methods: `execute`, `id`, `upload_files`, `download_files`.
deepagents' `BaseSandbox` builds ls/read/write/edit/grep/glob on top of
`execute` (it runs small python3 snippets inside the container), so the
agent's normal file tools also work inside the sandbox.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

DEFAULT_IMAGE = "python:3.12-slim"
WORKDIR = "/workspace"
MAX_OUTPUT_CHARS = 50_000  # keep tool output small enough for the context window


class SandboxUnavailable(RuntimeError):
    """Docker is not running or the image is not present locally."""


def docker_image_available(image: str = DEFAULT_IMAGE) -> bool:
    """True if Docker runs and the image is already pulled (we never auto-pull)."""
    try:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=15)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerSandbox(BaseSandbox):
    """One hardened, throw-away container per investigation (local experiments).

    In docker compose the app can't start containers (no Docker socket - on
    purpose), so production uses HttpSandbox below instead.
    """

    workdir = WORKDIR

    def __init__(self, image: str = DEFAULT_IMAGE, timeout: int = 60, memory: str = "512m", cpus: str = "1.0") -> None:
        self.image = image
        self.default_timeout = timeout
        self.memory = memory
        self.cpus = cpus
        self._name = f"supplyguard-sbx-{uuid.uuid4().hex[:10]}"
        self._started = False

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "DockerSandbox":
        """Create the container.  It only runs `sleep`; commands arrive via `docker exec`."""
        if not docker_image_available(self.image):
            raise SandboxUnavailable(f"Docker image {self.image} not available. Run: docker pull {self.image}")
        cmd = [
            "docker", "run", "-d", "--rm", "--name", self._name,
            "--network", "none",
            "--user", "65534:65534",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", f"{WORKDIR}:rw,size=64m,uid=65534,gid=65534",
            "--tmpfs", "/tmp:rw,size=32m",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "128",
            "-w", WORKDIR,
            self.image, "sleep", "infinity",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise SandboxUnavailable(r.stderr.strip())
        self._started = True
        return self

    def close(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self._name], capture_output=True, timeout=30)
            self._started = False

    def __enter__(self) -> "DockerSandbox":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------- BaseSandbox API
    @property
    def id(self) -> str:
        return self._name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command inside the container (never on the host)."""
        if not self._started:
            return ExecuteResponse(output="sandbox not started", exit_code=1)
        limit = timeout or self.default_timeout
        try:
            r = subprocess.run(
                ["docker", "exec", "-w", WORKDIR, self._name, "sh", "-c", command],
                capture_output=True, timeout=limit,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(output=f"command timed out after {limit}s", exit_code=124)
        out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
        truncated = len(out) > MAX_OUTPUT_CHARS
        return ExecuteResponse(output=out[:MAX_OUTPUT_CHARS], exit_code=r.returncode, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write files by piping bytes into `cat` inside the container.

        The path is passed as a positional argument ($1) - never pasted into
        the shell string - so a malicious file name can't inject commands.
        """
        results = []
        for path, data in files:
            r = subprocess.run(
                ["docker", "exec", "-i", self._name, "sh", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "sh", path],
                input=data, capture_output=True, timeout=30,
            )
            results.append(FileUploadResponse(path=path, error=None if r.returncode == 0 else "permission_denied"))
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for path in paths:
            r = subprocess.run(["docker", "exec", self._name, "cat", path], capture_output=True, timeout=30)
            if r.returncode == 0:
                results.append(FileDownloadResponse(path=path, content=r.stdout))
            else:
                results.append(FileDownloadResponse(path=path, error="file_not_found"))
        return results

    def upload_data_tables(self, data_dir: Path, tables: list[str]) -> list[str]:
        return upload_data_tables(self, data_dir, tables)


# Tables useful for allocation / timing analysis (no finance-confidential data).
ANALYSIS_TABLES = ["customer_orders", "customers", "inventory", "reservations", "bill_of_materials",
                   "production_schedule", "supplier_components", "purchase_orders"]


def upload_data_tables(sandbox, data_dir: Path, tables: list[str]) -> list[str]:
    """Copy selected operational CSVs to <workdir>/data for analysis scripts.

    Only internal operational tables - documents (with access scopes) are
    never copied into a sandbox; agents read those through scoped tools.
    """
    files = [(f"{sandbox.workdir}/data/{t}.csv", (data_dir / "structured" / f"{t}.csv").read_bytes()) for t in tables]
    failed = [r.path for r in sandbox.upload_files(files) if r.error]
    if failed:
        raise SandboxUnavailable(f"upload failed for {failed}")
    return [p for p, _ in files]


# =========================================================================== HTTP sandbox (compose)
class HttpSandbox(BaseSandbox):
    """Client of the sandbox service (src/supplyguard_ai/sandbox/server.py).

    Same four BaseSandbox methods as DockerSandbox, implemented as HTTP calls.
    One server-side SESSION (a private workdir) per investigation; `close()`
    deletes it.  Sync httpx client: deepagents calls these methods from its
    tool threads.
    """

    def __init__(self, url: str, timeout: int = 30) -> None:
        import httpx

        self.url = url.rstrip("/")
        self.default_timeout = timeout
        self._http = httpx.Client(timeout=timeout + 15)
        self.session_id: str | None = None
        self.workdir: str = ""

    def start(self) -> "HttpSandbox":
        import httpx

        try:
            r = self._http.post(f"{self.url}/sessions")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise SandboxUnavailable(f"sandbox service unreachable at {self.url}: {exc}") from exc
        data = r.json()
        self.session_id, self.workdir = data["session_id"], data["workdir"]
        return self

    def close(self) -> None:
        if self.session_id:
            try:
                self._http.delete(f"{self.url}/sessions/{self.session_id}")
            finally:
                self.session_id = None
        self._http.close()

    def __enter__(self) -> "HttpSandbox":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def _post(self, op: str, body: dict) -> dict:
        r = self._http.post(f"{self.url}/sessions/{self.session_id}/{op}", json=body)
        r.raise_for_status()
        return r.json()

    @property
    def id(self) -> str:
        return f"http-sandbox:{self.session_id}"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        import httpx

        try:
            d = self._post("execute", {"command": command, "timeout": timeout or self.default_timeout})
        except httpx.HTTPError as exc:  # report to the agent instead of crashing the run
            return ExecuteResponse(output=f"sandbox error: {exc}", exit_code=1)
        return ExecuteResponse(output=d["output"], exit_code=d["exit_code"], truncated=d["truncated"])

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        import base64

        out = []
        for path, data in files:
            d = self._post("upload", {"path": path, "content_b64": base64.b64encode(data).decode()})
            out.append(FileUploadResponse(path=path, error=d.get("error")))
        return out

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        import base64

        out = []
        for path in paths:
            d = self._post("download", {"path": path})
            if d.get("error"):
                out.append(FileDownloadResponse(path=path, error=d["error"]))
            else:
                out.append(FileDownloadResponse(path=path, content=base64.b64decode(d["content_b64"])))
        return out


def make_sandbox(settings) -> BaseSandbox | None:
    """The sandbox for one investigation, or None (then the agent has no `execute` tool).
    SANDBOX_URL set (docker compose) -> HttpSandbox; otherwise no sandbox."""
    if not getattr(settings, "sandbox_url", None):
        return None
    return HttpSandbox(settings.sandbox_url).start()

__all__ = ["DockerSandbox", "HttpSandbox", "SandboxUnavailable", "docker_image_available", "ANALYSIS_TABLES",
           "make_sandbox", "upload_data_tables"]
