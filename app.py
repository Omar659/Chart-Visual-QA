"""Orchestrator for Chart-Visual-QA — two modes (see docs/ARCHITECTURE.md).

    python app.py            PRODUCTION (default): backend + vLLM guard in Docker
                             (CLIP + Tesseract baked into the backend image); the
                             frontend runs as a local Vite server on :5173. Warns
                             and stops if the Docker engine (Docker Desktop) is down.

    python app.py --dev      LOCAL DEV: venv backend, local CLIP, local **Ollama**
                             guard, local Vite — no Docker. Self-bootstrapping.

Dev-mode flags (ignored in production):
    python app.py --dev --backend-only / --frontend-only / --setup-only /
                  --no-setup / --no-guard / --backend-port 5001 --frontend-port 5174

`--dev` first run is self-bootstrapping: creates the Python 3.12 virtualenv, installs the
backend + input-guard models, runs `npm install`, copies .env.example -> .env, and (if
Ollama is running) pulls llama-guard3:1b. Every guard layer fails open (allows + warns)
when its model/service is missing. See docs/PLAN.md and docs/TASK_B_LAYER3.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
IS_WINDOWS = os.name == "nt"

# Backend targets Python 3.12 specifically (see docs/PLAN.md / README).
PY_VERSION = "3.12"


def _venv_python() -> Path:
    """Path to the backend venv interpreter (may not exist yet)."""
    if IS_WINDOWS:
        return BACKEND_DIR / ".venv" / "Scripts" / "python.exe"
    return BACKEND_DIR / ".venv" / "bin" / "python"


def _backend_python() -> str:
    """Path to the backend interpreter: the venv if present, else this one."""
    venv_py = _venv_python()
    return str(venv_py) if venv_py.exists() else sys.executable


def _find_python_312() -> list[str]:
    """Locate a Python 3.12 launcher command to build the venv with.

    Prefers the Windows `py -3.12` launcher, then a `python3.12` on PATH, then
    falls back to the interpreter running this script.
    """
    if IS_WINDOWS and shutil.which("py"):
        try:
            subprocess.run(
                ["py", f"-{PY_VERSION}", "--version"],
                check=True, capture_output=True, text=True,
            )
            return ["py", f"-{PY_VERSION}"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    exe = shutil.which(f"python{PY_VERSION}")
    if exe:
        return [exe]
    print(
        f"[setup] WARNING: could not find Python {PY_VERSION}; "
        f"using {sys.executable} instead.",
        flush=True,
    )
    return [sys.executable]


def _run(cmd: list[str], cwd: Path) -> None:
    """Run a setup command synchronously, streaming its output; raise on failure."""
    print(f"[setup] $ {' '.join(cmd)}  (cwd={cwd.name})", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_backend_deps() -> None:
    """Create backend/.venv and install requirements if not already present."""
    venv_py = _venv_python()
    if not venv_py.exists():
        print("[setup] backend virtualenv missing; creating it...", flush=True)
        _run([*_find_python_312(), "-m", "venv", ".venv"], BACKEND_DIR)
        _run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"], BACKEND_DIR)
        _run([str(venv_py), "-m", "pip", "install", "-r", "requirements.txt"], BACKEND_DIR)
        print("[setup] backend ready.", flush=True)


def ensure_frontend_deps() -> None:
    """Run `npm install` if frontend/node_modules is missing."""
    if not (FRONTEND_DIR / "node_modules").exists():
        print("[setup] frontend node_modules missing; running npm install...", flush=True)
        npm = "npm.cmd" if IS_WINDOWS else "npm"
        _run([npm, "install"], FRONTEND_DIR)
        print("[setup] frontend ready.", flush=True)


def ensure_env_file() -> None:
    """Create .env from .env.example on first run so config exists out of the box."""
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists() and example.exists():
        env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print("[setup] created .env from .env.example (edit it to configure).", flush=True)


def ensure_guard_deps() -> None:
    """Install the Layer-2 guard models into the venv (default; skip with --no-guard)."""
    if _venv_can_import(["detoxify", "transformers", "presidio_analyzer"]):
        return  # already installed
    venv_py = str(_venv_python())
    print("[setup] installing input-guard models (one-time: torch/transformers/presidio)...",
          flush=True)
    _run([venv_py, "-m", "pip", "install", "-r", "requirements-guard.txt"], BACKEND_DIR)
    _run([venv_py, "-m", "spacy", "download", "en_core_web_lg"], BACKEND_DIR)
    print("[setup] input-guard models installed.", flush=True)


def _ollama_exe() -> str | None:
    """Path to the ollama CLI, or None (PATH, then the Windows default install dir)."""
    exe = shutil.which("ollama")
    if exe:
        return exe
    if IS_WINDOWS:
        win = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if win.exists():
            return str(win)
    return None


def ensure_ollama_model() -> None:
    """Pull the Layer-3 guard model if Ollama is running and the model isn't present."""
    reachable, has_model = _ollama_status(os.environ.get("GUARD_LLM_URL", "http://localhost:11434"))
    if not reachable or has_model:
        return  # unreachable -> readiness report warns; already present -> nothing to do
    exe = _ollama_exe()
    if not exe:
        return
    print("[setup] pulling Layer-3 guard model llama-guard3:1b (one-time)...", flush=True)
    subprocess.run([exe, "pull", "llama-guard3:1b"])


def ensure_deps(run_backend: bool, run_frontend: bool) -> None:
    """Install whatever the requested servers need before launching them."""
    ensure_env_file()
    if run_backend:
        ensure_backend_deps()
    if run_frontend:
        ensure_frontend_deps()


def _venv_can_import(modules: list[str]) -> bool:
    """True if the backend venv can import all the given modules."""
    venv_py = _venv_python()
    if not venv_py.exists():
        return False
    code = "import " + ", ".join(modules)
    return subprocess.run([str(venv_py), "-c", code], capture_output=True).returncode == 0


def _ollama_status(url: str = "http://localhost:11434") -> tuple[bool, bool]:
    """(reachable, has_llama_guard3_model)."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read() or b"{}")
        names = [m.get("name", "") for m in data.get("models", [])]
        return True, any("llama-guard3" in n for n in names)
    except Exception:  # noqa: BLE001
        return False, False


def report_guard_readiness() -> None:
    """Print whether each guard layer is actually wired, warning where it fails open."""
    print("[setup] --- input-guard readiness ---", flush=True)
    if _venv_can_import(["detoxify", "transformers", "presidio_analyzer"]):
        print("[setup]   Layer 2 (toxicity / injection / PII): READY", flush=True)
    else:
        print("[setup]   Layer 2: models NOT installed -> FAIL-OPEN. "
              "Enable with: python app.py --with-guard", flush=True)

    reachable, has_model = _ollama_status(os.environ.get("GUARD_LLM_URL", "http://localhost:11434"))
    if reachable and has_model:
        print("[setup]   Layer 3 (Ollama / llama-guard3): READY", flush=True)
    elif reachable:
        print("[setup]   WARNING: Ollama is up but llama-guard3 is missing -> Layer 3 "
              "FAIL-OPEN. Run: ollama pull llama-guard3:1b", flush=True)
    else:
        print("[setup]   WARNING: Ollama not reachable -> Layer 3 FAIL-OPEN; unsafe "
              "questions may pass. Install Ollama (https://ollama.com), then "
              "`ollama pull llama-guard3:1b` — or set GUARD_LLM_ENABLED=0 in .env.", flush=True)
    print("[setup] -------------------------------", flush=True)


def _stream_output(proc: subprocess.Popen, prefix: str) -> None:
    """Read a child's combined output line by line and echo it with a prefix."""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        print(f"[{prefix}] {line}", flush=True)


def _popen(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    """Start a child process in its own process group so we can kill the tree."""
    kwargs: dict = dict(
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if IS_WINDOWS:
        # New process group so CTRL_BREAK reaches children (e.g. node, flask).
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # own session/process group
    return subprocess.Popen(cmd, **kwargs)


def start_backend(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PORT"] = str(port)
    cmd = [_backend_python(), "app.py"]
    print(f"[orchestrator] starting backend on :{port} -> {cmd}", flush=True)
    return _popen(cmd, BACKEND_DIR, env)


def start_frontend(port: int, backend_port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["VITE_BACKEND_PORT"] = str(backend_port)
    npm = "npm.cmd" if IS_WINDOWS else "npm"
    cmd = [npm, "run", "dev", "--", "--port", str(port)]
    print(f"[orchestrator] starting frontend on :{port} -> {cmd}", flush=True)
    return _popen(cmd, FRONTEND_DIR, env)


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort clean shutdown of a child process tree."""
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def _docker_running() -> bool:
    """True if the Docker engine is reachable (i.e. Docker Desktop is running)."""
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=20
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _compose_base() -> list[str] | None:
    """`docker compose` (v2) if available, else `docker-compose`, else None."""
    if shutil.which("docker") and subprocess.run(
        ["docker", "compose", "version"], capture_output=True
    ).returncode == 0:
        return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None


def run_production(run_frontend: bool, frontend_port: int, backend_port: int) -> int:
    """Default mode: backend + vLLM guard in Docker; frontend stays a local Vite server."""
    if not _docker_running():
        print("[orchestrator] WARNING: the Docker engine isn't reachable — is Docker "
              "Desktop running?", flush=True)
        print("[orchestrator]   Start Docker Desktop and retry, or run the local flow: "
              "python app.py --dev", flush=True)
        return 1
    compose = _compose_base()
    if compose is None:
        print("[orchestrator] Docker is up but `docker compose` was not found.", flush=True)
        return 1

    print("[orchestrator] building & starting containers (backend + guard); the first "
          "build pulls torch + the vLLM image and can take a while...", flush=True)
    up = subprocess.run([*compose, "up", "--build", "-d", "backend", "guard"], cwd=str(ROOT))
    if up.returncode != 0:
        print("[orchestrator] `docker compose up` failed (see output above).", flush=True)
        return up.returncode

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        procs.append(("docker", _popen([*compose, "logs", "-f", "backend", "guard"],
                                        ROOT, os.environ.copy())))
        if run_frontend:
            procs.append(("frontend", start_frontend(frontend_port, backend_port)))

        for name, proc in procs:
            threading.Thread(target=_stream_output, args=(proc, name), daemon=True).start()

        print(f"[orchestrator] backend (container): http://127.0.0.1:{backend_port}/api/health",
              flush=True)
        if run_frontend:
            print(f"[orchestrator] frontend (local):   http://127.0.0.1:{frontend_port}",
                  flush=True)
        print("[orchestrator] press Ctrl-C to stop (containers are torn down on exit).",
              flush=True)

        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[orchestrator] {name} exited ({code}); shutting down.", flush=True)
                    return code or 0
            for _, proc in procs:
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        print("\n[orchestrator] Ctrl-C received; stopping...", flush=True)
        return 0
    finally:
        for _, proc in procs:
            _terminate(proc)
        print("[orchestrator] docker compose down...", flush=True)
        subprocess.run([*compose, "down"], cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Chart-Visual-QA stack.")
    parser.add_argument("--dev", action="store_true",
                        help="local dev flow (venv backend, local CLIP + Ollama, Vite); "
                             "no Docker. Default runs the production stack in Docker.")
    parser.add_argument("--backend-only", action="store_true", help="run only the Flask API")
    parser.add_argument("--frontend-only", action="store_true", help="run only the Vite dev server")
    parser.add_argument("--setup-only", action="store_true", help="install deps and exit")
    parser.add_argument("--no-setup", action="store_true", help="skip the dependency check")
    parser.add_argument("--no-guard", action="store_true",
                        help="light setup: skip installing the heavy guard models")
    parser.add_argument("--backend-port", type=int, default=5000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    args = parser.parse_args()

    if args.backend_only and args.frontend_only:
        parser.error("--backend-only and --frontend-only are mutually exclusive")

    run_backend = not args.frontend_only
    run_frontend = not args.backend_only

    # Default = production: backend + vLLM guard in Docker, frontend local Vite.
    # `--dev` keeps the local venv + Ollama flow below.
    if not args.dev:
        return run_production(run_frontend, args.frontend_port, args.backend_port)

    if not args.no_setup:
        try:
            ensure_deps(run_backend, run_frontend)
            if run_backend and not args.no_guard:
                ensure_guard_deps()
                ensure_ollama_model()
        except subprocess.CalledProcessError as exc:
            print(f"[setup] FAILED: {exc}", flush=True)
            return 1
    if run_backend and not args.no_setup:
        report_guard_readiness()
    if args.setup_only:
        print("[setup] done (--setup-only).", flush=True)
        return 0

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        if run_backend:
            procs.append(("backend", start_backend(args.backend_port)))
        if run_frontend:
            procs.append(("frontend", start_frontend(args.frontend_port, args.backend_port)))

        threads = []
        for name, proc in procs:
            t = threading.Thread(target=_stream_output, args=(proc, name), daemon=True)
            t.start()
            threads.append(t)

        if run_frontend:
            print(f"[orchestrator] frontend:  http://127.0.0.1:{args.frontend_port}", flush=True)
        if run_backend:
            print(f"[orchestrator] backend:   http://127.0.0.1:{args.backend_port}/api/health", flush=True)
        print("[orchestrator] press Ctrl-C to stop everything.", flush=True)

        # Wait until any child exits (or Ctrl-C). If one dies, tear down the rest.
        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[orchestrator] {name} exited with code {code}; shutting down.", flush=True)
                    return code or 0
            for _, proc in procs:
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        print("\n[orchestrator] Ctrl-C received; stopping children...", flush=True)
        return 0
    finally:
        for _, proc in procs:
            _terminate(proc)


if __name__ == "__main__":
    raise SystemExit(main())
