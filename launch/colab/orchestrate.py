"""Colab session orchestrator for jobs that must survive VM death.

Not part of the s4dpc research pipeline (CLAUDE.md's "sweep.py is the only
entry point" rule governs experiment code; this drives the Colab session
that experiment code happens to run on).

Built after THREE straight lost nu-gap-export runs. Root causes, verified
directly against the installed colab-cli's own source
(~/.local/share/uv/tools/google-colab-cli/lib/python3.13/site-packages/colab_cli/contents.py):

  1. `colab upload`/`colab download` are SINGLE-FILE ONLY.
     `ContentsClient.download` raises `IsADirectoryError` on a directory
     remote_path; `upload` requires `os.path.isfile(local_path)`. Every
     backup call in the last three attempts targeted a DIRECTORY
     (`docs/nu_gap_export`, its `ckpt/` subdir) and had its output piped to
     `/dev/null`, so it was failing on *every single cycle* regardless of
     session health - independent of whatever eventually killed the VM.
  2. A live local keep-alive daemon process is NOT evidence the remote VM
     is alive. Confirmed directly: nugap4's daemon `pid` still passed
     `ps -p <pid>` after the remote session had already 404'd. The only
     honest liveness signal is a real `colab exec` round trip.

This script fixes both: real per-file transfers verified by exit code AND
non-zero local size (never `> /dev/null`, every call logged), and a health
check that means "the remote kernel just executed code and returned a
token", not "a local pid exists". It also restores the local backup to a
freshly-relaunched VM BEFORE re-running the target script, so the script's
own per-case `case_done()` / checkpoint-reload logic (tools/nu_gap_export.py)
actually sees prior progress and resumes instead of restarting.

CAVEAT (matters for nu_gap_export.py specifically): the stall-based
death-detector below (N consecutive cycles with no new verified artifact =>
treat as dead) is only safe for phases that checkpoint frequently - M1,
M0_S4, truncated-M3 and full-M3 each write one .npz per (case, seed) every
few minutes. Raw M3 identification (run_identify, ~40k epochs) currently
writes NOTHING until it fully completes (no intermediate checkpointing
exists yet - see the identify.py proposal under discussion), so a real,
alive, slow identification run WILL look indistinguishable from a stall and
get spuriously killed and relaunched. Do not point this at a full
nu_gap_export.py run until that gap is closed; it is safe today for
partial/M1-only smoke tests and will be safe for the full run once
identification checkpoints periodically.

Usage:
    python launch/colab/orchestrate.py --session nugap5 \
        --script tools/nu_gap_export.py [--max-cycles N] [--gpu T4]

Status (from another shell, anytime):
    python launch/colab/status.py nugap5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ORCH_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_REPO_URL = "https://github.com/binoygeorge97/s4dpc.git"
DEFAULT_REMOTE_REPO = "content/s4dpc"


def _log_path(session: str) -> pathlib.Path:
    return ORCH_DIR / f"{session}.log"


def _status_path(session: str) -> pathlib.Path:
    return ORCH_DIR / f"{session}_status.json"


_session_for_log: str | None = None


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _session_for_log:
        with open(_log_path(_session_for_log), "a") as f:
            f.write(line + "\n")


def colab_call(args: list[str], input_str: str | None = None, timeout: float = 60) -> subprocess.CompletedProcess:
    """Thin wrapper around `colab <args>`. ALWAYS captures and logs
    stdout/stderr and the return code - the exact thing the previous
    `> /dev/null 2>&1` watch loop failed to do, which is why three straight
    directory-download calls failed silently for hours."""
    cmd = ["colab"] + args
    try:
        res = subprocess.run(cmd, input=input_str, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        log(f"CALL TIMEOUT cmd={cmd} timeout={timeout}s stdout={e.stdout!r} stderr={e.stderr!r}")
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout=e.stdout or "", stderr=str(e))
    return res


def remote_exec(session: str, code: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return colab_call(["exec", "-s", session, "--timeout", str(timeout)], input_str=code, timeout=timeout + 30)


def health_check(session: str) -> bool:
    token = f"HEALTH_OK_{int(time.time())}"
    res = remote_exec(session, f"print('{token}')", timeout=30)
    ok = res.returncode == 0 and token in res.stdout
    log(f"HEALTH {'OK' if ok else 'FAIL'} rc={res.returncode} stdout={res.stdout.strip()!r} stderr={res.stderr.strip()!r}")
    return ok


def ls_remote(session: str, path: str) -> list[tuple[str, bool]] | None:
    """Immediate children of `path` as [(name, is_dir), ...], or None if the
    path/session doesn't resolve. Every call logged (count only, to avoid
    spamming the log on directories polled every cycle)."""
    res = colab_call(["ls", "-s", session, path], timeout=30)
    if res.returncode != 0:
        log(f"LS FAIL path={path} rc={res.returncode} stdout={res.stdout.strip()!r} stderr={res.stderr.strip()!r}")
        return None
    items = []
    for line in res.stdout.splitlines():
        line = line.rstrip()
        if not line or line.startswith("[colab]"):
            continue
        is_dir = line.endswith("/")
        items.append((line[:-1] if is_dir else line, is_dir))
    log(f"LS OK path={path} -> {len(items)} entries")
    return items


def walk_remote_files(session: str, root: str) -> list[str] | None:
    """Recursively lists all FILE paths under `root` (full remote paths), or
    None if `root` itself doesn't exist / isn't reachable."""
    top = ls_remote(session, root)
    if top is None:
        return None
    files: list[str] = []
    for name, is_dir in top:
        child = f"{root}/{name}"
        if is_dir:
            sub = walk_remote_files(session, child)
            if sub:
                files.extend(sub)
        else:
            files.append(child)
    return files


def verified_download_file(session: str, remote_path: str, local_path: pathlib.Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    res = colab_call(["download", remote_path, str(local_path), "-s", session], timeout=60)
    size = local_path.stat().st_size if local_path.exists() else -1
    ok = res.returncode == 0 and size > 0
    log(f"DOWNLOAD {'OK' if ok else 'FAIL'} remote={remote_path} local={local_path} rc={res.returncode} "
        f"size={size} stdout={res.stdout.strip()!r} stderr={res.stderr.strip()!r}")
    return ok


def verified_upload_file(session: str, local_path: pathlib.Path, remote_path: str) -> bool:
    res = colab_call(["upload", str(local_path), remote_path, "-s", session], timeout=60)
    ok = res.returncode == 0
    log(f"UPLOAD {'OK' if ok else 'FAIL'} local={local_path} remote={remote_path} rc={res.returncode} "
        f"stdout={res.stdout.strip()!r} stderr={res.stderr.strip()!r}")
    return ok


def sync_backup_from_remote(session: str, remote_root: str, local_root: pathlib.Path) -> list[str]:
    """Downloads any remote file not yet present (verified, non-empty)
    locally - artifacts here are write-once (np.savez'd once per case), so
    an existing non-empty local copy is never re-fetched. Returns the
    relative paths newly verified this call."""
    remote_files = walk_remote_files(session, remote_root)
    if remote_files is None:
        return []
    new_files = []
    for rf in remote_files:
        rel = rf[len(remote_root):].lstrip("/")
        local_path = local_root / rel
        if local_path.exists() and local_path.stat().st_size > 0:
            continue
        if verified_download_file(session, rf, local_path):
            new_files.append(rel)
    return new_files


def check_and_fetch_summary(session: str, remote_repo: str, local_repo: pathlib.Path) -> bool:
    entries = ls_remote(session, f"{remote_repo}/docs")
    if entries is None:
        return False
    if "nu_gap_export_summary.csv" not in {n for n, _ in entries}:
        return False
    return verified_download_file(
        session, f"{remote_repo}/docs/nu_gap_export_summary.csv", local_repo / "docs" / "nu_gap_export_summary.csv"
    )


def restore_backup_to_remote(session: str, local_root: pathlib.Path, remote_root: str) -> int:
    if not local_root.exists():
        log(f"RESTORE nothing to restore, {local_root} does not exist locally yet")
        return 0
    local_files = [p for p in local_root.rglob("*") if p.is_file()]
    if not local_files:
        log("RESTORE local backup is empty, nothing to restore")
        return 0
    remote_dirs = sorted({
        str(pathlib.PurePosixPath(remote_root) / p.relative_to(local_root).parent) for p in local_files
    })
    mkdir_code = "import pathlib\n" + "\n".join(f"pathlib.Path('/{d}').mkdir(parents=True, exist_ok=True)" for d in remote_dirs)
    res = remote_exec(session, mkdir_code, timeout=60)
    if res.returncode != 0:
        log(f"RESTORE FAILED to create remote directories rc={res.returncode} stderr={res.stderr.strip()!r}")
        return 0
    uploaded = 0
    for p in local_files:
        remote_path = str(pathlib.PurePosixPath(remote_root) / p.relative_to(local_root))
        if verified_upload_file(session, p, remote_path):
            uploaded += 1
    log(f"RESTORE uploaded {uploaded}/{len(local_files)} files from {local_root} to {remote_root}")
    return uploaded


def bootstrap_remote(session: str, repo_url: str, remote_repo: str) -> bool:
    code = (
        "import subprocess, pathlib\n"
        f"repo = pathlib.Path('/{remote_repo}')\n"
        "if not repo.exists():\n"
        f"    r = subprocess.run(['git', 'clone', '-q', {repo_url!r}, str(repo)])\n"
        "    assert r.returncode == 0, 'git clone failed'\n"
        f"r2 = subprocess.run(['pip', 'install', '-q', '-r', str(repo / 'requirements.lock')])\n"
        "assert r2.returncode == 0, 'pip install failed'\n"
        "print('BOOTSTRAP_OK')\n"
    )
    res = remote_exec(session, code, timeout=300)
    ok = res.returncode == 0 and "BOOTSTRAP_OK" in res.stdout
    log(f"BOOTSTRAP {'OK' if ok else 'FAIL'} rc={res.returncode} stdout={res.stdout[-1500:]!r} stderr={res.stderr[-1500:]!r}")
    return ok


def launch_job(session: str, remote_repo: str, script: str, log_name: str = "job.log") -> bool:
    # `python -u` / PYTHONUNBUFFERED: without it, stdout redirected to a
    # file is fully (not line-) buffered, so the log can sit empty for a
    # long time even while the job is actively working - confirmed
    # directly (job.log stayed at 0 bytes for 2+ minutes of real,
    # non-stalled M1 training). That's a real blind spot for anyone
    # inspecting the log for signs of life, so always launch unbuffered.
    code = (
        "import subprocess, os\n"
        f"os.chdir('/{remote_repo}')\n"
        "env = dict(os.environ, PYTHONUNBUFFERED='1')\n"
        f"p = subprocess.Popen(['python', '-u', {script!r}], stdout=open('/content/{log_name}', 'a'), "
        "stderr=subprocess.STDOUT, start_new_session=True, env=env)\n"
        "print('LAUNCHED_PID', p.pid)\n"
    )
    res = remote_exec(session, code, timeout=60)
    ok = res.returncode == 0 and "LAUNCHED_PID" in res.stdout
    log(f"LAUNCH {'OK' if ok else 'FAIL'} rc={res.returncode} stdout={res.stdout.strip()!r} stderr={res.stderr.strip()!r}")
    return ok


def stop_session(session: str) -> None:
    res = colab_call(["stop", "-s", session], timeout=60)
    log(f"STOP rc={res.returncode} stdout={res.stdout.strip()!r} stderr={res.stderr.strip()!r}")


def ensure_new_session(session: str, gpu: str, retries: int = 5, backoff: float = 30.0) -> bool:
    """Google's own Colab assignment backend intermittently 503s
    (observed directly building this: transient `Service Unavailable`
    on a plain `colab new`, nothing wrong locally). A VM-acquisition
    hiccup is exactly the kind of transient failure this whole
    orchestrator exists to survive, so retry with linear backoff instead
    of treating attempt 1 as final."""
    for attempt in range(1, retries + 1):
        res = colab_call(["new", "-s", session, "--gpu", gpu], timeout=180)
        ok = res.returncode == 0
        log(f"NEW_SESSION attempt={attempt}/{retries} {'OK' if ok else 'FAIL'} rc={res.returncode} "
            f"stdout={res.stdout.strip()!r} stderr={res.stderr.strip()[-800:]!r}")
        if ok:
            return True
        if attempt < retries:
            wait = backoff * attempt
            log(f"NEW_SESSION retrying in {wait:.0f}s")
            time.sleep(wait)
    return False


def write_status(session: str, state: str, n_verified: int, verified_files: list[str]) -> None:
    status = {
        "session": session,
        "state": state,
        "n_verified_artifacts": n_verified,
        "recent_verified_files": verified_files[-20:],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _status_path(session).write_text(json.dumps(status, indent=2))


def relaunch_full(session: str, gpu: str, repo_url: str, remote_repo: str, script: str, local_backup: pathlib.Path) -> bool:
    log("RELAUNCH starting: stop -> new -> bootstrap -> restore backup -> launch job")
    stop_session(session)
    if not ensure_new_session(session, gpu):
        log("RELAUNCH ABORTED: could not create new session")
        return False
    if not bootstrap_remote(session, repo_url, remote_repo):
        log("RELAUNCH ABORTED: bootstrap failed")
        return False
    restore_backup_to_remote(session, local_backup, f"{remote_repo}/docs/nu_gap_export")
    if not launch_job(session, remote_repo, script):
        log("RELAUNCH ABORTED: job launch failed")
        return False
    log("RELAUNCH complete, job running")
    return True


def relaunch_full_with_retry(
    session: str, gpu: str, repo_url: str, remote_repo: str, script: str, local_backup: pathlib.Path,
    retries: int = 3, backoff: float = 60.0,
) -> bool:
    """`relaunch_full` itself can fail transiently past ensure_new_session's
    own retries (a bootstrap network hiccup, a launch exec timeout) - retry
    the WHOLE sequence a few times before giving up, so the orchestrator's
    own failure modes don't undermine the "any death costs minutes, not the
    run" goal."""
    for attempt in range(1, retries + 1):
        if relaunch_full(session, gpu, repo_url, remote_repo, script, local_backup):
            return True
        if attempt < retries:
            log(f"relaunch_full attempt {attempt}/{retries} failed, retrying in {backoff:.0f}s")
            time.sleep(backoff)
    return False


def main() -> None:
    global _session_for_log
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--script", default="tools/nu_gap_export.py")
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    ap.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--interval", type=float, default=120.0, help="seconds between health/sync cycles")
    ap.add_argument("--stall-limit", type=int, default=3, help="consecutive no-new-artifact cycles before relaunch")
    ap.add_argument("--local-backup", default=str(REPO_ROOT / "docs" / "nu_gap_export"))
    ap.add_argument("--max-cycles", type=int, default=None, help="stop the ORCHESTRATOR (not the job) after N cycles")
    ap.add_argument("--skip-initial-launch", action="store_true", help="attach to an already-running session/job")
    args = ap.parse_args()

    _session_for_log = args.session
    local_backup = pathlib.Path(args.local_backup)
    remote_export = f"{args.remote_repo}/docs/nu_gap_export"

    log(f"orchestrator starting: session={args.session} script={args.script} interval={args.interval}s "
        f"stall_limit={args.stall_limit} local_backup={local_backup}")
    write_status(args.session, "starting", 0, [])

    if not args.skip_initial_launch:
        if not relaunch_full_with_retry(args.session, args.gpu, args.repo_url, args.remote_repo, args.script, local_backup):
            write_status(args.session, "FAILED_TO_START", 0, [])
            sys.exit(1)

    verified_total: list[str] = []
    stall = 0
    cycle = 0
    while True:
        cycle += 1
        if args.max_cycles and cycle > args.max_cycles:
            log(f"reached --max-cycles={args.max_cycles}, orchestrator exiting (session left running)")
            write_status(args.session, "orchestrator_stopped_max_cycles", len(verified_total), verified_total)
            break

        time.sleep(args.interval)

        if not health_check(args.session):
            log("HEALTH CHECK FAILED -> treating session as dead")
            write_status(args.session, "session_dead_relaunching", len(verified_total), verified_total)
            if not relaunch_full_with_retry(args.session, args.gpu, args.repo_url, args.remote_repo, args.script, local_backup):
                write_status(args.session, "RELAUNCH_FAILED", len(verified_total), verified_total)
                break
            stall = 0
            continue

        new_files = sync_backup_from_remote(args.session, remote_export, local_backup)
        if new_files:
            verified_total.extend(new_files)
            stall = 0
            log(f"cycle {cycle}: {len(new_files)} new verified artifact(s): {new_files}")
        else:
            stall += 1
            log(f"cycle {cycle}: no new artifacts (stall={stall}/{args.stall_limit})")

        write_status(args.session, "running", len(verified_total), verified_total)

        if check_and_fetch_summary(args.session, args.remote_repo, REPO_ROOT):
            log("summary CSV verified locally -> job complete")
            write_status(args.session, "complete", len(verified_total), verified_total)
            break

        if stall >= args.stall_limit:
            log(f"STALL LIMIT REACHED ({stall} cycles, no new artifact) -> treating session as dead/stuck")
            write_status(args.session, "stalled_relaunching", len(verified_total), verified_total)
            if not relaunch_full_with_retry(args.session, args.gpu, args.repo_url, args.remote_repo, args.script, local_backup):
                write_status(args.session, "RELAUNCH_FAILED", len(verified_total), verified_total)
                break
            stall = 0


if __name__ == "__main__":
    main()
