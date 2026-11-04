"""In-process guard for the policy worker: an audit hook plus wall-clock and
resource restrictions. It rejects restricted imports and operations such as
file access, sockets, process creation, and wall-clock reads.

It is one layer, not the security boundary. For hidden evaluation the worker
also runs as an unprivileged user (AGENTKV_SANDBOX_USER) on a host where the
hidden traces are not readable by that user and there is no network.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import time

#: Imports a policy may not make (anything else in the stdlib is allowed).
DENY_IMPORTS = {
    "ctypes",
    "_ctypes",
    "cffi",
    "multiprocessing",
    "_multiprocessing",
    "subprocess",
    "_posixsubprocess",
    "socket",
    "_socket",
    "ssl",
    "mmap",
    "pty",
    "datetime",
    "_datetime",
    "sqlite3",
    "_sqlite3",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "webbrowser",
    "faulthandler",
    "agentkv.score",
    "agentkv.replay",
    "agentkv.engine",
    "agentkv.trace",
    "agentkv.tokenizer",
    "agentkv.baselines",
    "research_app",
    "tracegen",
    "grader",  # starter: allow-ref
}

DENY_EVENTS = (
    "socket.",
    "subprocess.",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.mkdir",
    "os.chdir",
    "os.chmod",
    "os.chown",
    "os.chflags",
    "os.link",
    "os.symlink",
    "os.truncate",
    "os.utime",
    "os.putenv",
    "os.unsetenv",
    "os.startfile",
    "os.setxattr",
    "os.removexattr",
    "shutil.",
    "ctypes.",
    "mmap.",
    "sqlite3.",
    "urllib.",
    "http.",
    "ftplib.",
    "smtplib.",
    "webbrowser.",
    "pty.",
    "fcntl.",
    "resource.",
    "signal.",
    "_thread.start_new_thread",
    "_thread.start_joinable_thread",  # the latter is Python 3.13's name
    "sys.addaudithook",
    "sys.setprofile",
    "sys.settrace",
    "sys.remote_exec",
    "time.sleep",
    "glob.",
    "tempfile.",
    "os.listxattr",
    "os.getxattr",
)

CODE_SUFFIXES = (".py", ".pyc", ".so", ".pyd", ".dylib")

violations: list[str] = []


class SandboxViolation(PermissionError):
    pass


def _roots() -> tuple[str, ...]:
    paths = {
        sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")
    }
    paths.add(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )  # the agentkv package
    return tuple(os.path.realpath(p).rstrip(os.sep) + os.sep for p in paths)


def install(*, wall_clock: bool = True) -> None:
    roots = _roots()

    def deny(what: str) -> None:
        violations.append(what)
        raise SandboxViolation(f"sandbox: {what} is not allowed in a policy")

    def under_roots(path) -> bool:
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, str):
            return False
        real = os.path.realpath(path)
        return any((real + os.sep).startswith(r) or real.startswith(r) for r in roots)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode = args[0], args[1]
            readonly = mode is None or (
                isinstance(mode, str) and "r" in mode and not set(mode) & set("wax+")
            )
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            if flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            ):
                readonly = False
            name = os.fsdecode(path) if isinstance(path, bytes) else path
            if not (
                readonly
                and isinstance(name, str)
                and name.endswith(CODE_SUFFIXES)
                and under_roots(name)
            ):
                deny(f"open({path!r})")
        elif event in ("os.listdir", "os.scandir"):
            if not args or not under_roots(args[0]):
                deny(f"{event}({args[0] if args else ''!r})")
        elif event == "import":
            name = args[0] or ""
            top = name.split(".")[0]
            if (
                name in DENY_IMPORTS
                or top in DENY_IMPORTS
                or any(name.startswith(d + ".") for d in DENY_IMPORTS)
            ):
                deny(f"import {name}")
        elif event.startswith(DENY_EVENTS):
            deny(event)

    sys.path[:] = [p for p in sys.path if p and under_roots(p)]
    # Preloaded libraries (numpy) can pull in denied modules such as ctypes. Drop
    # them from the module cache so a policy's own import of them goes through
    # the import machinery, and so through the hook; the libraries keep their
    # references.
    for name in list(sys.modules):
        if name in DENY_IMPORTS or name.split(".")[0] in DENY_IMPORTS:
            del sys.modules[name]
    sys.addaudithook(hook)
    if wall_clock:

        def no_clock(*_a, **_k):
            violations.append("wall clock")
            raise SandboxViolation(
                "sandbox: wall-clock time is not available to a policy; use event.t / view.now"
            )

        for name in (
            "time",
            "time_ns",
            "monotonic",
            "monotonic_ns",
            "perf_counter",
            "perf_counter_ns",
            "process_time",
            "process_time_ns",
            "thread_time",
            "thread_time_ns",
            "sleep",
            "localtime",
            "gmtime",
            "ctime",
            "asctime",
            "strftime",
        ):
            if hasattr(time, name):
                setattr(time, name, no_clock)


def limit_resources(*, memory_bytes: int = 2 << 30, cpu_seconds: int = 1800) -> None:
    try:
        import resource
    except ImportError:  # not on this platform
        return
    for lim, val in (
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_AS, memory_bytes),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_CORE, 0),
    ):
        try:
            soft, hard = resource.getrlimit(lim)
            if hard != resource.RLIM_INFINITY:
                val = min(val, hard)
            resource.setrlimit(lim, (val, hard))
        except (ValueError, OSError):
            pass  # e.g. RLIMIT_AS on macOS


def drop_privileges(user: str) -> None:
    """Switch to an unprivileged user (only possible when started as root)."""
    import pwd

    if os.geteuid() != 0:
        return
    pw = pwd.getpwnam(user)
    os.setgroups([])
    os.setgid(pw.pw_gid)
    os.setuid(pw.pw_uid)
    if os.geteuid() == 0:
        raise RuntimeError("failed to drop privileges")
