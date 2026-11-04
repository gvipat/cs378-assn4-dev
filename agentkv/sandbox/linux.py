"""Apply kernel restrictions when a course grader runs a policy on Linux."""

from __future__ import annotations

import sys


def restrict() -> None:
    """Block networking, process creation, and inspection of other processes.

    Call this before loading a submitted policy. A setup failure raises an
    exception so the grader cannot silently run without these restrictions.
    """
    if sys.platform != "linux":
        raise RuntimeError("The production policy sandbox requires Linux.")
    import ctypes
    import errno
    import resource

    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7FFF0000)  # Allow other system calls.
    if not ctx:
        raise RuntimeError("Could not initialize the policy syscall filter.")
    try:
        for name in (
            "socket",
            "socketpair",
            "connect",
            "bind",
            "listen",
            "accept",
            "accept4",
            "execve",
            "execveat",
            "fork",
            "vfork",
            "clone",
            "clone3",
            "setsid",
            "setpgid",
            "kill",
            "tkill",
            "tgkill",
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "mount",
            "umount2",
            "unshare",
            "setns",
            "bpf",
            "keyctl",
            "io_uring_setup",
        ):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if (
                number >= 0
                and lib.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0) != 0
            ):
                raise RuntimeError("Could not configure the policy syscall filter.")
        if lib.seccomp_load(ctx) != 0:
            raise RuntimeError("Could not activate the policy syscall filter.")
    finally:
        lib.seccomp_release(ctx)
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 << 20, 4 << 20))
