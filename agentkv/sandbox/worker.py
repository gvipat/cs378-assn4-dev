"""The policy worker process. Speaks line-delimited JSON on stdin/stdout:

    -> {"op": "init", "policy_path", "params", "seed", "block_size", "total_blocks", "sandbox"}
    <- {"ok": "ready"} | {"err": category, "msg": detail}
    -> {"op": "tc" | "cv" | "ev" | "end", "ev": [[kind, t, ...fields], ...], "h": header}
    <- {"ok": value, "cpu": seconds} | {"err": category, "msg": detail}

The evaluator never unpickles anything from here and validates every answer.
"""

from __future__ import annotations

import json
import os
import sys

# Preload common modules so their initialization is outside the policy hook CPU budget.
PRELOAD = (
    "math",
    "heapq",
    "bisect",
    "collections",
    "itertools",
    "functools",
    "statistics",
    "random",
    "dataclasses",
    "typing",
    "enum",
    "operator",
    "copy",
    "numbers",
    "array",
    "fractions",
    "decimal",
    "re",
    "string",
    "abc",
    "inspect",
    "pickle",
)


def _preload() -> None:
    import importlib

    for name in PRELOAD:
        importlib.import_module(name)
    try:
        importlib.import_module("numpy")
    except Exception:
        pass


class _Lines:
    def __init__(self, fd: int):
        self.fd, self.buf = fd, b""

    def readline(self) -> bytes:
        while b"\n" not in self.buf:
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                raise EOFError
            self.buf += chunk
        line, _, self.buf = self.buf.partition(b"\n")
        return line


def main() -> None:
    rfd, wfd = os.dup(0), os.dup(1)
    os.dup2(2, 1)  # print() in a policy goes to stderr, not the protocol
    sys.stdout = sys.stderr
    lines = _Lines(rfd)

    def send(obj) -> None:
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        while data:
            n = os.write(wfd, data)
            data = data[n:]

    from . import guard
    from .loader import instantiate, load_policy_class
    from .runtime import PolicyError, PolicyRuntime

    init = json.loads(lines.readline())
    try:
        with open(init["policy_path"], encoding="utf-8") as f:
            source = f.read()
    except OSError as exc:
        send({"err": "load", "msg": f"cannot read policy: {exc}"})
        return
    user = os.environ.get("AGENTKV_SANDBOX_USER")
    if user:
        guard.drop_privileges(user)
    _preload()
    restricted = False
    if os.environ.get("AGENTKV_SANDBOX_SECCOMP") == "1":
        from .linux import restrict

        if os.geteuid() == 0:
            raise RuntimeError("The policy worker must run as an unprivileged user.")
        restrict()
        restricted = True
    if init.get("sandbox", True):
        guard.limit_resources()
        guard.install()
    try:
        policy = instantiate(
            load_policy_class(source, init["policy_path"]), init.get("params")
        )
        rt = PolicyRuntime(
            policy,
            block_size=init["block_size"],
            total_blocks=init["total_blocks"],
            seed=init["seed"],
        )
    except PolicyError as exc:
        send(
            {"err": "sandbox" if guard.violations else exc.category, "msg": exc.detail}
        )
        return
    send({"ok": "ready", "hooks": list(rt.hooks), "restricted": restricted})

    while True:
        try:
            msg = json.loads(lines.readline())
        except EOFError:
            return
        op = msg["op"]
        try:
            rt.apply(msg.get("ev") or [])
            if op == "tc":
                out = rt.target_concurrency(msg["h"])
                if not isinstance(out, int) or isinstance(out, bool):
                    raise PolicyError(
                        "invalid",
                        f"target_concurrency returned {type(out).__name__}, not int",
                    )
            elif op == "cv":
                out = rt.choose_victim(msg["h"], msg.get("a"))
                if not isinstance(out, str):
                    raise PolicyError(
                        "invalid",
                        f"choose_victim returned {type(out).__name__}, not str",
                    )
            elif op == "do":
                out = rt.dispatch_order(msg["h"], msg["a"])
            elif op == "ao":
                out = rt.admission_order(msg["h"], msg["a"])
            elif op == "cp":
                out = rt.choose_preempted(msg["h"], msg["a"])
            elif op == "rc":
                out = rt.route_call(msg["h"], msg["a"])
            elif op == "bt":
                out = rt.batch_token_budget(msg["h"])
            elif op == "end":
                rt.check_state()
                out = rt.stats()
            else:
                out = None
            if guard.violations:
                raise PolicyError("sandbox", guard.violations[0])
        except PolicyError as exc:
            cat = "sandbox" if guard.violations else exc.category
            send(
                {
                    "err": cat,
                    "msg": guard.violations[0] if guard.violations else exc.detail,
                }
            )
            return
        except (
            Exception
        ) as exc:  # a bug in the runtime itself, or a broken event stream
            send({"err": "internal", "msg": f"{type(exc).__name__}: {exc}"})
            return
        send({"ok": out, "cpu": rt.cpu})
        if op == "end":
            return


if __name__ == "__main__":
    main()
