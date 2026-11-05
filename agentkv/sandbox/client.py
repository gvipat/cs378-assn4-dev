"""Evaluator-side handles to a policy: a sandboxed worker process (always used
for scoring) or an in-process runtime (unit tests and debugging only)."""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .loader import instantiate, load_policy_class
from .runtime import PolicyError, PolicyRuntime

PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
MAX_REPLY_BYTES = 4 << 20


class InProcessClient:
    """Runs the policy in this process with no isolation. Same results as the
    worker for a well-behaved policy; never used for grading."""

    def __init__(self, policy, *, params=None, seed=0, block_size, total_blocks):
        if isinstance(policy, (str, Path)):
            policy = instantiate(
                load_policy_class(Path(policy).read_text(), str(policy)), params
            )
        elif isinstance(policy, type):
            policy = instantiate(policy, params)
        self.rt = PolicyRuntime(
            policy, block_size=block_size, total_blocks=total_blocks, seed=seed
        )
        self.hooks = self.rt.hooks
        self.pending: list = []

    def event(self, ev: tuple) -> None:
        self.pending.append(ev)

    def _flush(self) -> None:
        ev, self.pending = self.pending, []
        self.rt.apply(ev)

    def target_concurrency(self, hdr: list):
        self._flush()
        return self.rt.target_concurrency(hdr)

    def choose_victim(self, hdr: list, candidates=None):
        self._flush()
        return self.rt.choose_victim(hdr, candidates)

    def route_call(self, hdr: list, call: str):
        self._flush()
        return self.rt.route_call(hdr, call)

    def batch_token_budget(self, hdr: list):
        self._flush()
        return self.rt.batch_token_budget(hdr)

    def dispatch_order(self, hdr: list, ready: list):
        self._flush()
        return self.rt.dispatch_order(hdr, ready)

    def admission_order(self, hdr: list, waiting: list):
        self._flush()
        return self.rt.admission_order(hdr, waiting)

    def choose_preempted(self, hdr: list, running: list):
        self._flush()
        return self.rt.choose_preempted(hdr, running)

    def finish(self) -> dict:
        self._flush()
        self.rt.check_state()
        return self.rt.stats()

    @property
    def cpu(self) -> float:
        return self.rt.cpu

    def close(self) -> None:
        pass


class WorkerClient:
    """A policy in its own sandboxed process."""

    def __init__(
        self,
        policy_path: str | Path,
        *,
        params=None,
        seed=0,
        block_size,
        total_blocks,
        sandbox: bool = True,
        stderr=None,
        start_timeout: float = 60.0,
    ):
        self.tmp = tempfile.mkdtemp(prefix="agentkv-worker-")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": PACKAGE_PARENT,
            "HOME": self.tmp,
            # A preloaded numpy would otherwise size OpenBLAS's buffers for every
            # core, which overruns the address-space limit on many-core machines.
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        if os.environ.get("AGENTKV_SANDBOX_USER"):
            env["AGENTKV_SANDBOX_USER"] = os.environ["AGENTKV_SANDBOX_USER"]
        if os.environ.get("AGENTKV_SANDBOX_SECCOMP"):
            env["AGENTKV_SANDBOX_SECCOMP"] = os.environ["AGENTKV_SANDBOX_SECCOMP"]
        self.proc = subprocess.Popen(
            [sys.executable, "-s", "-m", "agentkv.sandbox.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            cwd=self.tmp,
            env=env,
        )
        self.fd = self.proc.stdout.fileno()
        self.buf = b""
        self.pending: list = []
        self.cpu = 0.0
        self._send(
            {
                "op": "init",
                "policy_path": str(Path(policy_path).resolve()),
                "params": params,
                "seed": seed,
                "block_size": block_size,
                "total_blocks": total_blocks,
                "sandbox": sandbox,
            }
        )
        self._recv(start_timeout)
        if env.get("AGENTKV_SANDBOX_SECCOMP") == "1" and not self.last.get(
            "restricted"
        ):
            self.close()
            raise RuntimeError(
                "The policy worker did not activate its Linux restrictions."
            )
        self.hooks = tuple(
            self.last.get("hooks") or ()
        )  # optional hooks the policy defines

    def event(self, ev: tuple) -> None:
        self.pending.append(ev)

    def target_concurrency(self, hdr: list):
        return self._call("tc", hdr)

    def choose_victim(self, hdr: list, candidates=None):
        return self._call("cv", hdr, candidates)

    def route_call(self, hdr: list, call: str):
        return self._call("rc", hdr, call)

    def batch_token_budget(self, hdr: list):
        return self._call("bt", hdr)

    def dispatch_order(self, hdr: list, ready: list):
        return self._call("do", hdr, ready)

    def admission_order(self, hdr: list, waiting: list):
        return self._call("ao", hdr, waiting)

    def choose_preempted(self, hdr: list, running: list):
        return self._call("cp", hdr, running)

    def finish(self) -> dict:
        return self._call("end", None)

    def _call(self, op: str, hdr, args=None):
        ev, self.pending = self.pending, []
        self._send({"op": op, "ev": ev, "h": hdr, "a": args})
        # 50 ms of CPU per hook call, with generous slack for scheduling noise.
        return self._recv(2.0 + 0.1 * (len(ev) + 1))

    def _send(self, obj) -> None:
        try:
            self.proc.stdin.write(
                (json.dumps(obj, separators=(",", ":")) + "\n").encode()
            )
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise PolicyError("crash", "policy worker exited") from None

    def _recv(self, timeout: float):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready or remaining == 0:
                self.close()
                raise PolicyError(
                    "timeout", f"no answer from the policy within {timeout:.1f} s"
                )
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                code = self.proc.wait()
                raise PolicyError("crash", f"policy worker exited with status {code}")
            self.buf += chunk
            if len(self.buf) > MAX_REPLY_BYTES:
                self.close()
                raise PolicyError("crash", "policy worker reply exceeded 4 MiB")
        line, _, self.buf = self.buf.partition(b"\n")
        try:
            msg = json.loads(line)
        except ValueError:
            raise PolicyError(
                "crash", "malformed reply from the policy worker"
            ) from None
        if not isinstance(msg, dict):
            raise PolicyError("crash", "malformed reply from the policy worker")
        if "err" in msg:
            self.close()
            raise PolicyError(str(msg["err"]), str(msg.get("msg", "")))
        if isinstance(msg.get("cpu"), (int, float)):
            self.cpu = float(msg["cpu"])
        self.last = msg
        return msg.get("ok")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)
