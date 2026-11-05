"""Load a candidate policy class from its source file."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

from ..policy import Policy
from .runtime import PolicyError

MODULE_NAME = "agentkv_candidate"


def policy_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy_class(source: str, filename: str = "<policy>") -> type[Policy]:
    """Execute the candidate module and return its Policy subclass: the module's
    `POLICY` attribute if set, else the one Policy subclass it defines."""
    mod = types.ModuleType(MODULE_NAME)
    mod.__file__ = filename
    sys.modules[MODULE_NAME] = mod
    try:
        exec(compile(source, filename, "exec"), mod.__dict__)
    except Exception as exc:
        raise PolicyError("load", f"{type(exc).__name__}: {exc}") from exc
    cls = getattr(mod, "POLICY", None)
    if cls is None:
        found = [
            v
            for v in vars(mod).values()
            if isinstance(v, type)
            and issubclass(v, Policy)
            and v is not Policy
            and v.__module__ == MODULE_NAME
        ]
        if len(found) != 1:
            raise PolicyError(
                "load",
                f"expected one Policy subclass (or POLICY = ...), found {len(found)}",
            )
        cls = found[0]
    if not (isinstance(cls, type) and issubclass(cls, Policy)):
        raise PolicyError("load", "POLICY must be a subclass of agentkv.policy.Policy")
    return cls


def instantiate(cls: type[Policy], params: dict | None) -> Policy:
    try:
        return cls(params) if params is not None else cls()
    except Exception as exc:
        raise PolicyError("load", f"constructor: {type(exc).__name__}: {exc}") from exc
