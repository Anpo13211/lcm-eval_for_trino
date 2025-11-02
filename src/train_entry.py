"""Unified training entry point for Trino-related experiments.

This module provides a thin wrapper around the specialised training scripts
in :mod:`trino_lcm.scripts` and :mod:`train_trino`.  Users can now launch any
of the supported training pipelines through a single CLI or by calling
``run_training`` programmatically, without needing to remember the location of
the individual scripts.
"""

from __future__ import annotations

import argparse
from types import ModuleType
from typing import Dict, Iterable, Optional, Sequence

from trino_lcm.scripts import train_flat_vector, train_zeroshot

# ``train_trino`` lives at the repository root ``src/`` directory.
import train_trino


def _available_modules() -> Dict[str, ModuleType]:
    """Return the mapping from trainer names to their implementation modules."""

    return {
        "flat-vector": train_flat_vector,
        "zero-shot": train_zeroshot,
        "query-former": train_trino,
    }


def list_trainers() -> Iterable[str]:
    """Return the supported trainer names sorted alphabetically."""

    return sorted(_available_modules().keys())


def run_training(trainer: str, trainer_args: Optional[Sequence[str]] = None) -> int:
    """Execute a specific training pipeline.

    Args:
        trainer: Name of the training pipeline. Must be one of
            :func:`list_trainers`.
        trainer_args: Optional list of CLI-like arguments for the trainer.

    Returns:
        The exit code of the underlying training routine (``0`` on success).
    """

    modules = _available_modules()
    if trainer not in modules:
        raise ValueError(f"Unknown trainer '{trainer}'. Available: {', '.join(list_trainers())}")

    module = modules[trainer]

    parser = module.build_parser()
    parsed_args = parser.parse_args(list(trainer_args or ()))

    run_fn = getattr(module, "run", None)
    if run_fn is None:
        # Fallback to the module's ``main`` function if ``run`` is not exposed.
        result = module.main(list(trainer_args or ()))
    else:
        result = run_fn(parsed_args)

    return int(result) if result is not None else 0


def _build_entry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified training CLI")
    parser.add_argument("trainer", choices=list_trainers(), help="Training pipeline to execute")
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the selected trainer",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    entry_parser = _build_entry_parser()
    parsed = entry_parser.parse_args(argv)

    # Allow users to separate trainer arguments with "--" for clarity.
    forwarded_args = list(parsed.trainer_args or [])
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    if forwarded_args and forwarded_args[0] in {"-h", "--help"}:
        trainer_parser = _available_modules()[parsed.trainer].build_parser()
        trainer_parser.prog = f"{entry_parser.prog} {parsed.trainer}"
        trainer_parser.print_help()
        return 0

    return run_training(parsed.trainer, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())

