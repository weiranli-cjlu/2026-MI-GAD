"""Refactored MI-GAD package."""

from .config import Config, parse_args

__all__ = ["Config", "parse_args"]

try:
    from .models import GAD
    from .train import run_experiment

    __all__ += ["GAD", "run_experiment"]
except ModuleNotFoundError:
    pass
