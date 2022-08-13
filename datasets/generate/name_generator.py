from pathlib import Path
from random import choices
from typing import List

from .util import get_git_root


def load_names(filename: str) -> List[str]:
    with open(filename, "r") as f:
        return f.read().split("\n")


def generate_names(n: int = 1):
    repo_dir = Path(get_git_root())
    first_names = load_names(repo_dir / "datasets" / "raw" / "first_names.txt")
    last_names = load_names(repo_dir / "datasets" / "raw" / "animals.txt")
    fnames = choices(first_names, k=n)
    lnames = choices(last_names, k=n)
    return list(f"{x} {y}" for x, y in zip(fnames, lnames))
