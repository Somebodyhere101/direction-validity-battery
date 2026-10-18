"""Outcome parsers copied unchanged from paper 4 (cr4/cr4lib.py), so D3 uses the same definitions."""

from __future__ import annotations

import re

import numpy as np


def settle_top1(top1_row: torch.Tensor) -> int:
    """Smallest layer l such that the lens top-1 equals the final top-1 at every layer >= l."""
    L = top1_row.shape[0]
    final = top1_row[-1]
    l = L - 1
    while l > 0 and top1_row[l - 1] == final:
        l -= 1
    return l


def answer_span(tok, toks: list[int]) -> tuple[int | None, list[int]]:
    """First run of digit tokens (commas between digits allowed), read as an integer.
    Returns (value, indices of the digit tokens)."""
    pieces = [tok.decode([t]) for t in toks]
    i, idx = 0, []
    while i < len(pieces) and pieces[i].strip() == "":
        i += 1
    while i < len(pieces):
        p = pieces[i]
        if p.isascii() and p.isdigit():
            idx.append(i)
        elif not (p == "," and idx and i + 1 < len(pieces) and pieces[i + 1].isascii() and pieces[i + 1].isdigit()):
            break
        i += 1
    if not idx:
        return None, []
    return int("".join(pieces[j] for j in idx)), idx


NUM = re.compile(r"\d[\d,]*\d|\d")


def shown_working(text: str, a: int, b: int) -> tuple[int, int | None]:
    nums = [int(n.replace(",", "")) for n in NUM.findall(text)]
    if not nums:
        return 0, None
    final = nums[-1]
    return int(any(n not in (a, b, final) for n in nums)), final


def error_type(ans: int | None, a: int, b: int, correct: int) -> dict:
    if ans is None or ans == correct:
        return dict(OR=0, OP=0, DC=0, other=0)
    OR = any(ans == a2 * b for a2 in range(max(1, a - 2), a + 3) if a2 != a) or \
        any(ans == a * b2 for b2 in range(max(1, b - 2), b + 3) if b2 != b)
    OP = ans == a + b
    sa, sc = str(ans), str(correct)
    DC = len(sa) == len(sc) and sum(x != y for x, y in zip(sa, sc)) == 1
    return dict(OR=int(OR), OP=int(OP), DC=int(DC), other=int(not (OR or OP or DC)))
