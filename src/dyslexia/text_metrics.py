"""Small, dependency-free string metrics shared by the feature extractors."""

from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (iterative, O(len(a) * len(b)) time,
    O(min) space)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def similarity_ratio(reference: str, candidate: str) -> float:
    """1.0 when identical, →0.0 as the edit distance approaches the string
    length. Symmetric-ish; normalised by the reference length (+1 to avoid
    division by zero, mirroring the original project's formula)."""
    return (len(reference) - levenshtein(reference, candidate)) / (len(reference) + 1)
