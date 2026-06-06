"""Shared helpers for the SLO verify broadcast protocol."""

from __future__ import annotations


def _gamma_for(seq, gamma_map: dict[int, int]) -> int:
    return max(1, int(gamma_map.get(seq.seq_id, 1)))


def count_verify_tokens(seqs, gamma_map: dict[int, int]) -> int:
    return sum(1 if seq.pre_verify else _gamma_for(seq, gamma_map) for seq in seqs)


def count_next_round_tokens(seqs, gamma_map: dict[int, int]) -> int:
    return sum(_gamma_for(seq, gamma_map) for seq in seqs)


def build_verify_payload(seqs, gamma_map: dict[int, int]) -> tuple[list[int], list[int]]:
    to_be_verified_tokens: list[int] = []
    next_round_input: list[int] = []

    for seq in seqs:
        g = _gamma_for(seq, gamma_map)
        if seq.pre_verify:
            to_be_verified_tokens.append(seq.token_ids[-g])
        else:
            start = len(seq.token_ids) - 2 * g + 1
            end = len(seq.token_ids) - g + 1
            to_be_verified_tokens.extend(seq.token_ids[start:end])
        next_round_input.extend(seq.token_ids[-g:])

    return to_be_verified_tokens, next_round_input
