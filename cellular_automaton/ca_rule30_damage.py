"""How a single flipped bit spreads under Rule 30 (no model involved).

Supports the claim in docs/draft-5.1.md §5.3 that once a row is wrong, a model
that keeps applying Rule 30 cannot return to the true row.

Argument: Rule 30 is left-permutive, new_{i+1} = s_i XOR (s_{i+1} OR s_{i+2}).
On an infinite line, the rightmost cell where two rows differ always changes
its right neighbour at the next step, so two different rows never become
equal again. On the periodic ring used here there is no rightmost cell, so
this script checks it empirically.

Measurement: NUM_ROWS random rows of NUM_CELLS cells (seed 0), one random bit
flipped in each copy, both copies evolved with ca_gen.rule30 (periodic
boundary, the same function that generates the training data). At selected
steps it prints the share of pairs that are identical again ("merged") and the
mean number of differing bits (chance is NUM_CELLS / 2).

Usage:
  python -m cellular_automaton.ca_rule30_damage
"""
import argparse

import torch

from cellular_automaton.ca_gen import rule30

REPORT_STEPS = (1, 5, 10, 20, 30, 40, 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-rows", type=int, default=4096)
    parser.add_argument("--num-cells", type=int, default=64)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    true_row = torch.randint(0, 2, (args.num_rows, args.num_cells), generator=generator)
    flipped = true_row.clone()
    position = torch.randint(0, args.num_cells, (args.num_rows,), generator=generator)
    flipped[torch.arange(args.num_rows), position] ^= 1

    print(f"{args.num_rows} rows x {args.num_cells} cells, one bit flipped per row, periodic Rule 30")
    for step in range(1, args.steps + 1):
        true_row, flipped = rule30(true_row), rule30(flipped)
        if step in REPORT_STEPS or step == args.steps:
            differing = (true_row != flipped).sum(-1).float()
            print(f"step {step:3d}: merged {100 * (differing == 0).float().mean():.2f}%  "
                  f"mean differing bits {differing.mean():.1f} of {args.num_cells}")


if __name__ == "__main__":
    main()
