#!/usr/bin/env python
"""Report how many triplets survive the fully-visible lander filter."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piwm_diffusion.data import LunarTripletDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--min_pixels", type=int, default=30)
    args = parser.parse_args()

    unfiltered = LunarTripletDataset(
        args.data_dir, max_files=args.max_files, require_visible=False
    )
    filtered = LunarTripletDataset(
        args.data_dir, max_files=args.max_files,
        require_visible=True, visible_min_pixels=args.min_pixels
    )

    total = len(unfiltered)
    kept = len(filtered)
    print(f"Total triplets:   {total}")
    print(f"Visible triplets: {kept}  ({100*kept/total:.1f}% kept)")
    print(f"Dropped:          {total - kept}  ({100*(total-kept)/total:.1f}%)")


if __name__ == "__main__":
    main()
