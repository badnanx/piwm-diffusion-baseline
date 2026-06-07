#!/usr/bin/env python
import argparse
import os
import subprocess
import sys


def run(cmd, cwd):
    print()
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/smoke")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = args.output_dir

    run(
        [
            sys.executable,
            "scripts/train_autoencoder.py",
            "--train_dir",
            args.train_dir,
            "--test_dir",
            args.test_dir,
            "--output_dir",
            os.path.join(out, "ae"),
            "--epochs",
            "1",
            "--batch_size",
            "8",
            "--max_train_files",
            "2",
            "--max_test_files",
            "1",
            "--max_frames_per_file",
            "8",
            "--latent_dim",
            "16",
            "--state_indices",
            "0",
            "1",
            "4",
        ],
        cwd=root,
    )

    run(
        [
            sys.executable,
            "scripts/export_latents.py",
            "--checkpoint",
            os.path.join(out, "ae", "best.pt"),
            "--data_dir",
            args.train_dir,
            "--output_npz",
            os.path.join(out, "latents_train.npz"),
            "--max_files",
            "2",
            "--max_frames_per_file",
            "8",
        ],
        cwd=root,
    )

    run(
        [
            sys.executable,
            "scripts/export_latents.py",
            "--checkpoint",
            os.path.join(out, "ae", "best.pt"),
            "--data_dir",
            args.test_dir,
            "--output_npz",
            os.path.join(out, "latents_test.npz"),
            "--max_files",
            "1",
            "--max_frames_per_file",
            "8",
        ],
        cwd=root,
    )

    run(
        [
            sys.executable,
            "scripts/train_conditional_latent_ddpm.py",
            "--train_npz",
            os.path.join(out, "latents_train.npz"),
            "--val_npz",
            os.path.join(out, "latents_test.npz"),
            "--output_dir",
            os.path.join(out, "ddpm"),
            "--epochs",
            "1",
            "--batch_size",
            "8",
            "--diffusion_steps",
            "4",
            "--hidden_dim",
            "64",
            "--num_layers",
            "2",
        ],
        cwd=root,
    )

    run(
        [
            sys.executable,
            "scripts/sample_conditional.py",
            "--autoencoder_checkpoint",
            os.path.join(out, "ae", "best.pt"),
            "--diffusion_checkpoint",
            os.path.join(out, "ddpm", "best.pt"),
            "--conditions_npz",
            os.path.join(out, "latents_test.npz"),
            "--output_dir",
            os.path.join(out, "samples"),
            "--num_samples",
            "4",
            "--clamp_physical_dims",
        ],
        cwd=root,
    )

    print()
    print("Smoke test completed.")


if __name__ == "__main__":
    main()
