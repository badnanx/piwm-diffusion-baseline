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
    parser.add_argument("--output_dir", default="outputs/smoke_full_piwm")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = args.output_dir
    state = ["0", "1", "2", "3", "4", "5"]

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
            "--state_weight",
            "0.0",
            "--state_indices",
            *state,
        ],
        cwd=root,
    )

    run(
        [
            sys.executable,
            "scripts/train_physical_encoder.py",
            "--autoencoder_checkpoint",
            os.path.join(out, "ae", "best.pt"),
            "--train_dir",
            args.train_dir,
            "--test_dir",
            args.test_dir,
            "--output_dir",
            os.path.join(out, "physical"),
            "--epochs",
            "1",
            "--batch_size",
            "8",
            "--hidden_dim",
            "64",
            "--num_layers",
            "2",
            "--max_train_files",
            "2",
            "--max_test_files",
            "1",
            "--max_frames_per_file",
            "8",
            "--state_indices",
            *state,
        ],
        cwd=root,
    )

    run(
        [
            sys.executable,
            "scripts/train_dynamics.py",
            "--autoencoder_checkpoint",
            os.path.join(out, "ae", "best.pt"),
            "--physical_checkpoint",
            os.path.join(out, "physical", "best.pt"),
            "--train_dir",
            args.train_dir,
            "--test_dir",
            args.test_dir,
            "--output_dir",
            os.path.join(out, "dynamics"),
            "--epochs",
            "1",
            "--batch_size",
            "8",
            "--max_train_files",
            "2",
            "--max_test_files",
            "1",
            "--max_triplets_per_file",
            "6",
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
            "--state_indices",
            *state,
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
            "--state_indices",
            *state,
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
            "scripts/eval_piwm_diffusion_rollout.py",
            "--autoencoder_checkpoint",
            os.path.join(out, "ae", "best.pt"),
            "--physical_checkpoint",
            os.path.join(out, "physical", "best.pt"),
            "--dynamics_checkpoint",
            os.path.join(out, "dynamics", "best.pt"),
            "--diffusion_checkpoint",
            os.path.join(out, "ddpm", "best.pt"),
            "--data_dir",
            args.test_dir,
            "--output_dir",
            os.path.join(out, "rollout"),
            "--batch_size",
            "4",
            "--max_files",
            "1",
            "--max_triplets_per_file",
            "4",
            "--num_viz",
            "4",
        ],
        cwd=root,
    )

    print()
    print("Full PIWM + diffusion smoke test completed.")


if __name__ == "__main__":
    main()
