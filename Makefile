PYTHON ?= python
DEVICE ?= cuda
RUN_DIR ?= outputs/laptop_fast_v1
TRAIN_DIR ?= ../data/lunar/extracted/lunar/lunartrain
TEST_DIR ?= ../data/lunar/extracted/lunar/lunartest
SEED ?= 42

# Laptop-fast defaults for RTX 3050 Ti, 4 GB VRAM.
EPOCHS ?= 3
TRAIN_FILES ?= 24
TEST_FILES ?= 6
LATENT_DIM ?= 48
AE_BATCH ?= 24
PHYS_BATCH ?= 64
DYN_BATCH ?= 64
DDPM_BATCH ?= 128
DDPM_HIDDEN ?= 128
DDPM_LAYERS ?= 3
DDPM_STEPS ?= 15
STATE_INDICES ?= 0 1 4
STATE_WEIGHT ?= 1.0
CROP_WEIGHT ?= 1.0
CROP_SIZE ?= 24

.PHONY: full-fast full-fast-crop ae ae-crop24 vq vq-crop24 physical dynamics export-latents ddpm samples rollout eval-physical eval-dynamics check-cuda

full-fast: check-cuda ae physical dynamics export-latents ddpm samples rollout
	@echo "Full laptop-fast PIWM + latent diffusion run complete."
	@echo "Inspect: $(RUN_DIR)/ae/recon_best.png"
	@echo "Inspect: $(RUN_DIR)/random_samples/samples.png"
	@echo "Inspect: $(RUN_DIR)/rollout/rollout_real_vs_generated.png"
	@echo "Inspect: $(RUN_DIR)/*/loss_curves.png and summary.json"

full-fast-crop: check-cuda ae-crop24 physical eval-physical dynamics eval-dynamics export-latents ddpm samples rollout
	@echo "Full laptop-fast PIWM + latent diffusion run complete, using crop AE."
	@echo "Inspect: $(RUN_DIR)/ae/recon_best.png"
	@echo "Inspect: $(RUN_DIR)/ae/recon_best_boxed.png"
	@echo "Inspect: $(RUN_DIR)/physical_eval/scatter.png"
	@echo "Inspect: $(RUN_DIR)/dynamics_eval/scatter.png"
	@echo "Inspect: $(RUN_DIR)/random_samples/samples.png"
	@echo "Inspect: $(RUN_DIR)/rollout/rollout_real_vs_generated.png"

full-fast-crop-vq: check-cuda vq-crop24 physical eval-physical dynamics eval-dynamics export-latents ddpm samples rollout
	@echo "Full laptop-fast PIWM + latent diffusion run complete, using VQ-VAE with crop."
	@echo "Inspect: $(RUN_DIR)/ae/recon_best.png"
	@echo "Inspect: $(RUN_DIR)/ae/recon_best_boxed.png"
	@echo "Inspect: $(RUN_DIR)/physical_eval/scatter.png"
	@echo "Inspect: $(RUN_DIR)/dynamics_eval/scatter.png"
	@echo "Inspect: $(RUN_DIR)/random_samples/samples.png"
	@echo "Inspect: $(RUN_DIR)/rollout/rollout_real_vs_generated.png"

check-cuda:
	$(PYTHON) -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available. Activate the right env or set DEVICE=cpu for CPU testing.'"

ae:
	@echo "\n=== STAGE 1/7: AE (continuous VAE, no crop) ===\n"
	$(PYTHON) scripts/train_autoencoder.py \
	  --train_dir $(TRAIN_DIR) \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/ae \
	  --state_indices $(STATE_INDICES) \
	  --state_weight 0.0 \
	  --latent_dim $(LATENT_DIM) \
	  --epochs $(EPOCHS) \
	  --batch_size $(AE_BATCH) \
	  --max_train_files $(TRAIN_FILES) \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED)

ae-crop24:
	@echo "\n=== STAGE 1/9: AE (continuous VAE + crop-24 loss) ===\n"
	$(PYTHON) scripts/train_autoencoder.py \
	  --train_dir $(TRAIN_DIR) \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/ae \
	  --state_indices $(STATE_INDICES) \
	  --state_weight $(STATE_WEIGHT) \
	  --crop_weight $(CROP_WEIGHT) \
	  --crop_size $(CROP_SIZE) \
	  --latent_dim $(LATENT_DIM) \
	  --epochs $(EPOCHS) \
	  --batch_size $(AE_BATCH) \
	  --max_train_files $(TRAIN_FILES) \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED)

vq:
	@echo "\n=== STAGE 1/7: AE (VQ-VAE, no crop) ===\n"
	$(PYTHON) scripts/train_vq_autoencoder.py \
	  --train_dir $(TRAIN_DIR) \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/ae \
	  --state_indices $(STATE_INDICES) \
	  --latent_dim $(LATENT_DIM) \
	  --epochs $(EPOCHS) \
	  --batch_size $(AE_BATCH) \
	  --max_train_files $(TRAIN_FILES) \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED)

vq-crop24:
	@echo "\n=== STAGE 1/9: AE (VQ-VAE + crop-24 loss) ===\n"
	$(PYTHON) scripts/train_vq_autoencoder.py \
	  --train_dir $(TRAIN_DIR) \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/ae \
	  --state_indices $(STATE_INDICES) \
	  --crop_weight $(CROP_WEIGHT) \
	  --crop_size $(CROP_SIZE) \
	  --latent_dim $(LATENT_DIM) \
	  --epochs $(EPOCHS) \
	  --batch_size $(AE_BATCH) \
	  --max_train_files $(TRAIN_FILES) \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED)

physical:
	@echo "\n=== STAGE 2/9: Physical Encoder (z -> f) ===\n"
	$(PYTHON) scripts/train_physical_encoder.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --train_dir $(TRAIN_DIR) \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/physical \
	  --state_indices $(STATE_INDICES) \
	  --epochs $(EPOCHS) \
	  --batch_size $(PHYS_BATCH) \
	  --hidden_dim 128 \
	  --num_layers 2 \
	  --max_train_files $(TRAIN_FILES) \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED)

dynamics:
	@echo "\n=== STAGE 4/9: Dynamics Model (f_t, f_t1, a -> f_t2) ===\n"
	$(PYTHON) scripts/train_dynamics.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --train_dir $(TRAIN_DIR) \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/dynamics \
	  --epochs $(EPOCHS) \
	  --batch_size $(DYN_BATCH) \
	  --max_train_files $(TRAIN_FILES) \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED)

eval-dynamics:
	@echo "\n=== STAGE 5/9: Eval Dynamics (scatter + position overlay) ===\n"
	$(PYTHON) scripts/eval_dynamics.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --dynamics_checkpoint $(RUN_DIR)/dynamics/best.pt \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/dynamics_eval \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED)

eval-physical:
	@echo "\n=== STAGE 3/9: Eval Physical Encoder (scatter + R² per dim) ===\n"
	$(PYTHON) scripts/eval_physical_encoder.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/physical_eval \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED)

export-latents:
	@echo "\n=== STAGE 6/9: Export Latents (z + conditions -> .npz) ===\n"
	$(PYTHON) scripts/export_latents.py \
	  --checkpoint $(RUN_DIR)/ae/best.pt \
	  --data_dir $(TRAIN_DIR) \
	  --output_npz $(RUN_DIR)/latents_train.npz \
	  --max_files $(TRAIN_FILES) \
	  --state_indices $(STATE_INDICES) \
	  --device $(DEVICE)
	$(PYTHON) scripts/export_latents.py \
	  --checkpoint $(RUN_DIR)/ae/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_npz $(RUN_DIR)/latents_test.npz \
	  --max_files $(TEST_FILES) \
	  --state_indices $(STATE_INDICES) \
	  --device $(DEVICE)

ddpm:
	@echo "\n=== STAGE 7/9: DDPM (latent diffusion, conditioned on f) ===\n"
	$(PYTHON) scripts/train_conditional_latent_ddpm.py \
	  --train_npz $(RUN_DIR)/latents_train.npz \
	  --val_npz $(RUN_DIR)/latents_test.npz \
	  --output_dir $(RUN_DIR)/ddpm \
	  --epochs $(EPOCHS) \
	  --batch_size $(DDPM_BATCH) \
	  --diffusion_steps $(DDPM_STEPS) \
	  --hidden_dim $(DDPM_HIDDEN) \
	  --num_layers $(DDPM_LAYERS) \
	  --device $(DEVICE) \
	  --seed $(SEED)

samples:
	@echo "\n=== STAGE 8/9: Sample (conditional image generation) ===\n"
	$(PYTHON) scripts/sample_conditional.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --diffusion_checkpoint $(RUN_DIR)/ddpm/best.pt \
	  --conditions_npz $(RUN_DIR)/latents_test.npz \
	  --output_dir $(RUN_DIR)/random_samples \
	  --num_samples 16 \
	  --device $(DEVICE) \
	  --seed $(SEED)

rollout:
	@echo "\n=== STAGE: Rollout Eval (real vs generated, crop-MSE) ===\n"
	$(PYTHON) scripts/eval_piwm_diffusion_rollout.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --dynamics_checkpoint $(RUN_DIR)/dynamics/best.pt \
	  --diffusion_checkpoint $(RUN_DIR)/ddpm/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/rollout \
	  --batch_size 16 \
	  --max_files $(TEST_FILES) \
	  --num_viz 16 \
	  --max_triplets_per_file 0 \
	  --device $(DEVICE) \
	  --seed $(SEED)
