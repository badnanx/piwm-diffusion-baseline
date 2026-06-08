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
CROP_SIZE ?= 32
CROP_LATENT_DIM ?= 16
CROP_EPOCHS ?= 20
CROP_DDPM_HIDDEN ?= 128
CROP_DDPM_LAYERS ?= 3
CROP_DDPM_STEPS ?= 15
SDEDIT_T_START ?= 7
CONSTRAINT_ALPHA ?= 0.05
REQUIRE_VISIBLE ?= 0
VISIBLE_FLAG = $(if $(filter 1,$(REQUIRE_VISIBLE)),--require_visible,)
SPRITE_SIZE ?= 32
SPRITE_LATENT_DIM ?= 16
SPRITE_DDPM_HIDDEN ?= 128
SPRITE_DDPM_LAYERS ?= 3
SPRITE_DDPM_STEPS ?= 50
SPRITE_EPOCHS ?= 20
SPRITE_SDEDIT_T_START ?= 7
DETECTED_POSITION ?= 0
CURRENT_THETA ?= 0
SPRITE_CONSTRAINT_ALPHA ?= 0.0
SPRITE_CONSTRAINT_STEPS ?= 1
DETECTED_FLAG = $(if $(filter 1,$(DETECTED_POSITION)),--use_detected_position,)
CURRENT_THETA_FLAG = $(if $(filter 1,$(CURRENT_THETA)),--use_current_theta,)
SPRITE_CONSTRAINT_FLAG = $(if $(filter-out 0.0,$(SPRITE_CONSTRAINT_ALPHA)),--constraint_alpha $(SPRITE_CONSTRAINT_ALPHA) --constraint_steps $(SPRITE_CONSTRAINT_STEPS),)
SEGMENT_ROLLOUT_SUFFIX = $(if $(filter-out 0.0,$(SPRITE_CONSTRAINT_ALPHA)),_constrained_a$(SPRITE_CONSTRAINT_ALPHA),)

.PHONY: full-fast full-fast-crop full-fast-crop-p4 full-segment ae ae-crop24 vq vq-crop24 physical dynamics export-latents ddpm samples rollout rollout-sdedit rollout-constrained rollout-all rollout-compare eval-physical eval-dynamics crop-ae export-crop-latents crop-ddpm p4-eval segment-ae export-segment-latents segment-ddpm segment-rollout check-cuda

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
	  --seed $(SEED) \
	  --stage_label "Stage 1/7 | AE (continuous VAE)"

ae-crop24:
	@echo "\n=== STAGE 1/9: AE (continuous VAE + crop-32 loss) ===\n"
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
	  --seed $(SEED) \
	  --stage_label "Stage 1/9 | AE (continuous VAE + crop-32)" \
	  $(VISIBLE_FLAG)

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
	  --seed $(SEED) \
	  --stage_label "Stage 1/7 | AE (VQ-VAE)"

vq-crop24:
	@echo "\n=== STAGE 1/9: AE (VQ-VAE + crop-32 loss) ===\n"
	$(PYTHON) scripts/train_vq_autoencoder.py \
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
	  --seed $(SEED) \
	  --stage_label "Stage 1/9 | AE (VQ-VAE + crop-32)"

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
	  --seed $(SEED) \
	  --stage_label "Stage 2/9 | Physical Encoder" \
	  $(VISIBLE_FLAG)

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
	  --seed $(SEED) \
	  --stage_label "Stage 4/9 | Dynamics Model" \
	  $(VISIBLE_FLAG)

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
	  --seed $(SEED) \
	  $(VISIBLE_FLAG)

eval-physical:
	@echo "\n=== STAGE 3/9: Eval Physical Encoder (scatter + R² per dim) ===\n"
	$(PYTHON) scripts/eval_physical_encoder.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/physical_eval \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED) \
	  $(VISIBLE_FLAG)

export-latents:
	@echo "\n=== STAGE 6/9: Export Latents (z + conditions -> .npz) ===\n"
	$(PYTHON) scripts/export_latents.py \
	  --checkpoint $(RUN_DIR)/ae/best.pt \
	  --data_dir $(TRAIN_DIR) \
	  --output_npz $(RUN_DIR)/latents_train.npz \
	  --max_files $(TRAIN_FILES) \
	  --state_indices $(STATE_INDICES) \
	  --device $(DEVICE) \
	  $(VISIBLE_FLAG)
	$(PYTHON) scripts/export_latents.py \
	  --checkpoint $(RUN_DIR)/ae/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_npz $(RUN_DIR)/latents_test.npz \
	  --max_files $(TEST_FILES) \
	  --state_indices $(STATE_INDICES) \
	  --device $(DEVICE) \
	  $(VISIBLE_FLAG)

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
	  --seed $(SEED) \
	  --stage_label "Stage 7/9 | DDPM"

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
	@echo "\n=== STAGE 9/9: Rollout Eval (real vs generated, crop-MSE) ===\n"
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
	  --seed $(SEED) \
	  $(VISIBLE_FLAG)

rollout-sdedit:
	@echo "\n=== SDEdit Rollout Eval (seed from physics decoder, t_start=$(SDEDIT_T_START)) ===\n"
	$(PYTHON) scripts/eval_piwm_diffusion_rollout.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --dynamics_checkpoint $(RUN_DIR)/dynamics/best.pt \
	  --diffusion_checkpoint $(RUN_DIR)/ddpm/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/rollout_sdedit_t$(SDEDIT_T_START) \
	  --batch_size 16 \
	  --max_files $(TEST_FILES) \
	  --num_viz 16 \
	  --max_triplets_per_file 0 \
	  --sdedit_t_start $(SDEDIT_T_START) \
	  --device $(DEVICE) \
	  --seed $(SEED) \
	  $(VISIBLE_FLAG)

rollout-constrained:
	@echo "\n=== Constrained SDEdit Rollout (t_start=$(SDEDIT_T_START), alpha=$(CONSTRAINT_ALPHA)) ===\n"
	$(PYTHON) scripts/eval_piwm_diffusion_rollout.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --dynamics_checkpoint $(RUN_DIR)/dynamics/best.pt \
	  --diffusion_checkpoint $(RUN_DIR)/ddpm/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/rollout_constrained_t$(SDEDIT_T_START)_a$(CONSTRAINT_ALPHA) \
	  --batch_size 16 \
	  --max_files $(TEST_FILES) \
	  --num_viz 16 \
	  --max_triplets_per_file 0 \
	  --sdedit_t_start $(SDEDIT_T_START) \
	  --constraint_alpha $(CONSTRAINT_ALPHA) \
	  --device $(DEVICE) \
	  --seed $(SEED) \
	  $(VISIBLE_FLAG)

rollout-compare:
	@echo "\n=== Rollout Comparison Table ===\n"
	$(PYTHON) scripts/compare_rollout_runs.py --run_dir $(RUN_DIR)

rollout-all: rollout rollout-sdedit rollout-constrained rollout-compare

# ── SegmentVAE pipeline ──────────────────────────────────────────────────────
# Shares AE + physical encoder + dynamics with full-fast-crop.
# Only the sprite VAE, sprite DDPM, and rollout eval are new.

segment-ae:
	@echo "\n=== SEGMENT 1/3: SpriteVAE (segmented lander sprites) ===\n"
	$(PYTHON) scripts/train_segment_ae.py \
	  --train_dir $(TRAIN_DIR) \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/segment_ae \
	  --latent_dim $(SPRITE_LATENT_DIM) \
	  --sprite_size $(SPRITE_SIZE) \
	  --epochs $(SPRITE_EPOCHS) \
	  --batch_size $(AE_BATCH) \
	  --max_train_files $(TRAIN_FILES) \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED) \
	  --stage_label "Segment 1/3 | SpriteVAE"

export-segment-latents:
	@echo "\n=== SEGMENT 2/3 (train): Export sprite latents + theta ===\n"
	$(PYTHON) scripts/export_segment_latents.py \
	  --checkpoint $(RUN_DIR)/segment_ae/best.pt \
	  --data_dir $(TRAIN_DIR) \
	  --output_npz $(RUN_DIR)/sprite_latents_train.npz \
	  --max_files $(TRAIN_FILES) \
	  --device $(DEVICE)
	@echo "\n=== SEGMENT 2/3 (test): Export sprite latents + theta ===\n"
	$(PYTHON) scripts/export_segment_latents.py \
	  --checkpoint $(RUN_DIR)/segment_ae/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_npz $(RUN_DIR)/sprite_latents_test.npz \
	  --max_files $(TEST_FILES) \
	  --device $(DEVICE)

segment-ddpm:
	@echo "\n=== SEGMENT 3/3: Sprite DDPM (conditioned on theta) ===\n"
	$(PYTHON) scripts/train_conditional_latent_ddpm.py \
	  --train_npz $(RUN_DIR)/sprite_latents_train.npz \
	  --val_npz $(RUN_DIR)/sprite_latents_test.npz \
	  --output_dir $(RUN_DIR)/sprite_ddpm \
	  --epochs $(SPRITE_EPOCHS) \
	  --batch_size $(DDPM_BATCH) \
	  --diffusion_steps $(SPRITE_DDPM_STEPS) \
	  --hidden_dim $(SPRITE_DDPM_HIDDEN) \
	  --num_layers $(SPRITE_DDPM_LAYERS) \
	  --device $(DEVICE) \
	  --seed $(SEED) \
	  --stage_label "Segment 3/3 | SpriteDDPM"

segment-rollout:
	@echo "\n=== Segment Rollout Eval ===\n"
	$(PYTHON) scripts/eval_segment_rollout.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --dynamics_checkpoint $(RUN_DIR)/dynamics/best.pt \
	  --sprite_ae_checkpoint $(RUN_DIR)/segment_ae/best.pt \
	  --sprite_ddpm_checkpoint $(RUN_DIR)/sprite_ddpm/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/segment_rollout$(SEGMENT_ROLLOUT_SUFFIX) \
	  --max_files $(TEST_FILES) \
	  --max_triplets_per_file 0 \
	  --batch_size 16 \
	  --num_viz 16 \
	  --sdedit_t_start $(SPRITE_SDEDIT_T_START) \
	  --device $(DEVICE) \
	  --seed $(SEED) \
	  $(VISIBLE_FLAG) \
	  $(DETECTED_FLAG) \
	  $(CURRENT_THETA_FLAG) \
	  $(SPRITE_CONSTRAINT_FLAG)

full-segment: check-cuda ae-crop24 physical eval-physical dynamics eval-dynamics segment-ae export-segment-latents segment-ddpm segment-rollout
	@echo "Full SegmentVAE pipeline complete."
	@echo "Inspect: $(RUN_DIR)/ae/recon_best.png"
	@echo "Inspect: $(RUN_DIR)/segment_ae/recon_best.png"
	@echo "Inspect: $(RUN_DIR)/segment_rollout/segment_rollout_comparison.png"
	@echo "Inspect: $(RUN_DIR)/segment_rollout/summary.json"

# ── P4: CropVAE + CropDDPM compositor ────────────────────────────────────────

crop-ae:
	@echo "\n=== P4 STAGE 1/4: CropVAE (24x24 lander patches) ===\n"
	$(PYTHON) scripts/train_crop_ae.py \
	  --train_dir $(TRAIN_DIR) \
	  --test_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/crop_ae \
	  --latent_dim $(CROP_LATENT_DIM) \
	  --crop_size $(CROP_SIZE) \
	  --epochs $(CROP_EPOCHS) \
	  --batch_size $(AE_BATCH) \
	  --max_train_files $(TRAIN_FILES) \
	  --max_test_files $(TEST_FILES) \
	  --device $(DEVICE) \
	  --seed $(SEED) \
	  --stage_label "P4 Stage 1/4 | CropVAE"

export-crop-latents:
	@echo "\n=== P4 STAGE 2/4: Export crop latents (mu + theta) ===\n"
	$(PYTHON) scripts/export_crop_latents.py \
	  --crop_ae_checkpoint $(RUN_DIR)/crop_ae/best.pt \
	  --data_dir $(TRAIN_DIR) \
	  --output_npz $(RUN_DIR)/crop_latents_train.npz \
	  --max_files $(TRAIN_FILES) \
	  --device $(DEVICE)
	$(PYTHON) scripts/export_crop_latents.py \
	  --crop_ae_checkpoint $(RUN_DIR)/crop_ae/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_npz $(RUN_DIR)/crop_latents_test.npz \
	  --max_files $(TEST_FILES) \
	  --device $(DEVICE)

crop-ddpm:
	@echo "\n=== P4 STAGE 3/4: Crop DDPM (conditioned on theta) ===\n"
	$(PYTHON) scripts/train_crop_ddpm.py \
	  --train_npz $(RUN_DIR)/crop_latents_train.npz \
	  --val_npz $(RUN_DIR)/crop_latents_test.npz \
	  --output_dir $(RUN_DIR)/crop_ddpm \
	  --epochs $(CROP_EPOCHS) \
	  --batch_size $(DDPM_BATCH) \
	  --hidden_dim $(CROP_DDPM_HIDDEN) \
	  --num_layers $(CROP_DDPM_LAYERS) \
	  --diffusion_steps $(CROP_DDPM_STEPS) \
	  --device $(DEVICE) \
	  --seed $(SEED) \
	  --stage_label "P4 Stage 3/4 | CropDDPM"

p4-eval:
	@echo "\n=== P4 STAGE 4/4: P4 Compositor Eval ===\n"
	$(PYTHON) scripts/eval_p4_compositor.py \
	  --autoencoder_checkpoint $(RUN_DIR)/ae/best.pt \
	  --physical_checkpoint $(RUN_DIR)/physical/best.pt \
	  --dynamics_checkpoint $(RUN_DIR)/dynamics/best.pt \
	  --crop_ae_checkpoint $(RUN_DIR)/crop_ae/best.pt \
	  --crop_ddpm_checkpoint $(RUN_DIR)/crop_ddpm/best.pt \
	  --data_dir $(TEST_DIR) \
	  --output_dir $(RUN_DIR)/p4_eval \
	  --state_indices $(STATE_INDICES) \
	  --max_files $(TEST_FILES) \
	  --max_triplets_per_file 0 \
	  --batch_size 16 \
	  --num_viz 16 \
	  --device $(DEVICE) \
	  --seed $(SEED)

full-fast-crop-p4: check-cuda ae-crop24 physical eval-physical dynamics eval-dynamics export-latents ddpm samples rollout crop-ae export-crop-latents crop-ddpm p4-eval
	@echo "Full laptop-fast PIWM + latent diffusion + P4 compositor run complete."
	@echo "Inspect: $(RUN_DIR)/ae/recon_best.png"
	@echo "Inspect: $(RUN_DIR)/crop_ae/recon_best.png"
	@echo "Inspect: $(RUN_DIR)/p4_eval/p4_rollout_real_vs_generated.png"
	@echo "Inspect: $(RUN_DIR)/rollout/rollout_real_vs_generated.png"
