import torch
import torch.nn.functional as F


SCALE = 30.0
VIEWPORT_W = 600
VIEWPORT_H = 400

WORLD_W = VIEWPORT_W / SCALE
WORLD_H = VIEWPORT_H / SCALE

HALF_WORLD_W = WORLD_W / 2.0
HALF_WORLD_H = WORLD_H / 2.0

LEG_DOWN = 18
HELIPAD_Y = WORLD_H / 4.0
Y_OFFSET = HELIPAD_Y + LEG_DOWN / SCALE


def state_xy_to_pixel(states: torch.Tensor, img_h: int, img_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert LunarLander normalized state x,y into approximate pixel coordinates."""
    x_state = states[:, 0]
    y_state = states[:, 1]

    world_x = x_state * HALF_WORLD_W + HALF_WORLD_W
    world_y = y_state * HALF_WORLD_H + Y_OFFSET

    px = world_x / WORLD_W * img_w
    py = img_h - (world_y / WORLD_H * img_h)
    return px, py


def crop_around_state(images: torch.Tensor, states: torch.Tensor, crop_size: int = 24) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError(f"Expected images with shape (B,C,H,W), got {images.shape}")

    batch_size, channels, img_h, img_w = images.shape
    device = images.device
    dtype = images.dtype

    px, py = state_xy_to_pixel(states.to(device=device, dtype=dtype), img_h, img_w)
    half = (crop_size - 1) / 2.0
    offsets = torch.linspace(-half, half, crop_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")

    sample_x = px[:, None, None] + xx[None, :, :]
    sample_y = py[:, None, None] + yy[None, :, :]

    norm_x = 2.0 * sample_x / max(img_w - 1, 1) - 1.0
    norm_y = 2.0 * sample_y / max(img_h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)

    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def state_guided_crop_mse(
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    states: torch.Tensor,
    crop_size: int = 24,
) -> torch.Tensor:
    pred_crop = crop_around_state(pred_images, states, crop_size=crop_size)
    target_crop = crop_around_state(target_images, states, crop_size=crop_size)
    return F.mse_loss(pred_crop, target_crop, reduction="mean")
