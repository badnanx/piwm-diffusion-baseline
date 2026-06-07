import torch
import torch.nn as nn
import torch.nn.functional as F


class LunarSecondOrderDynamics(nn.Module):
    """
    Lightweight differentiable Lunar Lander dynamics for PIWM rollouts.

    The preferred state is 6D:
        [x, y, vx, vy, theta, omega]

    A 3D fallback is also supported:
        [x, y, theta]

    The model is intentionally structured and learnable rather than a perfect
    Box2D clone. It gives the architecture a physical transition module that
    can be trained from encoded physical states and actions.
    """

    def __init__(
        self,
        state_dim: int = 6,
        dt: float = 1.0 / 50.0,
        main_power: float = 0.22,
        side_power: float = 0.06,
        gravity: float = -0.06,
        angular_power: float = 0.08,
    ) -> None:
        super().__init__()
        if state_dim not in (3, 6):
            raise ValueError("LunarSecondOrderDynamics supports state_dim 3 or 6")
        self.state_dim = state_dim
        self.dt = dt

        self.main_power_raw = nn.Parameter(torch.tensor(float(main_power)))
        self.side_power_raw = nn.Parameter(torch.tensor(float(side_power)))
        self.angular_power_raw = nn.Parameter(torch.tensor(float(angular_power)))
        self.gravity = nn.Parameter(torch.tensor(float(gravity)))

    @property
    def main_power(self) -> torch.Tensor:
        return F.softplus(self.main_power_raw)

    @property
    def side_power(self) -> torch.Tensor:
        return F.softplus(self.side_power_raw)

    @property
    def angular_power(self) -> torch.Tensor:
        return F.softplus(self.angular_power_raw)

    def forward(
        self,
        state_prev: torch.Tensor,
        state_curr: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if self.state_dim == 6:
            x = state_curr[:, 0]
            y = state_curr[:, 1]
            vx = state_curr[:, 2]
            vy = state_curr[:, 3]
            theta = state_curr[:, 4]
            omega = state_curr[:, 5]
        else:
            x = state_curr[:, 0]
            y = state_curr[:, 1]
            theta = state_curr[:, 2]
            vx = (state_curr[:, 0] - state_prev[:, 0]) / self.dt
            vy = (state_curr[:, 1] - state_prev[:, 1]) / self.dt
            omega = (state_curr[:, 2] - state_prev[:, 2]) / self.dt

        action = action.long()
        fire_left = (action == 1).float()
        fire_main = (action == 2).float()
        fire_right = (action == 3).float()
        side_direction = fire_right - fire_left
        side_on = fire_left + fire_right

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        ax = -sin_t * self.main_power * fire_main
        ay = cos_t * self.main_power * fire_main + self.gravity

        side_ax = -cos_t * self.side_power * side_on * side_direction
        side_ay = -sin_t * self.side_power * side_on * side_direction
        alpha = self.angular_power * side_on * side_direction

        vx_next = vx + (ax + side_ax) * self.dt
        vy_next = vy + (ay + side_ay) * self.dt
        omega_next = omega + alpha * self.dt
        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt
        theta_next = theta + omega_next * self.dt

        if self.state_dim == 6:
            return torch.stack(
                [x_next, y_next, vx_next, vy_next, theta_next, omega_next],
                dim=1,
            )
        return torch.stack([x_next, y_next, theta_next], dim=1)
