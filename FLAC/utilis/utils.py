from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(
        self,
        shape: Union[int, Sequence[int]],
        *,
        epsilon: float = 1e-4,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)

        self.register_buffer("mean", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("var", torch.ones(shape, dtype=dtype, device=device))
        self.register_buffer("count", torch.tensor(epsilon, dtype=dtype, device=device))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return

        if x.ndim == self.mean.ndim:
            x = x.unsqueeze(0)

        x = x.to(dtype=self.mean.dtype)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int) -> None:
        batch_count_tensor = torch.tensor(batch_count, device=self.mean.device, dtype=self.mean.dtype)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count_tensor

        new_mean = self.mean + delta * batch_count_tensor / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count_tensor
        m_2 = m_a + m_b + (delta**2) * self.count * batch_count_tensor / total_count
        new_var = m_2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor, *, clip: Optional[float] = 10.0, eps: float = 1e-8) -> torch.Tensor:
        x = x.to(dtype=self.mean.dtype)
        x = (x - self.mean) / torch.sqrt(self.var + eps)
        if clip is not None:
            x = torch.clamp(x, -clip, clip)
        return x
