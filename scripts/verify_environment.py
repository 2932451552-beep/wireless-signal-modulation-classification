"""Verify the minimal NumPy and GPU-enabled PyTorch environment."""

from __future__ import annotations

import numpy as np
import torch


def main() -> None:
    """Run a small 1D convolution and backward pass on the selected device."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Conv1d(2, 8, kernel_size=5, padding=2).to(device)
    inputs = torch.randn(4, 2, 128, device=device)
    targets = torch.randn(4, 8, 128, device=device)

    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()

    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    restored = torch.from_numpy(source).to(device).cpu().numpy()

    print(f"torch_version={torch.__version__}")
    print(f"numpy_version={np.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device={device}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"forward_shape={tuple(outputs.shape)}")
    print(f"backward_ok={model.weight.grad is not None}")
    print(f"numpy_round_trip_ok={np.array_equal(source, restored)}")


if __name__ == "__main__":
    main()
