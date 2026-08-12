"""Run one GPU forward and backward pass through the 1D CNN."""

from __future__ import annotations

import torch

from signal_modulation.model import SimpleCNN1D, count_trainable_parameters


def main() -> None:
    torch.manual_seed(20260811)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN1D(num_classes=11).to(device)
    inputs = torch.randn(8, 2, 128, device=device)
    labels = torch.randint(0, 11, size=(8,), device=device)

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    print(f"device={device}")
    print(f"input_shape={tuple(inputs.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"label_shape={tuple(labels.shape)}")
    print(f"trainable_parameters={count_trainable_parameters(model)}")
    print(f"loss={loss.item():.6f}")
    print(f"backward_ok={model.classifier.weight.grad is not None}")
    print("accuracy_not_reported=true")


if __name__ == "__main__":
    main()
