"""Fast semantic-module screening injector for PyTorch workloads."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from research.reliability.injection.bitflip import FaultEvent


@dataclass(frozen=True)
class CandidatePoint:
    module_name: str
    invocation_index: int
    element_count: int


def candidate_module_names(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.SiLU))
    ]


def discover_candidate_points(model: torch.nn.Module, inputs: torch.Tensor) -> list[CandidatePoint]:
    """Discover executed semantic outputs, separating repeated module calls."""
    eligible = set(candidate_module_names(model))
    invocation_counts: dict[str, int] = defaultdict(int)
    points: list[CandidatePoint] = []
    handles = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, _args: tuple, output: torch.Tensor) -> None:
            invocation_counts[name] += 1
            points.append(CandidatePoint(name, invocation_counts[name], output.numel()))

        return hook

    for name, module in model.named_modules():
        if name in eligible:
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.inference_mode():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return points


def infer_with_fault(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    module_name: str,
    event: FaultEvent,
) -> torch.Tensor:
    """Run one inference and mutate the selected module output once."""
    output, _captured = _infer_with_fault(model, inputs, module_name=module_name, event=event, capture=False)
    return output


def infer_with_fault_and_capture(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    module_name: str,
    event: FaultEvent,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one injected inference and return the mutated selected activation."""
    output, captured = _infer_with_fault(model, inputs, module_name=module_name, event=event, capture=True)
    if captured is None:
        raise RuntimeError("selected fault invocation was not executed")
    return output, captured


def capture_candidate_output(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    module_name: str,
    invocation_index: int,
) -> torch.Tensor:
    """Capture a clean intermediate output at one semantic invocation."""
    named = dict(model.named_modules())
    if module_name not in named:
        raise KeyError(f"unknown module: {module_name}")
    invoked = 0
    captured: torch.Tensor | None = None

    def hook(_module: torch.nn.Module, _args: tuple, output: torch.Tensor) -> None:
        nonlocal invoked, captured
        invoked += 1
        if invoked == invocation_index:
            captured = output.detach().clone()

    handle = named[module_name].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(inputs)
    finally:
        handle.remove()
    if captured is None:
        raise RuntimeError("selected candidate invocation was not executed")
    return captured


def _infer_with_fault(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    module_name: str,
    event: FaultEvent,
    capture: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    named = dict(model.named_modules())
    if module_name not in named:
        raise KeyError(f"unknown module: {module_name}")
    invoked = 0
    captured: torch.Tensor | None = None

    def hook(_module: torch.nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
        nonlocal invoked, captured
        invoked += 1
        if invoked != event.invocation_index:
            return output
        if output.dtype != torch.float32:
            raise TypeError("only float32 module output injection is implemented")
        faulted = output.clone().contiguous()
        flat = faulted.view(-1)
        if event.element_index < 0 or event.element_index >= flat.numel():
            raise IndexError("fault event element index is outside module output")
        bits = flat.view(torch.int32)
        mask_value = -(1 << 31) if event.bit_index == 31 else 1 << event.bit_index
        mask = torch.tensor(mask_value, dtype=torch.int32, device=bits.device)
        bits[event.element_index] = torch.bitwise_xor(bits[event.element_index], mask)
        if capture:
            captured = faulted.detach().clone()
        return faulted

    handle = named[module_name].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            return model(inputs), captured
    finally:
        handle.remove()
