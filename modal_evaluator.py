"""
Modal-based evaluator for KernelBench RL environment.

This module provides a Modal app that runs GPU evaluation in the cloud,
integrated with the Tinker RL training loop.
"""

import os
from pathlib import Path
from typing import Dict, Any

import modal

# Modal app setup
app = modal.App("kernelbench-rl")

# GPU architecture mapping for TORCH_CUDA_ARCH_LIST
GPU_ARCH_MAPPING = {
    "L40S": ["Ada"],
    "H100": ["Hopper"],
    "A100": ["Ampere"],
    "A100-80GB": ["Ampere"],
    "L4": ["Ada"],
    "T4": ["Turing"],
    "A10G": ["Ampere"],
}

# Get the path to kernelbench
SCRIPT_DIR = Path(__file__).parent.absolute()
KERNELBENCH_PATH = os.environ.get("KERNELBENCH_PATH", str(SCRIPT_DIR / "KernelBench"))
SRC_DIR = os.path.join(KERNELBENCH_PATH, "src")

# Build Modal image with CUDA and dependencies
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Follow KernelBench's exact pattern
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .uv_sync(uv_project_dir=KERNELBENCH_PATH, extras=["gpu"])
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(SRC_DIR, remote_path="/root/src")  # must be last
)


@app.cls(image=image, timeout=600)
class KernelEvaluator:
    """Modal class for evaluating kernels on cloud GPUs."""

    @modal.method()
    def evaluate_kernel(
        self,
        reference_src: str,
        kernel_src: str,
        gpu_arch: list,
        backend: str = "triton",
        precision: str = "fp32",
        num_correct_trials: int = 3,
        num_perf_trials: int = 50,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a kernel against the reference on a cloud GPU.

        Args:
            reference_src: Source code of the reference Model
            kernel_src: Source code of the ModelNew to evaluate
            gpu_arch: GPU architecture list (e.g., ["Ada"] for L40S)
            backend: Kernel backend (triton, cuda, cute, tilelang)
            precision: Precision string (fp32, fp16, bf16)
            num_correct_trials: Number of trials for correctness check
            num_perf_trials: Number of trials for performance measurement
            verbose: Whether to print verbose output

        Returns:
            Dictionary with evaluation results
        """
        import torch

        # Import from kernelbench package
        from kernelbench.utils import set_gpu_arch
        from kernelbench.eval import eval_kernel_against_ref

        # Set GPU architecture
        set_gpu_arch(gpu_arch)

        # Map precision string to torch dtype
        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        torch_precision = precision_map.get(precision, torch.float32)

        try:
            result = eval_kernel_against_ref(
                original_model_src=reference_src.strip(),
                custom_model_src=kernel_src,
                device=0,
                precision=torch_precision,
                backend=backend,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                measure_performance=True,
                verbose=verbose,
            )

            if result is None:
                return {"success": False, "error": "eval returned None"}

            return {
                "success": True,
                "compiled": result.compiled,
                "correctness": result.correctness,
                "runtime": result.runtime if hasattr(result, 'runtime') else None,
                "ref_runtime": result.ref_runtime if hasattr(result, 'ref_runtime') else None,
            }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def evaluate_on_modal(
    reference_src: str,
    kernel_src: str,
    gpu: str = "L40S",
    backend: str = "triton",
    precision: str = "fp32",
    num_correct_trials: int = 3,
    num_perf_trials: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a kernel using Modal cloud GPUs.

    This is the main entry point for Modal evaluation.

    Args:
        reference_src: Source code of the reference Model
        kernel_src: Source code of the ModelNew to evaluate
        gpu: GPU type (L40S, H100, A100, T4, etc.)
        backend: Kernel backend (triton, cuda, cute, tilelang)
        precision: Precision (fp32, fp16, bf16)
        num_correct_trials: Number of trials for correctness check
        num_perf_trials: Number of trials for performance measurement
        verbose: Whether to print verbose output

    Returns:
        Dictionary with evaluation results
    """
    gpu_arch = GPU_ARCH_MAPPING.get(gpu, ["Ada"])

    with app.run():
        evaluator = KernelEvaluator.with_options(gpu=gpu)()
        result = evaluator.evaluate_kernel.remote(
            reference_src=reference_src,
            kernel_src=kernel_src,
            gpu_arch=gpu_arch,
            backend=backend,
            precision=precision,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
        )
        return result
