"""
KernelBench RL Environment for Tinker.

This module implements a Tinker RL environment for optimizing GPU kernels
using KernelBench as the evaluation framework.
"""

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional

import chz
import tinker
from tinker import ModelInput
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook import model_info

from modal_evaluator import evaluate_on_modal


# System prompt for kernel optimization
KERNEL_OPTIMIZATION_SYSTEM_PROMPT = """
You are an expert GPU kernel optimizer. Your task is to optimize PyTorch models by writing custom GPU kernels.

Given a reference PyTorch implementation, you should:
1. Analyze the computation pattern
2. Identify optimization opportunities
3. Implement an optimized kernel using {backend}
4. Ensure correctness matches the reference
5. Maximize performance

Output only the complete Python code with the optimized ModelNew class.
""".strip()


class KernelBenchEnv(Env):
    """
    Single-episode RL environment for kernel optimization.

    The agent receives a reference model and must generate an optimized kernel.
    Reward is based on compilation success, correctness, and performance.
    """

    def __init__(
        self,
        reference_src: str,
        problem_name: str,
        problem_description: str,
        renderer: Renderer,
        gpu: str = "L40S",
        backend: str = "triton",
        precision: str = "fp32",
        num_correct_trials: int = 3,
        num_perf_trials: int = 50,
        verbose: bool = False,
    ):
        self.reference_src = reference_src
        self.problem_name = problem_name
        self.problem_description = problem_description
        self.renderer = renderer
        self.gpu = gpu
        self.backend = backend
        self.precision = precision
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.verbose = verbose
        self.done = False

        # Build system message
        self.system_message = {
            "role": "system",
            "content": KERNEL_OPTIMIZATION_SYSTEM_PROMPT.format(backend=backend),
        }

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    def _build_prompt(self) -> str:
        """Build the optimization prompt for the agent."""
        prompt = f"""Problem: {self.problem_name}

{self.problem_description}

Reference Implementation:
```python
{self.reference_src}
```

Task: Write an optimized version using {self.backend} kernels. Create a new class called ModelNew that improves upon the reference implementation.

Requirements:
- Use {self.backend} for kernel implementation
- Maintain correctness (outputs must match reference within tolerance)
- Optimize for performance on {self.gpu} GPU
- Use {self.precision} precision

Output the complete Python code with the optimized ModelNew class:
"""
        return prompt

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        """Return initial observation with the optimization task."""
        user_message = {
            "role": "user",
            "content": self._build_prompt(),
        }

        messages = [self.system_message, user_message]
        observation = self.renderer.build_generation_prompt(messages)

        if self.verbose:
            logtree.log_text(f"[KernelBench] Problem: {self.problem_name}")
            logtree.log_text(f"[KernelBench] Backend: {self.backend}, GPU: {self.gpu}")

        return observation, self.stop_condition

    def _extract_code(self, response_text: str) -> Optional[str]:
        """Extract Python code from the agent's response."""
        # Try to extract code from markdown code blocks
        code_block_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response_text, re.DOTALL)

        if matches:
            # Take the last code block (most likely to be complete)
            return matches[-1].strip()

        # If no code blocks, return the whole response
        # (agent might output code directly)
        return response_text.strip()

    def _compute_reward(self, eval_result: dict) -> float:
        """
        Compute reward from evaluation result.

        Reward structure:
        - 0.0: Doesn't compile
        - 0.1: Compiles but incorrect
        - 1000.0 / runtime_us: Correct (higher reward for faster kernels)
        """
        if not eval_result.get("success"):
            return 0.0

        if not eval_result.get("compiled"):
            return 0.0

        if not eval_result.get("correctness"):
            return 0.1

        # Correct! Reward based on performance
        runtime_us = eval_result.get("runtime")
        if runtime_us is None or runtime_us <= 0:
            # Fallback if timing failed
            return 1.0

        # Higher reward for faster kernels
        # Use 1000 / runtime_us so typical values are in reasonable range
        reward = 1000.0 / runtime_us

        return reward

    async def step(self, action: Action) -> StepResult:
        """
        Execute one step: evaluate the agent's proposed kernel.

        The episode ends after this step (single-turn optimization).
        """
        # Parse the action (generated code)
        action_message, _parse_success = self.renderer.parse_response(action)
        response_text = action_message.get("content", "")

        if self.verbose:
            logtree.log_text(f"[KernelBench] Agent response length: {len(response_text)} chars")

        # Extract code from response
        kernel_src = self._extract_code(response_text)

        if not kernel_src:
            # No code found
            if self.verbose:
                logtree.log_text("[KernelBench] Error: No code found in response")

            return StepResult(
                next_observation=await self._get_final_observation(),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
                metrics={"error": "no_code_found"},
            )

        # Evaluate on Modal
        if self.verbose:
            logtree.log_text(f"[KernelBench] Evaluating kernel on {self.gpu}...")

        try:
            eval_result = evaluate_on_modal(
                reference_src=self.reference_src,
                kernel_src=kernel_src,
                gpu=self.gpu,
                backend=self.backend,
                precision=self.precision,
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                verbose=self.verbose,
            )
        except Exception as e:
            if self.verbose:
                logtree.log_text(f"[KernelBench] Evaluation error: {e}")

            return StepResult(
                next_observation=await self._get_final_observation(),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
                metrics={"error": str(e)},
            )

        # Compute reward
        reward = self._compute_reward(eval_result)

        # Build metrics for W&B logging
        # These will be aggregated and logged automatically by Tinker
        metrics = {
            "env/compiled": 1.0 if eval_result.get("compiled") else 0.0,
            "env/correct": 1.0 if eval_result.get("correctness") else 0.0,
            "env/reward": reward,  # Explicitly include reward in metrics
        }

        if eval_result.get("runtime"):
            metrics["env/runtime_us"] = eval_result["runtime"]
        if eval_result.get("ref_runtime"):
            metrics["env/ref_runtime_us"] = eval_result["ref_runtime"]
            # Compute speedup
            if eval_result["runtime"] > 0:
                speedup = eval_result["ref_runtime"] / eval_result["runtime"]
                metrics["env/speedup"] = speedup

        # Add logs for debugging (not aggregated, just displayed)
        logs = {
            "problem": self.problem_name,
            "backend": self.backend,
            "gpu": self.gpu,
        }

        # Log results
        if self.verbose:
            logtree.log_text(
                f"[KernelBench] Result - Compiled: {metrics.get('env/compiled', 0)}, "
                f"Correct: {metrics.get('env/correct', 0)}, "
                f"Runtime: {metrics.get('env/runtime_us', 'N/A')} us, "
                f"Speedup: {metrics.get('env/speedup', 'N/A')}x, "
                f"Reward: {reward:.4f}"
            )

        self.done = True

        return StepResult(
            next_observation=await self._get_final_observation(),
            next_stop_condition=self.stop_condition,
            episode_done=True,
            reward=reward,
            metrics=metrics,
            logs=logs,
        )

    async def _get_final_observation(self) -> ModelInput:
        """Return a dummy final observation (episode is done)."""
        # Return empty observation since episode is over
        return self.renderer.build_generation_prompt([
            {"role": "system", "content": "Episode complete."}
        ])


@dataclass(frozen=True)
class KernelBenchEnvGroupBuilder(EnvGroupBuilder):
    """Builder for creating groups of KernelBench environments."""

    reference_src: str
    problem_name: str
    problem_description: str
    renderer: Renderer
    num_envs: int
    gpu: str = "L40S"
    backend: str = "triton"
    precision: str = "fp32"
    num_correct_trials: int = 3
    num_perf_trials: int = 50
    verbose: bool = False

    async def make_envs(self) -> Sequence[Env]:
        """Create multiple identical environments for the same problem."""
        return [
            KernelBenchEnv(
                reference_src=self.reference_src,
                problem_name=self.problem_name,
                problem_description=self.problem_description,
                renderer=self.renderer,
                gpu=self.gpu,
                backend=self.backend,
                precision=self.precision,
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                verbose=self.verbose,
            )
            for _ in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        """Return tags for metric aggregation in W&B."""
        return ["kernelbench", self.backend, self.gpu, self.problem_name]


@dataclass(frozen=True)
class KernelBenchDataset(RLDataset):
    """Dataset of KernelBench problems for RL training."""

    problems: Sequence[dict]  # List of problem dicts with 'code', 'name', 'description'
    renderer: Renderer
    batch_size: int
    group_size: int
    gpu: str = "L40S"
    backend: str = "triton"
    precision: str = "fp32"
    num_correct_trials: int = 3
    num_perf_trials: int = 50
    verbose: bool = False

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        batch_builders = []

        for i in range(self.batch_size):
            problem_idx = (index * self.batch_size + i) % len(self.problems)
            problem = self.problems[problem_idx]

            builder = KernelBenchEnvGroupBuilder(
                reference_src=problem["code"],
                problem_name=problem["name"],
                problem_description=problem.get("description", ""),
                renderer=self.renderer,
                num_envs=self.group_size,
                gpu=self.gpu,
                backend=self.backend,
                precision=self.precision,
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                verbose=self.verbose,
            )
            batch_builders.append(builder)

        return batch_builders

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return (len(self.problems) + self.batch_size - 1) // self.batch_size


@chz.chz
class KernelBenchDatasetBuilder(RLDatasetBuilder):
    """Builder for KernelBench RL datasets."""

    level: int  # KernelBench level (1-4)
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    gpu: str = "L40S"
    backend: str = "triton"
    precision: str = "fp32"
    num_correct_trials: int = 3
    num_perf_trials: int = 50
    num_epochs: int = 1
    test_split: float = 0.2  # Fraction of problems to use for testing
    verbose: bool = False
    dataset_source: str = "huggingface"  # "huggingface" or "local"

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        """Build train and test datasets."""
        # Load problems from KernelBench
        problems = self._load_problems()

        # Split into train/test
        num_test = max(1, int(len(problems) * self.test_split))
        train_problems = problems[:-num_test] * self.num_epochs
        test_problems = problems[-num_test:]

        # Create renderer
        renderer = get_renderer(
            self.renderer_name,
            get_tokenizer(self.model_name_for_tokenizer)
        )

        # Create datasets
        train_dataset = KernelBenchDataset(
            problems=train_problems,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            gpu=self.gpu,
            backend=self.backend,
            precision=self.precision,
            num_correct_trials=self.num_correct_trials,
            num_perf_trials=self.num_perf_trials,
            verbose=self.verbose,
        )

        test_dataset = KernelBenchDataset(
            problems=test_problems,
            renderer=renderer,
            batch_size=min(self.batch_size, len(test_problems)),
            group_size=self.group_size,
            gpu=self.gpu,
            backend=self.backend,
            precision=self.precision,
            num_correct_trials=self.num_correct_trials,
            num_perf_trials=self.num_perf_trials,
            verbose=self.verbose,
        )

        return train_dataset, test_dataset

    def _load_problems(self) -> list[dict]:
        """Load KernelBench problems from HuggingFace or local."""
        if self.dataset_source == "huggingface":
            from datasets import load_dataset

            dataset = load_dataset("ScalingIntelligence/KernelBench")
            level_dataset = dataset[f"level_{self.level}"]

            problems = []
            for row in level_dataset:
                problems.append({
                    "code": row["code"],
                    "name": row.get("name", f"problem_{row.get('problem_id', len(problems))}"),
                    "description": self._get_problem_description(row),
                    "problem_id": row.get("problem_id"),
                })

            return problems

        else:  # local
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "KernelBench" / "src"))
            from kernelbench.dataset import construct_kernelbench_dataset
            from kernelbench.utils import read_file

            level_dataset = construct_kernelbench_dataset(self.level)

            problems = []
            for i, problem_path in enumerate(level_dataset):
                code = read_file(problem_path)
                problems.append({
                    "code": code,
                    "name": Path(problem_path).name,
                    "description": f"Optimize the kernel in {Path(problem_path).name}",
                    "problem_id": i + 1,
                })

            return problems

    def _get_problem_description(self, row: dict) -> str:
        """Generate a description for the problem."""
        # You can customize this based on what's in the dataset
        desc_parts = []

        if "name" in row:
            desc_parts.append(f"Optimize: {row['name']}")

        # Add any other relevant info from the row
        if "model_name" in row:
            desc_parts.append(f"Model: {row['model_name']}")

        if desc_parts:
            return " | ".join(desc_parts)

        return "Optimize this GPU kernel for maximum performance while maintaining correctness."
