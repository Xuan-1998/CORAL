"""CoralTrainer — step-based trainer for CORAL agent training.

Wraps rllm's AgentTrainer with defaults suited to CORAL's training paradigm,
where each step launches agents, collects traces, and updates model weights.

Key differences from standard rllm training:
- Batch size is always 1 (one agent episode per step)
- Training length is controlled by ``num_steps``, not epochs
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from rllm.data.dataset import Dataset
from rllm.experimental.unified_trainer import AgentTrainer


class CoralTrainer:
    """Step-based trainer for CORAL agent training.

    CORAL training is inherently step-based: each training step launches agents,
    waits for eval attempts, collects LLM traces, and performs a weight update.
    This trainer translates a simple ``num_steps`` parameter into the epoch and
    batch configuration that rllm's ``UnifiedTrainer`` expects.

    Parameters
    ----------
    agent_flow:
        An ``@rllm.rollout``-decorated function that runs CORAL agents and
        returns an ``Episode`` of trajectories.
    evaluator:
        An ``@rllm.evaluator``-decorated function that computes rewards from
        episode metadata.
    config:
        Hydra/OmegaConf config (rllm unified config).
    task_data:
        A single task dict (from the task YAML).  The trainer repeats it
        to build training and validation datasets internally.
    num_steps:
        Total number of training steps.  Each step corresponds to one
        agent episode (start agents → collect traces → gradient update).
    backend:
        rllm training backend (default ``"tinker"``).
    """

    def __init__(
        self,
        *,
        agent_flow: Any,
        evaluator: Any,
        config: DictConfig,
        task_data: dict,
        num_steps: int = 1000,
        backend: str = "tinker",
    ):
        train_dataset = Dataset(data=[task_data] * num_steps, name="task", split="train")
        val_dataset = Dataset(data=[task_data] * max(1, num_steps // 10), name="task", split="test")

        self._apply_defaults(config, num_steps, len(train_dataset))

        self._trainer = AgentTrainer(
            backend=backend,
            agent_flow=agent_flow,
            evaluator=evaluator,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    @staticmethod
    def _apply_defaults(config: DictConfig, num_steps: int, dataset_size: int) -> None:
        """Apply CORAL training defaults without overriding explicit user config.

        - ``train_batch_size=1`` — CORAL produces one episode per step.
        - ``total_batches=num_steps`` — step-based termination via rllm's
          built-in ``total_batches`` mechanism instead of an inflated epoch count.
        - ``total_epochs`` — set just high enough to cover ``num_steps``.
        """
        # CORAL produces one episode per training step
        if config.get("data", {}).get("train_batch_size") is None:
            config.data.train_batch_size = 1
        if config.get("data", {}).get("val_batch_size") is None:
            config.data.val_batch_size = 1

        # Step-based termination
        if config.get("rllm", {}).get("trainer", {}).get("total_batches") is None:
            config.rllm.trainer.total_batches = num_steps

        # Ensure enough epochs so the outer loop doesn't exit before total_batches
        actual_batches = config.rllm.trainer.total_batches
        batch_size = config.data.train_batch_size
        batches_per_epoch = max(dataset_size // batch_size, 1)
        min_epochs = -(-actual_batches // batches_per_epoch)  # ceiling division
        if config.get("rllm", {}).get("trainer", {}).get("total_epochs") is None:
            config.rllm.trainer.total_epochs = min_epochs

    def train(self) -> None:
        """Run the training loop."""
        self._trainer.train()
