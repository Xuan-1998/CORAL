"""Train solver-judge using the Python API.

Usage:
    python -m ttt.trainer +task_yaml=path/to/task.yaml

With Hydra overrides:
    python -m ttt.trainer +task_yaml=path/to/task.yaml model.name=Qwen/Qwen3-1.7B +repeat=5000
"""

import hydra
import yaml
from omegaconf import DictConfig

import ttt.generator as gen_module
from ttt.coral_trainer import CoralTrainer
from ttt.evaluator import evaluator
from ttt.generator import generator


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    task_yaml = config.get("task_yaml", None)
    if task_yaml is None:
        raise ValueError("task_yaml must be specified, e.g.: +task_yaml=path/to/task.yaml")

    with open(task_yaml) as f:
        task_config = yaml.safe_load(f)

    task_data = task_config.get("task", task_config)

    # Wire up the task_yaml path so the generator can find it
    gen_module._coral_state["task_yaml"] = str(task_yaml)

    trainer = CoralTrainer(
        agent_flow=generator,
        evaluator=evaluator,
        config=config,
        task_data=task_data,
        num_steps=int(config.get("repeat", 1000)),
    )
    trainer.train()


if __name__ == "__main__":
    main()
