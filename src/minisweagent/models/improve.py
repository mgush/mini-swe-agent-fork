from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from minisweagent import Model
from minisweagent.models import get_model
from minisweagent.utils.log import logger


@dataclass
class ImproveModelConfig:
    model_name: str = "improve"
    model_kwargs: Any = None  # ignored
    injection: str = "Actually, I think I might be able to improve my last answer. Let me think again."


class ImproveModel(Model):
    def __init__(self, *, config_class: Callable = ImproveModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.model = get_model(config=self.config.model_kwargs)

    @property
    def n_calls(self) -> int:
        return self.model.n_calls

    @property
    def cost(self) -> float:
        return self.model.cost

    def get_template_vars(self) -> dict:
        return self.model.get_template_vars()

    def query(self, messages: list[dict]) -> dict:
        result = self.model.query(messages)
        original_content = result["content"]
        result["content"] = result["content"].replace("```bash\n", "```\n")
        result["content"] += f"\n\n{self.config.injection}"
        messages = [*messages, result]
        new_result = self.model.query(messages)
        new_result["original_content"] = original_content
        logger.debug(f"Original command: {original_content}\nImproved command: {new_result['content']}")
        return new_result
