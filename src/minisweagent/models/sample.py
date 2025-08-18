import re
from collections.abc import Callable
from dataclasses import dataclass

from jinja2 import Template

from minisweagent import Model
from minisweagent.models import get_model


@dataclass
class SampleModelConfig:
    decider_template: str
    decider_model_kwargs: dict
    sample_model_kwargs: list[dict]
    model_name: str = "sample"
    n_samples: int = 10


class SampleModel(Model):
    def __init__(self, *, config_class: Callable = SampleModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.decider_model = get_model(**self.config.decider_model_kwargs)
        self.sample_models = [get_model(**config) for config in self.config.sample_model_kwargs]

    def _get_samples(self, messages: list[dict]) -> list[dict]:
        actions = []
        for i_sample in range(self.config.n_samples):
            model = self.sample_models[i_sample % len(self.sample_models)]
            response = model.query(messages)
            actions = re.findall(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)
            if len(actions) != 1:
                continue
            actions.append(actions[0].strip())
        return list(set(actions))

    def query(self, messages: list[dict]) -> dict:
        actions = self._get_samples(messages)
        prompt = Template(self.config.decider_template).render(actions=actions)
        messages = [*messages, {"role": "user", "content": prompt}]
        return self.decider_model.query(messages)
