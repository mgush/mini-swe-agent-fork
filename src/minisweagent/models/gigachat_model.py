import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

from minisweagent.models.giga import GigaChat

logger = logging.getLogger("gigachat_model")


@dataclass
class GigaChatModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""


class GigaChatAPIError(Exception):
    """Custom exception for GigaChat API errors."""

    pass


class GigaChatAuthenticationError(Exception):
    """Custom exception for GigaChat authentication errors."""

    pass


class GigaChatRateLimitError(Exception):
    """Custom exception for GigaChat rate limit errors."""

    pass


class GigaChatModel:
    def __init__(self, **kwargs):
        logger.critical(kwargs)
        # assert False
        self.client = GigaChat()
        # self.config = self.client.config
        self.config = GigaChatModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        # self._api_url = "https://gigachat.ai/api/v1/chat/completions"
        # self._api_key = os.getenv("GIGACHAT_API_KEY", "")

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                GigaChatAuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        # headers = {
        #     "Authorization": f"Bearer {self._api_key}",
        #     "Content-Type": "application/json",
        # }

        # payload = {
        #     "model": self.config.model_name,
        #     "prompt": messages,
        #     "usage": {"include": True},
        #     **(self.config.model_kwargs | kwargs),
        # }
        # logger.warning(kwargs)
        try:
            response = self.client.create_chat_completion(
                model=self.config.model_name,
                prompt=messages,
                temperature=0.01,
                top_p=1,
                **kwargs,
            )
            # response = requests.post(self._api_url, headers=headers, data=json.dumps(payload), timeout=60)
            # response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set GIGACHAT_API_KEY YOUR_KEY`."
                raise GigaChatAuthenticationError(error_msg) from e
            elif response.status_code == 429:
                raise GigaChatRateLimitError("Rate limit exceeded") from e
            else:
                raise GigaChatAPIError(f"HTTP {response.status_code}: {response.text}") from e
        except requests.exceptions.RequestException as e:
            raise GigaChatAPIError(f"Request failed: {e}") from e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query(messages, **kwargs)

        # Extract cost from usage information
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)
        assert cost >= 0.0, f"Cost is negative: {cost}"

        # If total_cost is not available, raise an error
        # if cost == 0.0:
        #     raise GigaChatAPIError(
        #         f"No cost information available from GigaChat API for model {self.config.model_name}. "
        #         "Cost tracking is required but not provided by the API response."
        #     )

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        try:
            return {
                "content": response["choices"][0]["message"]["content"] or "",
                "extra": {
                    "response": response,  # already is json
                },
            }
        except Exception as e:
            logger.critical(e)
            logger.critical("messages: %s", messages)
            logger.critical("response: %s", response)
            raise e

        # return result

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
