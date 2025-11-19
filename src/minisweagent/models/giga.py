import json
import logging
import os
import time
from typing import Dict, List, Optional, Union, Literal
from uuid import uuid4

import requests
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from dotenv import load_dotenv
    load_dotenv("./.env")
    print(sorted(os.environ.keys()))
except ImportError:
    pass  # If dotenv isn't installed, silently skip


logger = logging.getLogger(__name__)

PROFANITY_CHECK: Optional[bool] = (
    os.getenv("GIGA_PROFANITY_CHECK").lower().startswith("t")
    or os.getenv("GIGA_PROFANITY_CHECK").lower().startswith("y")
    if isinstance(os.getenv("GIGA_PROFANITY_CHECK"), str)
    else None
)


class GigaParams(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    profanity_check: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def check_zero_temperature(cls, data: Dict) -> Dict:
        if data["temperature"] == 0.0:
            logger.warning("Updated generation parameters. Before: %s", data)
            data["temperature"] = 1.0
            data["top_p"] = 0
            logger.warning("Updated generation parameters. After: %s", data)
        return data


class GigaApiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="giga_")

    base_url: str
    auth_url: Optional[str] = None
    route_models: str = "/models"
    route_chat: str = "/chat/completions"


class GigaCreds(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="giga_")

    credentials: Optional[str] = None
    scope: Optional[str] = None


class GigaConfig(BaseModel):
    api: GigaApiConfig
    credentials: Optional[str] = None  # base64.b64encode(self.config.login_password.encode()).decode("utf-8")
    scope: Optional[str] = None

    @property
    def need_auth(self) -> bool:
        return self.credentials is not None

    @classmethod
    def from_envs(cls) -> "GigaConfig":
        api = GigaApiConfig()
        creds = GigaCreds()
        return cls(**{"api": api, "credentials": creds.credentials, "scope": creds.scope})


class GigaChat:
    REQUEST_TIMEOUT: int = 10

    def __init__(self):
        self.config: GigaConfig = GigaConfig.from_envs()
        self.token = None
        self.token_exp = None

    def get_token(self) -> None:
        logger.info("getting token")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid4()),
            "Authorization": f"Basic {self.config.credentials}",
        }
        api = self.config.api

        assert api.auth_url is not None, "GIGA_AUTH_URL is not set"
        tokens_data = json.loads(requests.post(url=api.auth_url, headers=headers, timeout=self.REQUEST_TIMEOUT).text)
        self.token = tokens_data["tok"]
        self.token_exp = tokens_data["exp"]

    @property
    def need_token_update(self) -> bool:
        return self.config.need_auth and (self.token is None or self.token_exp < time.time())

    def create_chat_completion(
        self,
        prompt: Union[List[Dict[str, str]], Dict[str, str]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        if self.need_token_update:
            self.get_token()

        # logger.info(prompt)
        api = self.config.api
        url = api.base_url + api.route_chat
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if not isinstance(prompt, list):
            prompt = [prompt]

        params = GigaParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            profanity_check=PROFANITY_CHECK,
        )

        data = {
            "model": model,
            "messages": prompt,
        }
        data.update(kwargs)
        data.update(params.model_dump(exclude_none=True))

        # logger.info("http request data %s", data)
        response_raw = requests.post(url=url, headers=headers, json=data, timeout=self.REQUEST_TIMEOUT)
        # logger.info("http response status code %s", response_raw.status_code)
        response = json.loads(response_raw.text)

        # logger.info(response)
        return response

        result = [choice["message"]["content"] for choice in response["choices"]]

        return result
