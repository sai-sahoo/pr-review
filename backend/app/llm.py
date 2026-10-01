"""One place to create chat models.

Every agent asks for a model by *role* ("fast" or "smart") instead of a
hardcoded model name. The actual model comes from .env, so switching from
OpenAI to another provider is a config change, not a code change.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Load the project-root .env (one level above backend/) into os.environ.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

Role = Literal["fast", "smart"]

_DEFAULTS: dict[Role, str] = {
    "fast": "openai:gpt-6-luna",
    "smart": "openai:gpt-6.1-sol",
}


def get_model(role: Role = "smart", **kwargs) -> BaseChatModel:
    """Return a chat model for the given role, e.g. get_model("fast")."""
    model_id = os.getenv(f"MODEL_{role.upper()}", _DEFAULTS[role])
    # "openai:gpt-6.1-sol" -> provider "openai", model "gpt-6.1-sol"
    return init_chat_model(model_id, **kwargs)
