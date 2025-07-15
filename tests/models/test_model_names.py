import os
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from typing_extensions import TypedDict, get_args

from pydantic_ai.models import KnownModelName

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModelName
    from pydantic_ai.models.bedrock import BedrockModelName
    from pydantic_ai.models.cohere import CohereModelName
    from pydantic_ai.models.gemini import GeminiModelName
    from pydantic_ai.models.groq import GroqModelName
    from pydantic_ai.models.mistral import MistralModelName
    from pydantic_ai.models.openai import OpenAIModelName

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='some model package was not installed'),
    pytest.mark.vcr,
]


@pytest.fixture(scope='module')
def vcr_config():  # pragma: lax no cover
    if not os.getenv('CI'):
        return {'record_mode': 'rewrite'}
    return {'record_mode': 'none'}


def test_known_model_names():
    def get_model_names(model_name_type: Any) -> Iterator[str]:
        for arg in get_args(model_name_type):
            if isinstance(arg, str):
                yield arg
            else:
                yield from get_model_names(arg)

    anthropic_names = [f'anthropic:{n}' for n in get_model_names(AnthropicModelName)] + [
        n for n in get_model_names(AnthropicModelName) if n.startswith('claude')
    ]
    cohere_names = [f'cohere:{n}' for n in get_model_names(CohereModelName)]
    google_names = [f'google-gla:{n}' for n in get_model_names(GeminiModelName)] + [
        f'google-vertex:{n}' for n in get_model_names(GeminiModelName)
    ]
    groq_names = [f'groq:{n}' for n in get_model_names(GroqModelName)]
    mistral_names = [f'mistral:{n}' for n in get_model_names(MistralModelName)]
    openai_names = [f'openai:{n}' for n in get_model_names(OpenAIModelName)] + [
        n for n in get_model_names(OpenAIModelName) if n.startswith('o1') or n.startswith('gpt') or n.startswith('o3')
    ]
    bedrock_names = [f'bedrock:{n}' for n in get_model_names(BedrockModelName)]
    deepseek_names = ['deepseek:deepseek-chat', 'deepseek:deepseek-reasoner']
    heroku_names = get_heroku_model_names()
    extra_names = [
        'test',
        'openai:o3-pro',
        'openai:moonshotai/kimi-k2',
        'xai:grok-4',
        # Missing names added to match KnownModelName
        'anthropic:claude-4-opus-20250514',
        'anthropic:claude-4-sonnet-20250514',
        'anthropic:claude-opus-4-0',
        'anthropic:claude-opus-4-20250514',
        'anthropic:claude-sonnet-4-0',
        'anthropic:claude-sonnet-4-20250514',
        'claude-4-opus-20250514',
        'claude-4-sonnet-20250514',
        'claude-opus-4-0',
        'claude-opus-4-20250514',
        'claude-sonnet-4-0',
        'claude-sonnet-4-20250514',
        # heroku models are fetched dynamically, exclude to avoid duplicates
        'o3',
        'o3-2025-04-16',
        'openai:o3',
        'openai:o3-2025-04-16',
        'openai:o4-mini',
        'openai:o4-mini-2025-04-16',
    ]

    generated_names = sorted(
        anthropic_names
        + cohere_names
        + google_names
        + groq_names
        + mistral_names
        + openai_names
        + bedrock_names
        + deepseek_names
        + heroku_names
        + extra_names
    )

    known_model_names = sorted(get_args(KnownModelName.__value__))
    assert generated_names == known_model_names


class HerokuModel(TypedDict):
    model_id: str
    regions: list[str]
    type: list[str]


def get_heroku_model_names():
    response = httpx.get('https://us.inference.heroku.com/available-models')

    if response.status_code != 200:
        pytest.skip(f'Heroku AI returned status code {response.status_code}')  # pragma: lax no cover

    heroku_models: list[HerokuModel] = response.json()

    models: list[str] = []
    for model in heroku_models:
        if 'text-to-text' in model['type']:
            models.append(f'heroku:{model["model_id"]}')
    return sorted(models)
