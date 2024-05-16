"""Test Together LLM"""

import json
from typing import Any, cast
from unittest.mock import MagicMock, patch  # AsyncMock

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import SecretStr
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from pytest import CaptureFixture, MonkeyPatch

from langchain_together import ChatTogetherAI


def test_together_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    llm = ChatTogetherAI(
        together_api_key="secret-api-key",
        model="togethercomputer/RedPajama-INCITE-7B-Base",
        temperature=0.2,
        max_tokens=250,
    )
    assert isinstance(llm.together_api_key, SecretStr)


def test_together_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("TOGETHER_API_KEY", "secret-api-key")
    llm = ChatTogetherAI(
        model="togethercomputer/RedPajama-INCITE-7B-Base",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm.together_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_together_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    llm = ChatTogetherAI(
        together_api_key="secret-api-key",
        model="togethercomputer/RedPajama-INCITE-7B-Base",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm.together_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_together_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = ChatTogetherAI(
        together_api_key="secret-api-key",
        model="togethercomputer/RedPajama-INCITE-7B-Base",
        temperature=0.2,
        max_tokens=250,
    )
    assert cast(SecretStr, llm.together_api_key).get_secret_value() == "secret-api-key"


def test_together_model_param() -> None:
    llm = ChatTogetherAI(model="foo")
    assert llm.model_name == "foo"


def test_function_dict_to_message_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
        {
            "role": "function",
            "name": name,
            "content": content,
        }
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def test__convert_dict_to_message_human_with_name() -> None:
    message = {"role": "user", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_ai_with_name() -> None:
    message = {"role": "assistant", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_system_with_name() -> None:
    message = {"role": "system", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_tool() -> None:
    message = {"role": "tool", "content": "foo", "tool_call_id": "bar"}
    result = _convert_dict_to_message(message)
    expected_output = ToolMessage(content="foo", tool_call_id="bar")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bab",
                    "name": "KimSolar",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_together_invoke(mock_completion: dict) -> None:
    llm = ChatTogetherAI()
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.invoke("bab")
        assert res.content == "Bab"
    assert completed


##TODO ASYNC CLINET SUPPORT IN PLANING
# async def test_together_ainvoke(mock_completion: dict) -> None:
#     llm = ChatTogetherAI()
#     mock_client = AsyncMock()
#     completed = False

#     async def mock_create(*args: Any, **kwargs: Any) -> Any:
#         nonlocal completed
#         completed = True
#         return mock_completion

#     mock_client.create = mock_create
#     with patch.object(
#         llm,
#         "async_client",
#         mock_client,
#     ):
#         res = await llm.ainvoke("bab")
#         assert res.content == "Bab"
#     assert completed


def test_together_invoke_name(mock_completion: dict) -> None:
    llm = ChatTogetherAI()

    mock_client = MagicMock()
    mock_client.create.return_value = mock_completion

    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        messages = [
            HumanMessage(content="Foo", name="Zorba"),
        ]
        res = llm.invoke(messages)
        call_args, call_kwargs = mock_client.create.call_args
        assert len(call_args) == 0  # no positional args
        call_messages = call_kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "Foo"
        assert call_messages[0]["name"] == "Zorba"

        # check return type has name
        assert res.content == "Bab"
        assert res.name == "KimSolar"
