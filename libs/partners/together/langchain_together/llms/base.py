import os
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from together import Together

from langchain_together.version import __version__


class BaseTogetherAI(BaseLLM):
    """Base TogetherAI large language model class."""

    client: Any = Field(default=None, exclude=True)
    together_api_base: str = Field(
        default="https://api.together.ai/v1/", alias="base_url"
    )
    """Base completions API URL."""
    together_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Together AI API key. Get it here: https://api.together.xyz/settings/api-keys"""
    model_name: str = Field(default="meta-llama/Llama-3-8b-hf", alias="model")
    """Model name. Available models listed here: 
        Base Models: https://docs.together.ai/docs/inference-models#language-models
        Chat Models: https://docs.together.ai/docs/inference-models#chat-models
    """
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 0.7
    """Used to dynamically adjust the number of choices for each predicted token based 
        on the cumulative probabilities. A value of 1 will always yield the same 
        output. A temperature less than 1 favors more correctness and is appropriate 
        for question answering or summarization. A value greater than 1 introduces more 
        randomness in the output.
    """
    top_k: float = 50
    """Used to limit the number of choices for the next predicted word or token. It 
        specifies the maximum number of tokens to consider at each step, based on their 
        probability of occurrence. This technique helps to speed up the generation 
        process and can improve the quality of the generated text by focusing on the 
        most likely options.
    """
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""
    n: int = 1
    """How many completions to generate for each prompt."""
    repetition_penalty: float = 0
    """A number that controls the diversity of generated text by reducing the 
        likelihood of repeated sequences. Higher values decrease repetition.
    """
    logprobs: Optional[int] = None
    """An integer that specifies how many top token log probabilities are included in 
        the response for each token generation step.
    """
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, httpx.Timeout or 
        None."""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        together_api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = (
            convert_to_secret_str(together_api_key) if together_api_key else None
        )

        values["together_api_base"] = values["together_api_base"] or os.getenv(
            "TOGETHER_API_BASE"
        )

        client_params = {
            "api_key": (
                values["together_api_key"].get_secret_value()
                if values["together_api_key"]
                else None
            ),
            "base_url": values["together_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
        }
        values["client"] = Together(**client_params).completions
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "together"

    def _format_output(self, output: dict) -> str:
        return output["choices"][0]["text"]

    @staticmethod
    def get_user_agent() -> str:
        return f"langchain-together/{__version__}"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": self.n,
            "logprobs": self.logprobs,
        }
        if self.max_tokens is not None:
            normal_params["max_tokens"] = self.max_tokens

        return normal_params

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return self._default_params

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to TOGETHERAI's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The full LLM output.

        Example:
            .. code-block:: python

                response = togetherai.generate(["Tell me a joke."])
        """
        params = {**self._invocation_params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: Optional[str] = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")

                generation: Optional[GenerationChunk] = None
                for chunk in self._stream(_prompts[0], stop, run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None

                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": (
                            generation.generation_info.get("finish_reason")
                            if generation.generation_info
                            else None
                        ),
                        "logprobs": (
                            generation.generation_info.get("logprobs")
                            if generation.generation_info
                            else None
                        ),
                    }
                )
            else:
                response = self.client.create(prompt=_prompts, **params)

                # V1 client returns the response in an PyDantic object instead of
                # dict. For the transition period, we deep convert it to dict.
                response = (
                    response if isinstance(response, dict) else response.model_dump()
                )

                # Sometimes the AI Model calling will get error, we should raise it.
                # Otherwise, the next code 'choices.extend(response["choices"])'
                # will throw a "TypeError: 'NoneType' object is not iterable" error
                # to mask the true error. Because 'response["choices"]' is None.
                if response.get("error"):
                    raise ValueError(response.get("error"))

                choices.extend(response["choices"])
                _update_token_usage(_keys, response, token_usage)
                if not system_fingerprint:
                    system_fingerprint = response.get("system_fingerprint")

        return self.create_llm_result(
            choices,
            prompts,
            params,
            token_usage,
            system_fingerprint=system_fingerprint,
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {**self._invocation_params, **kwargs, "stream": True}
        self.get_sub_prompts(params, [prompt], stop)  # this mutates params
        for stream_resp in self.client.create(prompt=prompt, **params):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.model_dump()

            chunk = _stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                    logprobs=(
                        chunk.generation_info["logprobs"]
                        if chunk.generation_info
                        else None
                    ),
                )
            yield chunk

    def create_llm_result(
        self,
        choices: Any,
        prompts: List[str],
        params: Dict[str, Any],
        token_usage: Dict[str, int],
        *,
        system_fingerprint: Optional[str] = None,
    ) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []
        n = params.get("n", self.n)
        for i, _ in enumerate(prompts):
            sub_choices = choices[i * n : (i + 1) * n]
            generations.append(
                [
                    Generation(
                        text=choice["text"],
                        generation_info=dict(
                            finish_reason=choice.get("finish_reason"),
                            logprobs=choice.get("logprobs"),
                        ),
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        if system_fingerprint:
            llm_output["system_fingerprint"] = system_fingerprint
        return LLMResult(generations=generations, llm_output=llm_output)

    def get_sub_prompts(
        self,
        params: Dict[str, Any],
        prompts: List[str],
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Get the sub prompts for llm call."""
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["max_tokens"] = self.max_tokens
        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        return sub_prompts


class TogetherAI(BaseTogetherAI):
    """BaseTogetherAI large language models.

    To use, you should have the environment variable ``Together_API_KEY``
    set with your API key, or pass it as a named parameter to the constructor.

    Any parameters that are valid to be passed to the together.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_together import TogetherAI

            model = TogetherAI(model_name="meta-llama/Llama-3-8b-hf")
    """

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "together"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"together_api_key": "TOGETHER_API_KEY"}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.together_api_base:
            attributes["together_api_base"] = self.together_api_base

        return attributes


def _update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage with counts from response."""

    for key in keys.intersection(response.get("usage", {})):
        token_usage[key] = token_usage.get(key, 0) + response["usage"][key]


def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    if not stream_response["choices"]:
        return GenerationChunk(text="")
    return GenerationChunk(
        text=stream_response["choices"][0]["text"],
        generation_info=dict(
            finish_reason=stream_response["choices"][0].get("finish_reason", None),
            logprobs=stream_response["choices"][0].get("logprobs", None),
        ),
    )
