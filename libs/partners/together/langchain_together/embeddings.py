from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from together import Together  # type: ignore


class TogetherEmbeddings(BaseModel, Embeddings):
    """TogetherEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_together import TogetherEmbeddings

            model = TogetherEmbeddings(
                model='togethercomputer/m2-bert-80M-8k-retrieval'
            )
    """

    _client: Any = Field(default=None, exclude=True)  #: :meta private:
    together_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `TOGETHER_API_KEY` if not provided."""
    model: str = "BAAI/bge-large-en-v1.5"

    @root_validator()
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate environment variables."""
        together_api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = (
            convert_to_secret_str(together_api_key) if together_api_key else None
        )

        values["_client"] = Together()
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [
            i.embedding
            for i in self._client.embeddings.create(input=texts, model=self.model).data
        ]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
