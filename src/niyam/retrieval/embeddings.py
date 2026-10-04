"""Text embeddings (bge-small-en-v1.5, 384-d) via fastembed's ONNX runtime: no PyTorch needed.

bge models expect queries and passages to be embedded differently; use embed_query for
search text and embed_passages for chunks.
"""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from niyam.config import get_settings

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    dim = 384

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path | None = None):
        from fastembed import TextEmbedding  # heavy import, only when actually embedding

        cache = cache_dir or Path(get_settings().data_dir) / "models"
        self._model = TextEmbedding(model_name, cache_dir=str(cache))

    def embed_passages(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(list(texts), batch_size=batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()


@lru_cache
def get_embedder() -> FastEmbedder:
    return FastEmbedder()
