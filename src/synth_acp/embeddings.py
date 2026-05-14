"""Standalone embedding module for semantic session search.

Handles model download, tokenization, ONNX inference, and cosine similarity.
No broker or UI imports.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError

_HF_BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
_MODEL_FILES = {"model": "onnx/model_O4.onnx", "tokenizer": "tokenizer.json"}
_DEFAULT_MODELS_DIR = Path.home() / ".synth" / "models" / "all-MiniLM-L6-v2"


class ModelNotAvailableError(Exception):
    """Raised when onnxruntime or tokenizers are not installed."""


class ModelDownloadError(Exception):
    """Raised when model file download fails."""


def embedding_available() -> bool:
    """Return True if onnxruntime and tokenizers are importable."""
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def _download_file(
    url: str,
    dest: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Download a single file via urllib with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as fd:
                tmp_path = Path(fd.name)
                try:
                    downloaded = 0
                    while chunk := resp.read(65536):
                        fd.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total)
                except BaseException:
                    tmp_path.unlink(missing_ok=True)
                    raise
            shutil.move(str(tmp_path), str(dest))
    except (URLError, OSError) as e:
        raise ModelDownloadError(f"Failed to download {url}: {e}") from e


def _mean_pool_and_normalize(token_embeddings, attention_mask):
    """Apply mean pooling over token embeddings, then L2-normalize."""
    import numpy as np

    mask_expanded = attention_mask[:, :, None].astype(np.float32)
    summed = (token_embeddings * mask_expanded).sum(axis=1)
    counts = mask_expanded.sum(axis=1).clip(min=1e-9)
    pooled = summed / counts
    norm = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
    return (pooled / norm).squeeze(0)


class EmbeddingEngine:
    """Lazy-loading embedding engine. Thread-safe after initialization."""

    def __init__(self, models_dir: Path | None = None) -> None:
        """Initialize with model storage directory. Default: ~/.synth/models/all-MiniLM-L6-v2/"""
        self._models_dir = models_dir or _DEFAULT_MODELS_DIR
        self._session = None
        self._tokenizer = None

    def is_available(self) -> bool:
        """Return True if onnxruntime and tokenizers are importable."""
        return embedding_available()

    def ensure_model(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> None:
        """Download model files if not present. Raises ModelDownloadError on failure."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        for file_path in _MODEL_FILES.values():
            dest = self._models_dir / Path(file_path).name
            if not dest.exists():
                url = f"{_HF_BASE}/{file_path}"
                _download_file(url, dest, progress_callback)

    def embed(self, text: str):
        """Embed text into a pre-normalized 384-dim float32 vector.

        Loads model on first call. Truncates to 256 tokens.
        Returns: numpy array of shape (384,) with L2 norm = 1.0.
        Raises: ModelNotAvailableError if deps not installed.
        """
        if not self.is_available():
            raise ModelNotAvailableError("onnxruntime and tokenizers are required")

        import numpy as np

        if self._tokenizer is None:
            import tokenizers

            tokenizer_path = self._models_dir / "tokenizer.json"
            if not tokenizer_path.exists():
                self.ensure_model()
            self._tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_truncation(max_length=256)
            self._tokenizer.no_padding()

        if self._session is None:
            import onnxruntime

            model_path = self._models_dir / "model_O4.onnx"
            if not model_path.exists():
                self.ensure_model()
            self._session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

        encoding = self._tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        return _mean_pool_and_normalize(outputs[0], attention_mask)

    def similarity(self, query_embedding, corpus_embeddings):
        """Compute cosine similarities between query and corpus.

        Both must be pre-normalized.
        query_embedding: shape (384,)
        corpus_embeddings: shape (N, 384)
        Returns: shape (N,) similarity scores in [-1, 1].
        """
        return corpus_embeddings @ query_embedding
