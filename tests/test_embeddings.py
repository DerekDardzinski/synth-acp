"""Tests for synth_acp.embeddings module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from synth_acp.embeddings import (
    ModelDownloadError,
    _download_file,
    embedding_available,
)


class TestEmbeddingAvailable:
    def test_returns_false_when_deps_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        # Force re-evaluation by calling the function (it does try-import each time)
        assert embedding_available() is False

    def test_returns_true_when_deps_present(self):
        # If search deps are installed in test env, this should pass
        # If not installed, skip
        try:
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401
        except ImportError:
            pytest.skip("search deps not installed")
        assert embedding_available() is True


class TestDownloadFile:
    def test_success_writes_file_and_reports_progress(self, tmp_path):
        dest = tmp_path / "model.onnx"
        content = b"fake model data" * 100
        progress_calls = []

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(content))}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("synth_acp.embeddings.urllib.request.urlopen", return_value=mock_resp):
            _download_file(
                "https://example.com/model.onnx",
                dest,
                lambda downloaded, total: progress_calls.append((downloaded, total)),
            )

        assert dest.read_bytes() == content
        assert progress_calls == [(len(content), len(content))]

    def test_network_error_raises_model_download_error(self, tmp_path):
        dest = tmp_path / "model.onnx"

        with patch(
            "synth_acp.embeddings.urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ), pytest.raises(ModelDownloadError, match="connection refused"):
            _download_file("https://example.com/model.onnx", dest)

        assert not dest.exists()


@pytest.mark.skipif(
    not embedding_available(), reason="requires [search] optional deps (onnxruntime, tokenizers)"
)
class TestEmbeddingEngine:
    def test_ensure_model_downloads_missing_files(self, tmp_path):
        from synth_acp.embeddings import _HF_BASE, _MODEL_FILES, EmbeddingEngine

        engine = EmbeddingEngine(models_dir=tmp_path)
        calls = []

        with patch(
            "synth_acp.embeddings._download_file",
            side_effect=lambda url, dest, *_: calls.append((url, dest)),
        ):
            engine.ensure_model()

        expected = [
            (f"{_HF_BASE}/{path}", tmp_path / Path(path).name)
            for path in _MODEL_FILES.values()
        ]
        assert sorted(calls) == sorted(expected)

    def test_ensure_model_skips_existing_files(self, tmp_path):
        from synth_acp.embeddings import _MODEL_FILES, EmbeddingEngine

        # Pre-create the files
        for path in _MODEL_FILES.values():
            (tmp_path / Path(path).name).write_bytes(b"data")

        engine = EmbeddingEngine(models_dir=tmp_path)

        with patch("synth_acp.embeddings._download_file") as mock_dl:
            engine.ensure_model()

        mock_dl.assert_not_called()

    def test_embed_returns_normalized_384_vector(self, tmp_path):
        import numpy as np

        from synth_acp.embeddings import EmbeddingEngine

        engine = EmbeddingEngine(models_dir=tmp_path)

        # Mock tokenizer
        mock_encoding = MagicMock()
        mock_encoding.ids = list(range(10))
        mock_encoding.attention_mask = [1] * 10

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = mock_encoding

        # Mock ONNX session - return token embeddings of shape (1, 10, 384)
        token_embeddings = np.random.randn(1, 10, 384).astype(np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [token_embeddings]

        engine._tokenizer = mock_tokenizer
        engine._session = mock_session

        with patch("synth_acp.embeddings.embedding_available", return_value=True):
            result = engine.embed("hello world")

        assert result.shape == (384,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_embed_raises_when_unavailable(self, tmp_path):
        from synth_acp.embeddings import EmbeddingEngine, ModelNotAvailableError

        engine = EmbeddingEngine(models_dir=tmp_path)

        with patch("synth_acp.embeddings.embedding_available", return_value=False), \
             pytest.raises(ModelNotAvailableError):
            engine.embed("hello")

    def test_similarity_scores(self):
        import numpy as np

        from synth_acp.embeddings import EmbeddingEngine

        engine = EmbeddingEngine()
        query = np.array([1.0, 0.0])
        corpus = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = engine.similarity(query, corpus)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0])
