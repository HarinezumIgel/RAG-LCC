from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from chromadb import Collection

from Commons.Exceptions import (
    ChromaInstallCurrentEmbeddingsMismatch,
    EmbedModelMismatch,
)
from Configuration import Config_Models
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.Helpers import Helpers

HF_MODEL = "snowflake/snowflake-arctic-embed-l-v2.0"
REMOTE_MODEL = "snowflake-embedder"


def _helper(endpoint: str = "ollama", bits: int = 32) -> ChromaDBHelper:
    cfg = MagicMock()
    cfg.get_str.side_effect = lambda key, default="", **kw: {
        "_ACTIVE_EMBED": "snowflake",
        "_ACTIVE_ENDPOINT": endpoint,
    }.get(key, default)
    cfg.get_int.side_effect = lambda key, default=0: (
        bits if key == "EMBEDDER_BITS" else 32
    )
    cfg.get_dict.return_value = Config_Models._MODELS
    helper = object.__new__(ChromaDBHelper)
    helper.cfg = cfg
    helper.pretty = MagicMock()
    helper.helpers = Helpers(cfg=cfg, pretty=helper.pretty)
    return helper


def _collection(
    monkeypatch: pytest.MonkeyPatch,
    helper: ChromaDBHelper,
    metadata: dict[str, Any],
) -> MagicMock:
    collection = MagicMock(spec=Collection)
    collection.metadata = metadata
    collection.count.return_value = 0
    monkeypatch.setattr(helper, "_persistent_chroma_client", MagicMock())
    monkeypatch.setattr(
        helper, "get_or_create_collection", MagicMock(return_value=collection)
    )
    return collection


@pytest.mark.parametrize("endpoint", ["ollama", "vllm"])
@pytest.mark.parametrize("stored_model", [HF_MODEL, REMOTE_MODEL, "old-served-name"])
@pytest.mark.parametrize("bits", [16, 32])
def test_alias_and_bits_accept_different_effective_models(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, stored_model: str, bits: int
) -> None:
    helper = _helper(endpoint, bits)
    collection = _collection(
        monkeypatch,
        helper,
        {
            "embed_model": stored_model,
            "embed_alias": "snowflake",
            "embed_bits": str(bits),
        },
    )

    _, actual = helper.get_chroma_client_and_collection("unused", "test")

    assert actual is collection
    collection.modify.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        {"embed_alias": "bge_m3", "embed_model": HF_MODEL, "embed_bits": "32"},
        {"embed_alias": "snowflake", "embed_model": HF_MODEL, "embed_bits": "16"},
        {"embed_alias": "snowflake", "embed_model": HF_MODEL, "embed_bits": 32},
        {"embed_model": HF_MODEL, "embed_bits": "16"},
    ],
)
@pytest.mark.parametrize("stamp", [False, True])
def test_mismatches_are_logged_and_never_stamped(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, Any], stamp: bool
) -> None:
    helper = _helper()
    collection = _collection(monkeypatch, helper, metadata)

    with pytest.raises(EmbedModelMismatch, match="alias="):
        helper.get_chroma_client_and_collection("unused", "test", stamp=stamp)

    assert isinstance(helper.pretty, MagicMock)
    helper.pretty.write.assert_called_once()
    collection.modify.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"embed_model": HF_MODEL, "embed_bits": "32"},
        {"embed_alias": "snowflake"},
    ],
)
@pytest.mark.parametrize("stamp", [False, True])
def test_nonempty_collections_without_identity_are_rejected(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, Any], stamp: bool
) -> None:
    helper = _helper()
    collection = _collection(monkeypatch, helper, metadata)
    collection.count.return_value = 1
    with pytest.raises(EmbedModelMismatch):
        helper.get_chroma_client_and_collection("unused", "test", stamp=stamp)
    collection.modify.assert_not_called()
    assert isinstance(helper.pretty, MagicMock)
    helper.pretty.write.assert_called_once()


def test_stamp_adds_alias_and_preserves_custom_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _helper("vllm")
    collection = _collection(
        monkeypatch,
        helper,
        {
            "embed_model": HF_MODEL,
            "embed_alias": "snowflake",
            "embed_bits": "32",
            "hnsw:space": "cosine",
            "hnsw:M": 32,
            "owner": "test",
        },
    )

    helper.get_chroma_client_and_collection("unused", "test", stamp=True)

    collection.modify.assert_called_once_with(
        metadata={
            "embed_model": REMOTE_MODEL,
            "embed_alias": "snowflake",
            "embed_bits": "32",
            "owner": "test",
        }
    )


def test_stamp_failure_is_logged_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _helper()
    collection = _collection(monkeypatch, helper, {})
    collection.modify.side_effect = ValueError("metadata rejected")

    with pytest.raises(
        ChromaInstallCurrentEmbeddingsMismatch, match="metadata rejected"
    ):
        helper.get_chroma_client_and_collection("unused", "test", stamp=True)

    assert isinstance(helper.pretty, MagicMock)
    helper.pretty.write.assert_called_once()


@pytest.mark.parametrize(
    "writer_endpoint,reader_endpoint",
    [
        ("ollama", "vllm"),
        ("vllm", "ollama"),
    ],
)
def test_real_chroma_persists_identity_and_reopens_across_backends(
    tmp_path: Path, writer_endpoint: str, reader_endpoint: str
) -> None:
    writer = _helper(writer_endpoint)
    directory = str(tmp_path)
    _, collection = writer.get_chroma_client_and_collection(
        directory, "portable", stamp=True
    )
    collection.add(ids=["chunk-1"], embeddings=[[1.0, 0.0]], documents=["test chunk"])

    reader = _helper(reader_endpoint)
    _, reopened = reader.get_chroma_client_and_collection(directory, "portable")

    assert dict(reopened.metadata or {}) == {
        "embed_model": HF_MODEL if writer_endpoint == "ollama" else REMOTE_MODEL,
        "embed_alias": "snowflake",
        "embed_bits": "32",
    }
    assert reopened.query(query_embeddings=[[1.0, 0.0]], n_results=1)["ids"] == [
        ["chunk-1"]
    ]

    with patch.object(reader.cfg, "get_int", return_value=16), pytest.raises(
        EmbedModelMismatch
    ):
        reader.get_chroma_client_and_collection(directory, "portable")


@pytest.mark.parametrize("stamp", [False, True])
def test_real_chroma_rejects_untagged_data_without_migrating(
    tmp_path: Path, stamp: bool
) -> None:
    helper = _helper()
    directory = str(tmp_path)
    client = helper._persistent_chroma_client(directory)
    collection = helper.get_or_create_collection(client, "untagged")
    collection.add(ids=["chunk-1"], embeddings=[[1.0, 0.0]])
    original_metadata = dict(collection.metadata or {})

    with pytest.raises(EmbedModelMismatch):
        helper.get_chroma_client_and_collection(directory, "untagged", stamp=stamp)

    assert dict(client.get_collection("untagged").metadata or {}) == original_metadata
    assert collection.count() == 1
