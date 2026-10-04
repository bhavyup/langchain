"""Test embedding model integration."""

from unittest.mock import patch

from langchain_nomic.embeddings import NomicEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    NomicEmbeddings(model="nomic-embed-text-v1")


def test_embed_image_uses_vision_model() -> None:
    """vision_model should be passed to the SDK (see #41020)."""
    embeddings = NomicEmbeddings(
        model="nomic-embed-text-v1.5", vision_model="nomic-embed-vision-v1.5"
    )
    with patch(
        "langchain_nomic.embeddings.embed.image",
        return_value={"embeddings": [[0.1, 0.2]]},
    ) as mock_image:
        result = embeddings.embed_image(["https://example.com/cat.jpg"])
        assert result == [[0.1, 0.2]]
        mock_image.assert_called_once()
        assert mock_image.call_args.kwargs["model"] == "nomic-embed-vision-v1.5"


def test_embed_image_omits_model_when_vision_model_unset() -> None:
    """Without vision_model, model must be omitted so the SDK default applies."""
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    with patch(
        "langchain_nomic.embeddings.embed.image",
        return_value={"embeddings": [[0.1, 0.2]]},
    ) as mock_image:
        result = embeddings.embed_image(["https://example.com/cat.jpg"])
        assert result == [[0.1, 0.2]]
        mock_image.assert_called_once()
        assert "model" not in mock_image.call_args.kwargs
