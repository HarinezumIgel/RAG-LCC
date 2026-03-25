"""
Custom exception classes for RAG-LCC application.

These exceptions are used to handle critical errors and compliance violations
while allowing proper cleanup through exception propagation.
"""


class RAGLCCException(Exception):
    """
    Base exception for all RAG-LCC errors.
    Use this as the root for custom exceptions in the application.
    """

    pass


class NoVirtualEnvError(RAGLCCException):
    """
    Raised when no Python virtual environment is detected.
    """

    pass


class RerankError(RAGLCCException):
    """
    Raised when reranking fails or is not possible.
    """

    pass


class OllamaNotRunning(RAGLCCException):
    """
    Raised when Ollama server is not running or unreachable.
    """

    pass


class LLMResultError(RAGLCCException):
    """
    Raised when LLM result is invalid or cannot be parsed.
    """

    pass


class PersistDirError(RAGLCCException):
    """
    Raised when persistence directory is missing or inaccessible.
    """

    pass


class ExclusionsError(RAGLCCException):
    """
    Raised when exclusions file or logic fails.
    """

    pass


class DocumentsDirError(RAGLCCException):
    """
    Raised when documents directory is missing or inaccessible.
    """

    pass


class ComplianceViolationError(RAGLCCException):
    """
    Raised when a compliance check fails.

    This is a critical error that must stop execution immediately.
    All cleanup handlers (finally blocks) will execute before termination.
    """

    pass


class PromptComplianceError(ComplianceViolationError):
    """Raised when user-provided prompt fails compliance validation."""

    pass


class LLMComplianceCheckError(ComplianceViolationError):
    """Raised when LLM compliance check returns non-compliant result."""

    pass


class DeviceConfigurationError(RAGLCCException):
    """
    Raised when system device configuration is invalid.

    Examples:
    - CPU requested with 16-bit embeddings (incompatible)
    - Invalid GPU/device index
    - Missing required hardware
    """

    pass


class ConfigurationError(RAGLCCException):
    """Raised when required configuration values are missing or invalid."""

    pass


class DataProcessingError(RAGLCCException):
    """Raised when document extraction or processing fails critically."""

    pass


class ClassifyCSVNotFoundError(RAGLCCException):
    """Raised when a required DocClassify CSV log file is not found."""

    pass


class ConfigPathError(RAGLCCException):
    """Raised when a configuration key path cannot be resolved."""

    pass


class InternetConnectionDisabledError(RAGLCCException):
    """Raised when internet access is required but not available."""

    pass


class HFDownloaderError(RAGLCCException):
    """Raised when a Hugging Face model download fails."""

    pass


class ModelLoadError(RAGLCCException):
    """Raised when LLM or embedding model fails to load."""

    pass


class CollectionNotFoundError(RAGLCCException):
    """Raised when a ChromaDB collection does not exist at the expected path."""

    pass


class ChromaInstallCurrentEmbeddingsMismatch(RAGLCCException):
    """
    Raised when ChromaDB refuses to update collection metadata with the current
    embedding stamp (e.g. because the distance function cannot be changed once set).
    Delete the collection directory and re-run RAGLoad to rebuild it with the
    current embedder settings.
    """

    pass


class EmbedModelMismatch(RAGLCCException):
    """
    Raised when the embedding model or quantization bits stored in a ChromaDB
    collection differ from the current configuration.

    The collection vectors were produced by a different model and cannot be
    queried reliably with the currently configured embedder.
    Re-run RAGLoad with CHROMA_COLLECTION_KEEP = False to rebuild the collection.
    """

    pass


class UserNoDownLoadAccept(RAGLCCException):
    """Raised Licenses are consented and user defines to pre-install embedder and cross-encoder or download from internet."""

    pass


class HfHubHTTPError(RAGLCCException):
    """Raised when Hugging Face Hub returns an HTTP error during model download."""

    pass


class InvalidCollectionName(RAGLCCException):
    """
    Raised when a collection name contains path separators or other
    characters that are not valid in a plain collection identifier.
    """

    pass


class ArgosConsentMissingError(ComplianceViolationError):
    """
    Raised when ARGOS_STANZA_DOWNLOAD is enabled but the user has not yet
    accepted the Argos Translate license via src/Scripts/ArgosTranslatePackages.py.
    """

    pass


class UnsupportedLanguageError(RAGLCCException):
    """Raised when a document's language is not installed and the configured
    action is NOT_OK, preventing processing."""

    pass
