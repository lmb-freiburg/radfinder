"""Custom exceptions for the Rad Text Engine (RaTE)."""


class ReportEngineError(Exception):
    """Base exception for Rad Text Engine (RaTE) errors."""

    pass


class BatchProcessingError(ReportEngineError):
    """Exception raised when batch processing fails."""

    pass


class ValidationError(ReportEngineError):
    """Exception raised when data validation fails."""

    pass


class StorageError(ReportEngineError):
    """Exception raised when storage operations fail."""

    pass


class ConfigurationError(ReportEngineError):
    """Exception raised when configuration is invalid."""

    pass
