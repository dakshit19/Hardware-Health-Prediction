"""Project-specific exception types."""


class MotherboardHealthError(Exception):
    """Base exception for the project."""


class DependencyMissingError(MotherboardHealthError):
    """Raised when an optional or required dependency is missing."""


class ModelArtifactError(MotherboardHealthError):
    """Raised when a model artifact cannot be loaded or used."""


class InputValidationError(MotherboardHealthError):
    """Raised when a prediction input is invalid."""
