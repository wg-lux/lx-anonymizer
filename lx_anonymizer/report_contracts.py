from __future__ import annotations

from lx_dtypes.models.contracts.report_anonymization import (
    REPORT_ANONYMIZATION_CONTRACT_VERSION,
    ReportAnonymizationContractVersion,
    ReportAnonymizationErrorCode,
    ReportAnonymizationFailureV2,
    ReportAnonymizationOptions,
    ReportAnonymizationPhase,
    ReportAnonymizationProvenanceV2,
    ReportAnonymizationRequestV2,
    ReportAnonymizationResultV2,
    ReportAnonymizationWarningCode,
    ReportAnonymizationWarningV2,
    ReportArtifactValidationV2,
)


"""Compatibility imports and runtime errors for report anonymization v2.

The serializable contract is owned by lx_dtypes. Runtime exceptions remain in
lx-anonymizer because they describe local execution behavior, not persisted or
transported data.
"""


class ReportAnonymizationError(RuntimeError):
    """Base class for local report-anonymization execution failures."""


class ReportContractError(ReportAnonymizationError, ValueError):
    """The caller supplied an invalid report-processing contract."""


class SourceIdentityMismatchError(ReportContractError):
    """The source no longer matches the identity asserted by the caller."""


class ArtifactAlreadyExistsError(ReportContractError):
    """An attempt-local output already exists and must not be overwritten."""


class AnonymizationArtifactError(ReportAnonymizationError):
    """The anonymizer failed to produce a valid attempt-local artifact."""


class OperationDeadlineExceededError(ReportAnonymizationError, TimeoutError):
    """The host-assigned monotonic deadline expired."""


__all__ = [
    "REPORT_ANONYMIZATION_CONTRACT_VERSION",
    "AnonymizationArtifactError",
    "ArtifactAlreadyExistsError",
    "OperationDeadlineExceededError",
    "ReportAnonymizationContractVersion",
    "ReportAnonymizationError",
    "ReportAnonymizationErrorCode",
    "ReportAnonymizationFailureV2",
    "ReportAnonymizationOptions",
    "ReportAnonymizationPhase",
    "ReportAnonymizationProvenanceV2",
    "ReportAnonymizationRequestV2",
    "ReportAnonymizationResultV2",
    "ReportAnonymizationWarningCode",
    "ReportAnonymizationWarningV2",
    "ReportArtifactValidationV2",
    "ReportContractError",
    "SourceIdentityMismatchError",
]
