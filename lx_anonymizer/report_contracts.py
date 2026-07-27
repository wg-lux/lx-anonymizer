from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReportContractError(ValueError):
    """The caller supplied an invalid report-processing contract."""


class SourceIdentityMismatchError(ReportContractError):
    """The source no longer matches the identity asserted by the caller."""


class ArtifactAlreadyExistsError(ReportContractError):
    """An attempt-local output already exists and must not be overwritten."""


class AnonymizationArtifactError(RuntimeError):
    """The anonymizer failed to produce a valid attempt-local artifact."""


class ReportAnonymizationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    use_ensemble: bool = False
    verbose: bool = True
    use_llm: bool | None = None


class ReportAnonymizationRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract_version: Literal["report_anonymization_v2"] = "report_anonymization_v2"
    attempt_id: UUID
    source_path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(gt=0)
    output_directory: Path
    create_anonymized_pdf: Literal[True] = True
    options: ReportAnonymizationOptions = Field(
        default_factory=ReportAnonymizationOptions
    )

    @model_validator(mode="after")
    def validate_local_paths(self) -> ReportAnonymizationRequestV2:
        if self.source_path.is_symlink() or not self.source_path.is_file():
            raise ReportContractError("source_path must be a regular non-symlink file")
        if self.output_directory.is_symlink() or not self.output_directory.is_dir():
            raise ReportContractError(
                "output_directory must be an existing non-symlink directory"
            )
        if self.source_path.resolve() == self.output_directory.resolve():
            raise ReportContractError(
                "source_path and output_directory must be different"
            )
        return self


class ReportAnonymizationProvenanceV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract_version: Literal["report_anonymization_v2"] = "report_anonymization_v2"
    implementation: Literal["lx_anonymizer.ReportReader"] = "lx_anonymizer.ReportReader"


class ReportAnonymizationResultV2(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
    )

    contract_version: Literal["report_anonymization_v2"] = "report_anonymization_v2"
    attempt_id: UUID
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_text: str
    anonymized_text: str
    extracted_metadata: dict[str, object]
    artifact_path: Path
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_size_bytes: int = Field(gt=0)
    provenance: ReportAnonymizationProvenanceV2 = Field(
        default_factory=ReportAnonymizationProvenanceV2
    )
    warnings: tuple[str, ...] = ()


__all__ = [
    "AnonymizationArtifactError",
    "ArtifactAlreadyExistsError",
    "ReportAnonymizationOptions",
    "ReportAnonymizationProvenanceV2",
    "ReportAnonymizationRequestV2",
    "ReportAnonymizationResultV2",
    "ReportContractError",
    "SourceIdentityMismatchError",
]
