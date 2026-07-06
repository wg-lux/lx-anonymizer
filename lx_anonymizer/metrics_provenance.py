from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence
from statistics import mean
from typing import Mapping, cast

from pydantic import BaseModel, ConfigDict, Field


def _package_version() -> str:
    try:
        return importlib.metadata.version("lx-anonymizer")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class AnonymizerProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    anonymizer_version: str = Field(default_factory=_package_version)
    detector_sources: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    proposal_counts: dict[str, int] = Field(default_factory=dict)


class RedactionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    page_count: int = 0
    redaction_region_count: int = 0
    detector_sources: list[str] = Field(default_factory=list)
    confidence_min: float | None = None
    confidence_max: float | None = None
    confidence_mean: float | None = None


def build_anonymizer_provenance(
    *,
    detector_sources: list[str] | None = None,
    model_names: list[str] | None = None,
    model_versions: dict[str, str] | None = None,
    proposal_counts: dict[str, int] | None = None,
) -> AnonymizerProvenance:
    return AnonymizerProvenance(
        detector_sources=sorted(set(detector_sources or [])),
        model_names=sorted(set(model_names or [])),
        model_versions=dict(model_versions or {}),
        proposal_counts=dict(proposal_counts or {}),
    )


def summarize_frame_observations(
    observations: list[Mapping[str, object]],
) -> tuple[list[str], dict[str, int]]:
    detector_sources: set[str] = set()
    proposal_counts = {
        "frame_observations": len(observations),
        "sensitive_frames": 0,
        "phi_regions": 0,
    }
    for observation in observations:
        if bool(observation.get("is_sensitive")):
            proposal_counts["sensitive_frames"] += 1
        source_tags = observation.get("source_tags")
        if isinstance(source_tags, Sequence) and not isinstance(
            source_tags, (str, bytes)
        ):
            for source_tag in cast(Sequence[object], source_tags):
                if isinstance(source_tag, str) and source_tag:
                    detector_sources.add(source_tag)
        regions = observation.get("phi_regions")
        if isinstance(regions, list):
            phi_regions = cast(list[object], regions)
            proposal_counts["phi_regions"] += len(phi_regions)
            if phi_regions:
                detector_sources.add("phi_detector")
    return sorted(detector_sources), proposal_counts


def summarize_pdf_redactions(
    rois_per_page: Mapping[int, list[tuple[int, int, int, int]]],
    *,
    detector_sources: list[str] | None = None,
    confidences: list[float] | None = None,
) -> RedactionSummary:
    confidence_values = list(confidences or [])
    region_count = sum(len(regions) for regions in rois_per_page.values())
    return RedactionSummary(
        page_count=len(rois_per_page),
        redaction_region_count=region_count,
        detector_sources=sorted(set(detector_sources or [])),
        confidence_min=min(confidence_values) if confidence_values else None,
        confidence_max=max(confidence_values) if confidence_values else None,
        confidence_mean=mean(confidence_values) if confidence_values else None,
    )
