from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from lx_dtypes.models.meta.ReportMeta import ReportReaderFlags


DEFAULT_FIRST_NAMES = [
    "Markus",
    "Linhel",
    "Rainer",
    "Hans",
    "Anja",
    "Dorothea",
    "Doro",
    "Angelika",
    "Sven",
    "Theodor",
    "Alexander",
    "Mandy",
    "Kathrin",
    "Florian",
    "Philip",
    "Laura",
]

DEFAULT_LAST_NAMES = [
    "Kozielski",
    "Reiter",
    "Purrer",
    "Kudlich",
    "Brand",
    "Weich",
    "Lux",
    "Meining",
    "Hann",
    "Retzbach",
    "Hose",
    "Henniger",
    "Weich",
    "Dela Cruz",
    "Wiese",
    "Weise",
    "Sodmann",
]

DEFAULT_PATIENT_INFO_LINE_FLAG = "Patient: "
DEFAULT_ENDOSCOPE_INFO_LINE_FLAG = "Gerät: "
DEFAULT_EXAMINER_INFO_LINE_FLAG = "1. Unters.:"
DEFAULT_CUT_OFF_BELOW_LINE_FLAG = "________________"


def default_report_reader_flags() -> ReportReaderFlags:
    return ReportReaderFlags(
        patient_info_line=DEFAULT_PATIENT_INFO_LINE_FLAG,
        endoscope_info_line=DEFAULT_ENDOSCOPE_INFO_LINE_FLAG,
        examiner_info_line=DEFAULT_EXAMINER_INFO_LINE_FLAG,
        cut_off_below=[DEFAULT_CUT_OFF_BELOW_LINE_FLAG],
        cut_off_above=[
            DEFAULT_ENDOSCOPE_INFO_LINE_FLAG,
            DEFAULT_EXAMINER_INFO_LINE_FLAG,
        ],
    )


class ReportReaderSettings(BaseSettings):
    """Pydantic settings for report-reader parsing and anonymization defaults."""

    locale: str = "de_DE"
    first_names: list[str] = Field(default_factory=lambda: list(DEFAULT_FIRST_NAMES))
    last_names: list[str] = Field(default_factory=lambda: list(DEFAULT_LAST_NAMES))
    text_date_format: str = "%d.%m.%Y"
    flags: ReportReaderFlags = Field(default_factory=default_report_reader_flags)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="REPORT_READER_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("first_names", "last_names", mode="before")
    @classmethod
    def normalize_name_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ]
        return [str(value).strip()] if str(value).strip() else []

    @field_validator("flags", mode="before")
    @classmethod
    def normalize_flags(cls, value: Any) -> ReportReaderFlags | Mapping[str, Any]:
        if value is None:
            return default_report_reader_flags()
        if isinstance(value, ReportReaderFlags):
            return value
        if isinstance(value, Mapping):
            return value
        raise TypeError(
            f"flags must be a mapping or ReportReaderFlags, got {type(value)!r}"
        )

    @property
    def default_settings_dict(self) -> dict[str, Any]:
        return {
            "locale": self.locale,
            "first_names": list(self.first_names),
            "last_names": list(self.last_names),
            "text_date_format": self.text_date_format,
            "flags": self.flags.model_dump(),
        }


@lru_cache(maxsize=1)
def get_report_reader_settings() -> ReportReaderSettings:
    return ReportReaderSettings()


report_reader_settings = get_report_reader_settings()
DEFAULT_SETTINGS = report_reader_settings.default_settings_dict


__all__ = [
    "DEFAULT_CUT_OFF_BELOW_LINE_FLAG",
    "DEFAULT_ENDOSCOPE_INFO_LINE_FLAG",
    "DEFAULT_EXAMINER_INFO_LINE_FLAG",
    "DEFAULT_FIRST_NAMES",
    "DEFAULT_LAST_NAMES",
    "DEFAULT_PATIENT_INFO_LINE_FLAG",
    "DEFAULT_SETTINGS",
    "ReportReaderSettings",
    "default_report_reader_flags",
    "get_report_reader_settings",
    "report_reader_settings",
]
