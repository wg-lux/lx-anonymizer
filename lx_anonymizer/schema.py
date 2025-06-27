# lx_anonymizer/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
from datetime import date

class PatientMeta(BaseModel):
    patient_first_name: Optional[str] = Field(None, description="Given name")
    patient_last_name:  Optional[str] = Field(None, description="Family name")
    patient_gender:     Optional[str] = Field(None, description="m/f/div/unknown")
    patient_dob:        Optional[date] = None
    casenumber:         Optional[str] = None

    @field_validator('patient_dob', mode='before')
    @classmethod
    def check_dob(cls, value: Any) -> Any:
        """Convert common non-date strings to None before validation."""
        if isinstance(value, str):
            # Handle common non-date strings returned by LLMs
            if value.lower() in ['n/a', 'na', 'unknown', 'none', 'null', '']:
                return None
        return value
