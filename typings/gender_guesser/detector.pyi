from typing import Literal

class Detector:
    def __init__(self, case_sensitive: bool = True) -> None: ...
    def get_gender(
        self, name: str, country: str | None = None
    ) -> Literal["male", "mostly_male", "female", "mostly_female", "andy", "unknown"]: ...
