from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol, Sequence, TextIO, cast

import gender_guesser.detector as gender  # type: ignore[import-untyped]

from lx_anonymizer.pseudonymization import names_adder
from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.setup.directory_setup import create_temp_directory
from lx_dtypes.models.contracts.text_anonymization import (
    DateOfBirthCore,
    GenderDisplayLabel,
    GenderGuess,
    PersonNameMetadata,
)

logger = get_logger(__name__)

temp_dir, data_base_dir, csv_dir = create_temp_directory()


class _GenderDetector(Protocol):
    def get_gender(self, name: str) -> str: ...


class _AddNameToImage(Protocol):
    def __call__(
        self,
        first_name: str,
        last_name: str,
        gender_par: str,
        first_name_box: tuple[int, int, int, int],
        last_name_box: tuple[int, int, int, int],
        device: str,
    ) -> Path: ...


class _AddFullNameToImage(Protocol):
    def __call__(
        self, name: str, gender_par: str, box: tuple[int, int, int, int]
    ) -> Path: ...


class _AddDeviceNameToImage(Protocol):
    def __call__(self, name: str, gender_par: str, device: str) -> Path: ...


BoundingBox = tuple[int, int, int, int]
DetectedWords = Sequence[str]


_gender_detector = cast(_GenderDetector, gender.Detector())
_add_name_to_image = cast(_AddNameToImage, names_adder.add_name_to_image)
_add_full_name_to_image = cast(_AddFullNameToImage, names_adder.add_full_name_to_image)
_add_device_name_to_image = cast(
    _AddDeviceNameToImage, names_adder.add_device_name_to_image
)


# Define the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Define file paths
names_dict_dir = Path(parent_dir) / "names_dict"
female_names_file = Path(names_dict_dir) / "first_and_last_names_female_ascii.txt"
male_names_file = Path(names_dict_dir) / "first_and_last_names_male_ascii.txt"
female_first_names_file = Path(names_dict_dir) / "first_names_female_ascii.txt"
female_last_names_file = Path(names_dict_dir) / "last_names_female_ascii.txt"
neutral_first_names_file = Path(names_dict_dir) / "first_names_neutral_ascii.txt"
neutral_last_names_file = Path(names_dict_dir) / "last_names_neutral_ascii.txt"
male_first_names_file = Path(names_dict_dir) / "first_names_male_ascii.txt"
male_last_names_file = Path(names_dict_dir) / "last_names_male_ascii.txt"


def load_names(file_path: Path) -> list[str]:
    with file_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


female_names: list[str] = load_names(female_names_file)
male_names: list[str] = load_names(male_names_file)
neutral_first_names: list[str] = load_names(neutral_first_names_file)
neutral_last_names: list[str] = load_names(neutral_last_names_file)
female_first_names: list[str] = load_names(female_first_names_file)
female_last_names: list[str] = load_names(female_last_names_file)
male_first_names: list[str] = load_names(male_first_names_file)
male_last_names: list[str] = load_names(male_last_names_file)


def _pick_random_name(choices: list[str]) -> str:
    index = random.randint(0, len(choices) - 1)
    return choices[index]


def _normalize_gender_guess(name: str) -> GenderGuess:
    guess = _gender_detector.get_gender(name).lower().strip()
    for gender_value in GenderGuess:
        if gender_value.value == guess:
            return gender_value
    return GenderGuess.UNKNOWN


def get_random_full_name(name: str) -> str:
    gender_guess = _normalize_gender_guess(name)
    if gender_guess in [GenderGuess.MALE, GenderGuess.MOSTLY_MALE]:
        logger.info("Male gender")
        return f"{_pick_random_name(male_first_names)} {_pick_random_name(male_last_names)}"
    if gender_guess in [GenderGuess.FEMALE, GenderGuess.MOSTLY_FEMALE]:
        logger.info("Female gender")
        return f"{_pick_random_name(female_first_names)} {_pick_random_name(female_last_names)}"
    logger.info("Neutral or unknown gender")
    return f"{_pick_random_name(neutral_first_names)} {_pick_random_name(neutral_last_names)}"


def person_meta(name: str) -> PersonNameMetadata:
    gender_guess = _normalize_gender_guess(name)
    if gender_guess in [GenderGuess.MALE, GenderGuess.MOSTLY_MALE]:
        logger.info("Male gender")
        first_name = _pick_random_name(male_first_names)
        last_name = _pick_random_name(male_last_names)
        gender_label = GenderDisplayLabel.MALE
    elif gender_guess in [GenderGuess.FEMALE, GenderGuess.MOSTLY_FEMALE]:
        logger.info("Female gender")
        first_name = _pick_random_name(female_first_names)
        last_name = _pick_random_name(female_last_names)
        gender_label = GenderDisplayLabel.FEMALE
    else:
        logger.info("Neutral or unknown gender")
        first_name = _pick_random_name(neutral_first_names)
        last_name = _pick_random_name(neutral_last_names)
        gender_label = GenderDisplayLabel.NEUTRAL
    return PersonNameMetadata(
        first_name=first_name,
        last_name=last_name,
        dob=DateOfBirthCore(
            day=random.randint(1, 28),
            month=random.randint(1, 12),
            year=random.randint(1950, 2020),
        ),
        gender_label=gender_label,
    )


def getindex(file: TextIO) -> int:
    file_length = len(file.readlines())
    file.seek(0)
    return random.randint(0, file_length - 1)


def _to_box_key(box: BoundingBox) -> str:
    return f"{box[0]},{box[1]},{box[2]},{box[3]}"


def gender_and_handle_full_names(
    words: DetectedWords,
    box: BoundingBox,
    image_path: str,
    device: str = "olympus_cv_1500",
) -> tuple[dict[tuple[str, str], Path], GenderGuess]:
    if not words:
        raise ValueError("Expected detected words to contain at least one entry")
    box_to_image_map: dict[tuple[str, str], Path] = {}
    gender_guess = _normalize_gender_guess(words[0])
    if gender_guess in [GenderGuess.MALE, GenderGuess.MOSTLY_MALE]:
        name = random.choice(male_names)
        output_image_path = _add_full_name_to_image(name, "male", box)
    elif gender_guess in [GenderGuess.FEMALE, GenderGuess.MOSTLY_FEMALE]:
        name = random.choice(female_names)
        output_image_path = _add_full_name_to_image(name, "female", box)
    else:
        name = random.choice(female_names + male_names)
        output_image_path = _add_full_name_to_image(name, "neutral", box)

    box_to_image_map[(_to_box_key(box), image_path)] = output_image_path
    return box_to_image_map, gender_guess


def gender_and_handle_separate_names(
    words: DetectedWords,
    first_name_box: BoundingBox,
    last_name_box: BoundingBox,
    image_path: str,
    device: str,
) -> tuple[dict[tuple[str, str], Path], GenderGuess]:
    if not words:
        raise ValueError("Expected detected words to contain at least one entry")
    box_to_image_map: dict[tuple[str, str], Path] = {}
    gender_guess = _normalize_gender_guess(words[0])
    if gender_guess in [GenderGuess.MALE, GenderGuess.MOSTLY_MALE]:
        logger.info("Male gender")
        output_image_path = _add_name_to_image(
            _pick_random_name(male_first_names),
            _pick_random_name(male_last_names),
            "male",
            first_name_box,
            last_name_box,
            device,
        )
    elif gender_guess in [GenderGuess.FEMALE, GenderGuess.MOSTLY_FEMALE]:
        logger.info("Female gender")
        output_image_path = _add_name_to_image(
            _pick_random_name(female_first_names),
            _pick_random_name(female_last_names),
            "female",
            first_name_box,
            last_name_box,
            device,
        )
    else:
        logger.info("Neutral or unknown gender")
        output_image_path = _add_name_to_image(
            _pick_random_name(neutral_first_names),
            _pick_random_name(neutral_last_names),
            "neutral",
            first_name_box,
            last_name_box,
            device,
        )

    box = (
        first_name_box[0],
        first_name_box[1],
        last_name_box[2],
        last_name_box[3],
    )
    box_to_image_map[(_to_box_key(box), image_path)] = output_image_path
    return box_to_image_map, gender_guess


def gender_and_handle_device_names(
    words: DetectedWords,
    box: BoundingBox,
    image_path: str,
    device: str = "olympus_cv_1500",
) -> tuple[dict[tuple[str, str], Path], GenderGuess]:
    if not words:
        raise ValueError("Expected detected words to contain at least one entry")
    box_to_image_map: dict[tuple[str, str], Path] = {}
    gender_guess = _normalize_gender_guess(words[0])
    if gender_guess in [GenderGuess.MALE, GenderGuess.MOSTLY_MALE]:
        logger.info("Male gender")
        output_image_path = _add_device_name_to_image(
            f"{_pick_random_name(male_first_names)} {_pick_random_name(male_last_names)}",
            "male",
            device,
        )
    elif gender_guess in [GenderGuess.FEMALE, GenderGuess.MOSTLY_FEMALE]:
        logger.info("Female gender")
        output_image_path = _add_device_name_to_image(
            f"{_pick_random_name(female_first_names)} {_pick_random_name(female_last_names)}",
            "female",
            device,
        )
    else:
        logger.info("Neutral or unknown gender")
        output_image_path = _add_device_name_to_image(
            f"{_pick_random_name(neutral_first_names)} {_pick_random_name(neutral_last_names)}",
            "neutral",
            device,
        )
    box_to_image_map[(_to_box_key(box), image_path)] = output_image_path
    return box_to_image_map, gender_guess
