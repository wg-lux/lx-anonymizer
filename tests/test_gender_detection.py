"""
Tests for the gender detection functionality.
"""
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lx_anonymizer.determine_gender import determine_gender

def test_determine_gender_male():
    """Test gender detection for male names."""
    male_names = ["Thomas", "Michael", "Andreas", "Peter", "Stefan"]
    for name in male_names:
        gender = determine_gender(name)
        assert gender == "male", f"Expected 'male' for {name}, got {gender}"

def test_determine_gender_female():
    """Test gender detection for female names."""
    female_names = ["Maria", "Anna", "Julia", "Sabine", "Lisa"]
    for name in female_names:
        gender = determine_gender(name)
        assert gender == "female", f"Expected 'female' for {name}, got {gender}"

def test_determine_gender_unknown():
    """Test gender detection for unknown or gender-neutral names."""
    unknown_names = ["", None, "XXXX", "123"]
    for name in unknown_names:
        gender = determine_gender(name)
        assert gender == "unknown", f"Expected 'unknown' for {name}, got {gender}"
