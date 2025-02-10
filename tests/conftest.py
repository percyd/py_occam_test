"""Test configuration and fixtures"""
from pathlib import Path
import pytest
import pandas as pd

@pytest.fixture
def test_data_dir():
    """Get path to test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture 
def dementia_data(test_data_dir):
    """Load dementia test data"""
    data = pd.read_csv(test_data_dir / "dementia05ApEdC_no_occam_header.tsv", sep="\t")
    return data[["APOE", "EDU", "C", "CaseCon"]]

@pytest.fixture
def dementia_yaml(test_data_dir):
    """Get path to dementia YAML config"""
    return test_data_dir / "dementia05ApEdC_no_occam_header.yml"