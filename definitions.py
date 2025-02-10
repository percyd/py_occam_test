"""Core dataclass definitions for OCCAM variable-based reconstructability analysis"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Union
import numpy as np

@dataclass
class VariableDefinition:
    """Represents a single variable definition from YAML config"""
    name: str            # Full name (e.g., "APOE") - for display 
    abbrev: str         # Abbreviation (e.g., "Ap") - used internally
    cardinality: int    # Number of possible values
    type: int           # 0=ignore, 1=IV, 2=DV 
    rebin: Optional[str] = None  # Optional rebinning specification

@dataclass
class VariableList:
    """Manages variable definitions and mappings"""
    variables: Dict[str, VariableDefinition] = field(default_factory=dict)
    _dv: Optional[str] = field(init=False, default=None)  # DV abbreviation
    _ivs: List[str] = field(init=False, default_factory=list)  # IV abbreviations

@dataclass(frozen=True)
class State:
    """Immutable state tuple with variable values"""
    values: Tuple[Union[int, str], ...]  # Allow ints and "." for missing
    varlist: "VariableList"
    subset_vars: Optional[List[str]] = None  # For projected states
    _active_vars: List[str] = field(init=False, default_factory=list)

@dataclass
class StateSpace:
    """Manages state frequencies and calculations"""
    data: "pd.DataFrame"  # reference to avoid circular import
    varlist: VariableList
    dv_col: str  # DV abbreviation  
    iv_cols: List[str] = field(init=False)  # IV abbreviations
    n: int = field(init=False)  # Sample size
    keysize: int = field(init=False)  # Number of variables
    state_frequencies: Dict[State, float] = field(init=False)  # Frequency table
    cache_projections: Dict[Tuple[str, ...], Dict[State, float]] = field(default_factory=dict)
    varcardinalities: Dict[str, int] = field(init=False)  # Variable cardinalities

@dataclass
class Relation:
    """Represents a model relation/component matching C++ implementation"""
    variables: List[str]  # Variable abbreviations in relation
    varlist: VariableList
    attributes: Dict[str, float] = field(default_factory=dict)  # Cached statistics
    table: Dict[State, float] = field(default_factory=dict)  # Frequency table

@dataclass 
class Model:
    """Represents a reconstructability model built from relations"""
    components: List[List[str]]  # Lists of variable abbreviations 
    variables: List[str]        # All variable abbreviations
    dv_col: str                # DV abbreviation
    state_space: StateSpace    # Reference to state space
    statistics: Dict[str, float] = field(default_factory=dict)  # Model statistics
    id: Optional[int] = None  # Model ID in search
    level: Optional[int] = None  # Model level in search
    progenitor: Optional["Model"] = None  # Parent model in search
    incremental_alpha: float = 0.0  # Incremental significance
    progenitor_id: Optional[int] = None  # Parent model ID

@dataclass
class Occam:
    """Main OCCAM implementation integrating all components"""
    data: "pd.DataFrame"  # reference to avoid circular import
    varlist: VariableList
    state_space: StateSpace = field(init=False)  # Created in post_init