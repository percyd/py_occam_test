"""Implementation of StateSpace class for OCCAM variable-based reconstructability analysis"""

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from collections import defaultdict
from .definitions import StateSpace, State, VariableList
from .relations import Relation  # Add this import

class StateSpaceImplementation:
    """Implementation class for StateSpace operations"""
    
    @staticmethod
    def initialize(state_space: StateSpace) -> None:
        """Initialize derived attributes and frequency tables"""
        # Get all active variables using abbreviations
        state_space.variables = state_space.varlist.get_active_abbrevs()
        state_space.iv_cols = state_space.varlist.get_iv_abbrevs()
        state_space.keysize = len(state_space.variables)
        state_space.n = len(state_space.data)
        
        # Calculate variable cardinalities
        state_space.varcardinalities = {
            var: state_space.varlist.variables[var].cardinality
            for var in state_space.variables
        }
        
        # Initialize frequency tables
        state_space.state_frequencies = StateSpaceImplementation._compute_frequencies(state_space)

    @staticmethod
    def _compute_frequencies(state_space: StateSpace) -> Dict[State, float]:
        """Compute frequencies exactly as C++ VBMManager does"""
        freqs = defaultdict(float)
        
        # Match C++ handling of missing values
        for _, row in state_space.data.iterrows():
            values = []
            for var in state_space.variables:
                val = row[var]
                # Match C++ missing value handling precisely
                values.append(int(val) if pd.notna(val) else '.')
                
            state = State(
                values=tuple(values),
                varlist=state_space.varlist 
            )            
            freqs[state] += 1.0  # Always increment by 1.0 as C++ does
            
        return dict(freqs)

    @staticmethod
    def project_frequencies(state_space: StateSpace,
                          states: Dict[State, float], 
                          variables: List[str]) -> Dict[State, float]:
        """
        Project frequencies onto a subset of variables.
        Caches results for efficiency like C++ VBMManager.
        """
        # Check cache first
        cache_key = tuple(sorted(variables))
        if cache_key in state_space.cache_projections:
            return state_space.cache_projections[cache_key]
            
        # Compute projection
        projected = defaultdict(float)
        for state, freq in states.items():
            proj_state = state.project(variables)
            projected[proj_state] += freq
            
        # Cache result
        result = dict(projected)
        state_space.cache_projections[cache_key] = result
        return result

    @staticmethod
    def compute_entropy(state_space: StateSpace,
                       freqs: Dict[State, float], 
                       base: float = 2.0) -> float:
        """
        Compute entropy of a frequency distribution.
        Matches C++ VBMManager entropy calculation.
        """
        total = sum(freqs.values())
        if total <= 0:
            return 0.0
            
        entropy = 0.0
        for freq in freqs.values():
            if freq > 0:
                prob = freq / total
                entropy -= prob * np.log(prob) / np.log(base)
                
        return entropy

    @staticmethod
    def compute_conditional_entropy(state_space: StateSpace,
                                  joint_freqs: Dict[State, float],
                                  conditioning_vars: List[str]) -> float:
        """
        Compute conditional entropy H(Y|X).
        Matches C++ VBMManager conditional entropy calculation.
        """
        # Project to get marginal distribution
        marg_freqs = state_space.project_frequencies(joint_freqs, conditioning_vars)
        
        h_cond = 0.0
        total = sum(joint_freqs.values())
        
        # For each state of conditioning variables
        for marg_state, marg_freq in marg_freqs.items():
            # Skip if marginal probability is 0
            if marg_freq <= 0:
                continue
                
            # Get conditional distribution
            cond_dist = {}
            marg_prob = marg_freq / total
            
            for joint_state, freq in joint_freqs.items():
                if joint_state.project(conditioning_vars) == marg_state:
                    cond_dist[joint_state] = freq / marg_freq
                    
            # Add weighted conditional entropy
            if cond_dist:
                h_cond += marg_prob * state_space.compute_entropy(cond_dist)
                
        return h_cond

    @staticmethod
    def compute_reference_entropies(state_space: StateSpace) -> Tuple[float, float, float]:
        """
        Compute reference entropies (top, bottom, independent).
        Match C++ VBMManager.cpp computeH() method exactly.
        """
        # Get top model entropy (data)
        all_vars = sorted(state_space.variables)  # Sort to match C++ order
        top_freqs = state_space.state_frequencies
        top_h = state_space.compute_entropy(top_freqs)
        
        # Get independent relation entropy
        iv_freqs = state_space.project_frequencies(state_space.state_frequencies, 
                                                 state_space.iv_cols)
        iv_h = state_space.compute_entropy(iv_freqs)
        
        # Get bottom model entropy
        # Project to get single variable distributions
        bottom_h = 0.0
        for var in state_space.variables:
            var_freqs = state_space.project_frequencies(state_space.state_frequencies, [var])
            bottom_h += state_space.compute_entropy(var_freqs)
        
        return top_h, bottom_h, iv_h

    @staticmethod 
    def get_state_space_size(state_space: StateSpace) -> int:
        """Calculate total size of state space"""
        size = 1
        for card in state_space.varcardinalities.values():
            size *= card
        return size

    @staticmethod
    def compute_df(state_space: StateSpace,
                  relation_vars: List[str]) -> int:
        """
        Compute degrees of freedom for a relation.
        Matches C++ VBMManager df calculation.
        """
        df = 1
        for var in relation_vars:
            df *= state_space.varcardinalities[var]
        return df - 1

# Add implementation methods to StateSpace class
def _state_space_post_init(self):
    """Post-init initialization"""
    StateSpaceImplementation.initialize(self)

def _compute_entropy(self, freqs, base=2.0):
    """Entropy calculation"""
    return StateSpaceImplementation.compute_entropy(self, freqs, base)

def _project_frequencies(self, states, variables):
    """Frequency projection"""
    return StateSpaceImplementation.project_frequencies(self, states, variables)

def _compute_conditional_entropy(self, joint_freqs, conditioning_vars):
    """Conditional entropy calculation"""
    return StateSpaceImplementation.compute_conditional_entropy(
        self, joint_freqs, conditioning_vars)

def _compute_reference_entropies(self):
    """Reference entropy calculations"""
    return StateSpaceImplementation.compute_reference_entropies(self)

def _get_state_space_size(self):
    """Get state space size"""
    return StateSpaceImplementation.get_state_space_size(self)

def _compute_df(self, relation_vars):
    """Compute degrees of freedom"""
    return StateSpaceImplementation.compute_df(self, relation_vars)

# Add implementation methods to StateSpace class
StateSpace.__post_init__ = _state_space_post_init
StateSpace.compute_entropy = _compute_entropy
StateSpace.project_frequencies = _project_frequencies 
StateSpace.compute_conditional_entropy = _compute_conditional_entropy
StateSpace.compute_reference_entropies = _compute_reference_entropies
StateSpace.get_state_space_size = _get_state_space_size
StateSpace.compute_df = _compute_df