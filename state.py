"""Implementation of State class for OCCAM variable-based reconstructability analysis"""

from typing import List, Dict, Tuple, Optional, Union
from .definitions import State, VariableList

class State:
    """Implementation class for State operations"""
    
    @staticmethod 
    def validate_state(state: State) -> None:
        """Validate state against variable list"""
        if not isinstance(state.varlist, VariableList):
            raise TypeError(f"varlist must be VariableList instance, got {type(state.varlist)}")
            
        # Initialize _active_vars based on subset_vars or all active vars
        active_vars = []
        if state.subset_vars:
            # Verify all subset vars are valid
            for var in state.subset_vars:
                if var not in state.varlist.variables:
                    raise ValueError(f"Invalid variable {var} in subset_vars")
            active_vars = state.subset_vars
        else:
            active_vars = state.varlist.get_active_abbrevs()
            
        # Set _active_vars (use object.__setattr__ since State is frozen)
        object.__setattr__(state, '_active_vars', active_vars)
                          
        if len(state.values) != len(state._active_vars):
            raise ValueError(
                f"Values count ({len(state.values)}) must match variables ({len(state._active_vars)})"
            )

    @staticmethod
    def hash_state(state: State) -> int:
        """Hash based on values and variable names"""
        return hash((state.values, tuple(state._active_vars)))
        
    @staticmethod
    def equals(state: State, other: State) -> bool:
        """Equality check for dictionary keys"""
        if not isinstance(other, State):
            return False
        return (state.values == other.values and 
                state._active_vars == other._active_vars)
    
    @staticmethod
    def less_than(state: State, other: State) -> bool:
        """Less than comparison for sorting"""
        if not isinstance(other, State):
            return NotImplemented
            
        # Compare values first
        for v1, v2 in zip(state.values, other.values):
            # Handle missing values ('.')
            if v1 == '.' and v2 != '.':
                return True
            if v2 == '.' and v1 != '.':
                return False
            if v1 != v2:
                # Convert to strings for comparison if needed
                s1 = str(v1)
                s2 = str(v2)
                return s1 < s2
                
        # If values are equal, compare by variables
        for v1, v2 in zip(state._active_vars, other._active_vars):
            if v1 != v2:
                return v1 < v2
                
        # If everything is equal, default to False
        return False
        
    @staticmethod
    def project(state: State, vars: List[str]) -> State:
        """Project state onto subset of variables"""
        # Verify all projection variables are valid
        proj_vars = [v for v in vars if v in state._active_vars]
        if not proj_vars:
            raise ValueError(f"No valid variables to project onto: {vars}")
            
        # Get values for projection variables
        proj_values = []
        for var in proj_vars:
            if var in state._active_vars:
                idx = state._active_vars.index(var)
                proj_values.append(state.values[idx])
                
        # Create new state with projected variables
        return State(
            values=tuple(proj_values),
            varlist=state.varlist,
            subset_vars=proj_vars
        )
        
    @staticmethod
    def get_variable_values(state: State, vars: List[str]) -> List[Union[int, str]]:
        """Get values for specified variables"""
        return [state.values[state._active_vars.index(v)] 
                for v in vars if v in state._active_vars]

    @staticmethod
    def format_state(state: State) -> str:
        """Format state as string"""
        var_strs = []
        for var, val in zip(state._active_vars, state.values):
            var_strs.append(f"{var}={val}")
        return ", ".join(var_strs)

# Add implementation methods to State class
def _state_post_init(self):
    """Post-init validation"""
    StateImplementation.validate_state(self)
    
def _state_hash(self):
    """Hash implementation"""
    return StateImplementation.hash_state(self)
    
def _state_eq(self, other):
    """Equality implementation""" 
    return StateImplementation.equals(self, other)
    
def _state_lt(self, other):
    """Less than implementation"""
    return StateImplementation.less_than(self, other)
    
def _state_project(self, vars: List[str]):
    """Project implementation"""
    return StateImplementation.project(self, vars)
    
def _state_get_values(self, vars: List[str]):
    """Get values implementation"""
    return StateImplementation.get_variable_values(self, vars)
    
def _state_format(self):
    """String representation"""
    return StateImplementation.format_state(self)

# Add implementation methods to State class
State.__post_init__ = _state_post_init
State.__hash__ = _state_hash
State.__eq__ = _state_eq
State.__lt__ = _state_lt
State.project = _state_project
State.get_variable_values = _state_get_values
State.__str__ = _state_format
State.__repr__ = _state_format