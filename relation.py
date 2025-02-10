"""Implementation of Relation class for OCCAM variable-based reconstructability analysis"""

from typing import List, Dict, Optional
from .definitions import Relation, State, VariableList, StateSpace

class RelationImplementation:
    """Implementation class for Relation operations"""

    @staticmethod
    def validate_relation(relation: Relation) -> None:
        """Validate relation"""
        # Verify all variables are valid
        for var in relation.variables:
            if var not in relation.varlist._ivs and var != relation.varlist._dv:
                raise ValueError(f"Invalid variable {var}")
                
        # Sort variables to ensure consistent ordering like C++
        relation.variables.sort()

    @staticmethod
    def compute_df(relation: Relation, state_space: StateSpace) -> int:
        """Compute degrees of freedom matching C++ VBMManager"""
        if 'df' not in relation.attributes:
            df = 1  # Start with 1 as in C++
            for var in relation.variables:
                df *= state_space.varcardinalities[var]
            df = df - 1  # Subtract 1 for constraint
            relation.attributes['df'] = df
        return relation.attributes['df']

    @staticmethod
    def compute_entropy(relation: Relation, state_space: StateSpace) -> float:
        """Compute entropy matching C++ implementation"""
        if 'h' not in relation.attributes:
            total = sum(relation.table.values())
            if total <= 0:
                relation.attributes['h'] = 0.0
            else:
                h = 0.0
                for freq in relation.table.values():
                    if freq > 0:
                        prob = freq / total
                        h -= prob * np.log2(prob)
                relation.attributes['h'] = h
        return relation.attributes['h']

    @staticmethod
    def has_variable(relation: Relation, var: str) -> bool:
        """Check if relation contains variable"""
        return var in relation.variables
        
    @staticmethod
    def is_independent_only(relation: Relation) -> bool:
        """Check if relation only contains IVs (no DV)"""
        return not relation.has_variable(relation.varlist._dv)
        
    @staticmethod
    def get_variable_count(relation: Relation) -> int:
        """Get number of variables in relation"""
        return len(relation.variables)
        
    @staticmethod
    def get_variable(relation: Relation, index: int) -> str:
        """Get variable by index"""
        return relation.variables[index]
        
    @staticmethod
    def get_print_name(relation: Relation) -> str:
        """Get relation name in Occam format"""
        return ''.join(relation.variables)
        
    @staticmethod
    def get_long_name(relation: Relation) -> str:
        """Get full variable names, semicolon separated"""
        full_names = [relation.varlist.variables[v].name 
                     for v in relation.variables]
        return '; '.join(full_names)

    @staticmethod
    def project_state(relation: Relation, state: State) -> State:
        """Project state onto variables in this relation"""
        # Get indices of our variables in the state
        indices = []
        for var in relation.variables:
            try:
                idx = state.variables.index(var)
                indices.append(idx)
            except ValueError:
                raise ValueError(f"State missing variable {var}")
                
        # Create new state with just our variables
        new_values = tuple(state.values[i] for i in indices)
        return State(new_values, relation.varlist)
        
    @staticmethod
    def compare(relation: Relation, other: "Relation") -> int:
        """
        Compare relations matching C++ ordering.
        Returns:
            -1 if relation < other
             0 if relation == other
             1 if relation > other
        """
        # Compare by variable count first
        if len(relation.variables) < len(other.variables):
            return -1
        if len(relation.variables) > len(other.variables):
            return 1
            
        # Same length - compare variables lexicographically
        for s, o in zip(relation.variables, other.variables):
            if s < o:
                return -1
            if s > o:
                return 1
                
        return 0
        
    @staticmethod
    def is_subset(relation: Relation, other: "Relation") -> bool:
        """Check if this relation is a subset of other"""
        return all(v in other.variables for v in relation.variables)

# Add implementation methods to Relation class
def _relation_post_init(self):
    """Post-init validation"""
    RelationImplementation.validate_relation(self)
    
def _relation_compare(self, other):
    """Comparison implementation"""
    return RelationImplementation.compare(self, other)
    
def _relation_lt(self, other):
    """Less than comparison"""
    return self.compare(other) < 0
    
def _relation_eq(self, other):
    """Equality comparison"""
    return self.compare(other) == 0

def _relation_compute_df(self, state_space):
    """Compute degrees of freedom"""
    return RelationImplementation.compute_df(self, state_space)
    
def _relation_compute_entropy(self, state_space):
    """Compute entropy"""
    return RelationImplementation.compute_entropy(self, state_space)
    
def _relation_has_variable(self, var):
    """Check for variable"""
    return RelationImplementation.has_variable(self, var)
    
def _relation_is_independent_only(self):
    """Check if IV-only"""
    return RelationImplementation.is_independent_only(self)
    
def _relation_get_print_name(self):
    """Get relation name"""
    return RelationImplementation.get_print_name(self)
    
def _relation_get_long_name(self):
    """Get full variable names"""
    return RelationImplementation.get_long_name(self)
    
def _relation_project_state(self, state):
    """Project state"""
    return RelationImplementation.project_state(self, state)
    
def _relation_is_subset(self, other):
    """Check subset"""
    return RelationImplementation.is_subset(self, other)

def _relation_str(self):
    """String representation"""
    return f"Relation({','.join(self.variables)})"

# Add implementation methods to Relation class
Relation.__post_init__ = _relation_post_init
Relation.compare = _relation_compare
Relation.__lt__ = _relation_lt
Relation.__eq__ = _relation_eq
Relation.compute_df = _relation_compute_df
Relation.compute_entropy = _relation_compute_entropy
Relation.has_variable = _relation_has_variable
Relation.is_independent_only = _relation_is_independent_only
Relation.get_print_name = _relation_get_print_name
Relation.get_long_name = _relation_get_long_name
Relation.project_state = _relation_project_state
Relation.is_subset = _relation_is_subset
Relation.__str__ = _relation_str
Relation.__repr__ = _relation_str