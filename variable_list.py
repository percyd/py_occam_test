from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

@dataclass
class VariableDefinition:
    """Represents a single variable definition from YAML"""
    name: str            # Full name (e.g., "APOE") - only for display
    abbrev: str         # Abbreviation (e.g., "Ap") - used internally
    cardinality: int    # Number of possible values
    type: int           # 0=ignore, 1=IV, 2=DV
    rebin: Optional[str] = None  # Optional rebinning specification

    def __post_init__(self):
        """Validate variable definition"""
        # Validate type
        if self.type not in [0, 1, 2]:
            raise ValueError(f"Invalid variable type {self.type}")
            
        # Validate cardinality
        if self.cardinality < 1:
            raise ValueError(f"Invalid cardinality {self.cardinality}")
            
        # Validate abbreviation format
        if not self.abbrev.isalpha():
            raise ValueError("Abbreviation must contain only letters")

@dataclass
class VariableList:
    """Handles variable definitions and mappings"""
    variables: Dict[str, VariableDefinition] = field(default_factory=dict)
    _dv: Optional[str] = field(init=False, default=None)
    _ivs: List[str] = field(init=False, default_factory=list)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'VariableList':
        """Create VariableList from YAML variable definitions file"""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
            
        varlist = cls()
        for name, info in config['variables'].items():
            vardef = VariableDefinition(
                name=name,
                abbrev=info['abbrev'],
                cardinality=info['cardinality'],
                type=info['type'],
                rebin=info.get('rebin')
            )
            varlist.variables[vardef.abbrev] = vardef
            
            if vardef.type == 2:  # DV
                if varlist._dv is not None:
                    raise ValueError("Multiple DVs not supported")
                varlist._dv = vardef.abbrev
            elif vardef.type == 1:  # IV
                varlist._ivs.append(vardef.abbrev)
                
        return varlist

    def get_active_abbrevs(self) -> List[str]:
        """Get abbreviations of all active (non-ignored) variables"""
        return [abbrev for abbrev, var in self.variables.items() 
                if var.type != 0]  # type 0 means ignored
                
    def get_dv_abbrev(self) -> str:
        """Get abbreviation of DV"""
        if self._dv is None:
            raise ValueError("No DV found")
        return self._dv
        
    def get_iv_abbrevs(self) -> List[str]:
        """Get abbreviations of all IVs"""
        return self._ivs.copy()
        
    def get_full_name(self, abbrev: str) -> str:
        """Get full name for display purposes"""
        return self.variables[abbrev].name

    def verify_model_spec(self, model_spec: str):
        """Verify that a model specification is valid"""
        if not model_spec:
            raise ValueError("Empty model specification")
            
        # Split into components
        components = model_spec.split(':')
        
        # Track variables used
        used_vars = set()
        
        for comp in components:
            if comp == 'IV':
                # IV component should contain all IVs
                used_vars.update(self.get_iv_abbrevs())
            else:
                # Parse component into variables
                vars = self._parse_component(comp)
                used_vars.update(vars)
                
                # Verify DV is present
                if self.get_dv_abbrev() not in vars:
                    raise ValueError(f"Component {comp} missing DV")
                    
        # Verify all active variables are used
        all_vars = set(self.get_active_abbrevs())
        if not used_vars == all_vars:
            missing = all_vars - used_vars
            raise ValueError(f"Model missing variables: {missing}")

    def _parse_component(self, comp: str) -> List[str]:
        """Parse component string into list of variable abbreviations"""
        vars = []
        current = []
        
        for c in comp:
            if c == 'Z':  # DV
                if current:
                    vars.append(''.join(current))
                    current = []
                vars.append(self.get_dv_abbrev())
            else:
                current.append(c)
                
        if current:
            vars.append(''.join(current))
            
        return vars

    def add_variable(self, vardef: VariableDefinition):
        """Add a variable definition"""
        # Check for duplicate abbreviations
        if vardef.abbrev in self.variables:
            raise ValueError(f"Duplicate abbreviation {vardef.abbrev}")
            
        # Add to variables dict
        self.variables[vardef.abbrev] = vardef
        
        # Update DV/IV tracking 
        if vardef.type == 2:  # DV
            if self._dv is not None:
                raise ValueError("Multiple DVs not supported")
            self._dv = vardef.abbrev
        elif vardef.type == 1:  # IV
            self._ivs.append(vardef.abbrev)