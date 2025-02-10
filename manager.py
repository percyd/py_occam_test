"""Main OCCAM manager implementation matching VBMManager.cpp"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from .definitions import Occam, Model, VariableList, VariableDefinition

class OccamImplementation:
    """Main OCCAM implementation class"""
    
    @staticmethod
    def initialize(occam: Occam) -> None:
        """Initialize from data and variables"""
        # Convert columns to use abbreviations  
        name_to_abbrev = {var.name: var.abbrev 
                         for var in occam.varlist.variables.values()}
        occam.data = occam.data.rename(columns=name_to_abbrev)
        
        # Get all active variables using abbreviations
        occam.all_vars = occam.varlist.get_active_abbrevs()
        occam.dv = occam.varlist.get_dv_abbrev()
        occam.ivs = occam.varlist.get_iv_abbrevs()
        
        # Filter to active columns
        occam.data = occam.data[occam.all_vars]
        
        # Initialize state space
        occam.state_space = StateSpace(
            occam.data,
            occam.varlist,
            occam.dv
        )

    @staticmethod
    def from_files(data_file: Path, yaml_file: Path) -> "Occam":
        """Create Occam instance from data and YAML files"""
        # Load data
        data = pd.read_csv(data_file, sep='\t')
        
        # Load variable definitions
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        
        # Create variable list
        varlist = VariableList()
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
                
        return Occam(data, varlist)

    @staticmethod
    def make_model(occam: Occam, model_name: str) -> Model:
        """Create model from string specification matching C++ exactly"""
        # Parse components
        components = []
        model_parts = model_name.split(':')
        
        # Always put IV component first for directed systems
        if occam.dv:  # If we have a DV it's directed
            iv_comp = sorted(occam.ivs)  # Sort IV names alphabetically
            components.append(iv_comp)
            
            # Handle remaining components
            for part in model_parts:
                if part != 'IV':  # Skip IV part since we added it
                    comp = occam.varlist._parse_component(part)
                    if set(comp) != set(occam.ivs):  # Skip if this is just IVs
                        components.append(sorted(comp))  # Sort variables in component
        else:
            # For neutral systems, just parse all components
            for part in model_parts:
                comp = occam.varlist._parse_component(part)
                components.append(sorted(comp))
                
        # Sort remaining components in same order as C++
        if len(components) > 1:
            components[1:] = sorted(components[1:], 
                                  key=lambda x: (''.join(x), len(x)))
            
        # Create model
        return Model(
            components=components,
            variables=occam.all_vars,
            dv_col=occam.dv,
            state_space=occam.state_space
        )

    @staticmethod
    def make_independence_model(occam: Occam) -> Model:
        """Create independence model (IV:Z)"""
        return occam.make_model("IV:Z")

    @staticmethod
    def generate_parents(occam: Occam, model: Model) -> List[Model]:
        """Generate parent models one level up"""
        parents = []
        
        # Get reference model for statistics computation 
        if not hasattr(occam, '_ref_model'):
            occam._ref_model = occam.make_independence_model()
        
        # For each component except IV
        for i, comp in enumerate(model.components):
            if set(comp) == set(occam.ivs):
                continue
                
            # Try adding each unused variable
            for var in occam.all_vars:
                if var not in comp:
                    new_comps = [c.copy() for c in model.components]
                    new_comps[i] = sorted(comp + [var])
                    
                    parent = Model(
                        new_comps,
                        occam.all_vars,
                        occam.dv,
                        occam.state_space
                    )
                    
                    # Compute statistics before setting progenitor
                    parent.compute_model_statistics(occam._ref_model)
                    parent.set_progenitor(model)
                    
                    parents.append(parent)
                    
        return parents

    @staticmethod
    def remove_duplicate_models(models: List[Model]) -> List[Model]:
        """Remove duplicate models"""
        unique_models = []
        seen = set()  # Track model signatures
        
        for model in models:
            # Create signature from sorted components
            signature = tuple(sorted(
                tuple(sorted(comp)) for comp in model.components
            ))
            
            # Only keep if we haven't seen this signature
            if signature not in seen:
                seen.add(signature)
                unique_models.append(model)
                
        return unique_models

    @staticmethod
    def sort_models(models: List[Model], criterion: str) -> List[Model]:
        """Sort models exactly as C++ VBMManager does"""
        # Define key functions to match C++ ordering
        key_funcs = {
            'h': lambda m: (m.statistics['h'], m.get_name()),
            'information': lambda m: (-m.statistics['information'], 
                                   m.statistics['h'], m.get_name()),
            'aic': lambda m: (-m.statistics['aic'], 
                            m.statistics['h'], m.get_name()),
            'bic': lambda m: (-m.statistics['bic'], 
                            m.statistics['h'], m.get_name())
        }
        
        # Get sort key function
        key_func = key_funcs.get(criterion, key_funcs['h'])
        
        # Sort using stable sort
        return sorted(models, key=key_func)

    @staticmethod
    def compute_pct_correct(occam: Occam, model: Model) -> float:
        """Compute percent correct predictions for a model"""
        q = model.get_fitted_probs()
        total_correct = 0
        total_cases = 0
        
        # Group states by IV values
        iv_groups = defaultdict(list)
        for state in q.keys():
            iv_state = state.project(occam.ivs)
            iv_groups[iv_state].append(state)
            
        # For each IV state
        for iv_state in iv_groups:
            states = iv_groups[iv_state]
            freq = sum(occam.state_space.state_frequencies[s] 
                      for s in states)
            
            if freq > 0:
                # Get predicted DV value (highest probability)
                dv_probs = defaultdict(float)
                for s in states:
                    dv_val = s.get_variable_values([occam.dv])[0]
                    dv_probs[dv_val] += q[s]
                pred_dv = max(dv_probs.items(), key=lambda x: x[1])[0]
                
                # Count correct predictions
                correct = sum(occam.state_space.state_frequencies[s] 
                            for s in states 
                            if s.get_variable_values([occam.dv])[0] == pred_dv)
                
                total_correct += correct
                total_cases += freq
                
        return (total_correct / total_cases * 100) if total_cases > 0 else 0

# Add implementation methods to Occam class
def _occam_post_init(self):
    """Post-init initialization"""
    OccamImplementation.initialize(self)

def _occam_make_model(self, model_name):
    """Create model from specification"""
    return OccamImplementation.make_model(self, model_name)

def _occam_make_independence_model(self):
    """Create independence model"""
    return OccamImplementation.make_independence_model(self)

def _occam_generate_parents(self, model):
    """Generate parent models"""
    return OccamImplementation.generate_parents(self, model)

def _occam_remove_duplicate_models(self, models):
    """Remove duplicate models"""
    return OccamImplementation.remove_duplicate_models(models)

def _occam_sort_models(self, models, criterion):
    """Sort models"""
    return OccamImplementation.sort_models(models, criterion)

def _occam_compute_pct_correct(self, model):
    """Compute percent correct predictions"""
    return OccamImplementation.compute_pct_correct(self, model)

@classmethod
def _occam_from_files(cls, data_file, yaml_file):
    """Create from files"""
    return OccamImplementation.from_files(data_file, yaml_file)

# Add implementation methods to Occam class
Occam.__post_init__ = _occam_post_init
Occam.make_model = _occam_make_model
Occam.make_independence_model = _occam_make_independence_model
Occam.generate_parents = _occam_generate_parents
Occam.remove_duplicate_models = _occam_remove_duplicate_models
Occam.sort_models = _occam_sort_models
Occam.compute_pct_correct = _occam_compute_pct_correct
Occam.from_files = _occam_from_files