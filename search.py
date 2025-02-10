# py_occam/search.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import time
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from .core import Model, StateSpace, VariableList

@dataclass
class Search:
    """Main search implementation matching VBMManager.cpp behavior"""
    data: pd.DataFrame
    varlist: VariableList
    state_space: StateSpace = field(init=False)
    width: int = 3
    levels: int = 7
    sort_by: str = 'information'
    search_dir: str = 'up'
    search_filter: str = 'loopless'
    ref_model: Optional[str] = None
    start_model: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, data: pd.DataFrame, yaml_path: str) -> 'Search':
        """Create Search instance from data and YAML config"""
        varlist = VariableList.from_yaml(yaml_path)
        return cls(data, varlist)
        
    def __post_init__(self):
        """Initialize StateSpace and validate parameters"""
        # Initialize state space
        self.state_space = StateSpace(self.data, self.varlist)
        
        # Validate parameters
        if self.width < 1:
            raise ValueError("Search width must be >= 1")
        if self.levels < 1:
            raise ValueError("Search levels must be >= 1")
            
        # Set ref/start models to defaults if not specified
        if not self.ref_model:
            self.ref_model = self._make_independence_model()
        if not self.start_model:
            self.start_model = self._make_independence_model()
            
    def run(self) -> List[Model]:
        """Run search according to parameters"""
        # Print header exactly matching C++ format
        self._print_header()
        
        # Initialize tracking
        start_time = time.time()
        last_time = start_time
        models = [self.start_model]  # Current level models
        all_models = [self.start_model]  # All models found
        
        # Search through levels
        for level in range(self.levels):
            print(f"\nLevel {level + 1}:", end=' ')
            
            # Generate and evaluate candidates
            candidates = []
            for model in models:
                new_models = self._generate_models(model)
                candidates.extend(new_models)
                
            if not candidates:
                print("No more candidates found")
                break
                
            # Remove duplicates
            candidates = self._remove_duplicates(candidates)
            
            # Print progress
            total_models = len(all_models) + len(candidates) 
            kept_models = len(all_models) + min(self.width, len(candidates))
            current_time = time.time()
            print(f"{len(candidates)} new models, {min(self.width, len(candidates))} kept; "
                  f"{total_models} total models, {kept_models} total kept; "
                  f"0 kb memory used; "
                  f"{current_time - last_time:.1f} seconds, {current_time - start_time:.1f} total")
            last_time = current_time
            
            # Sort and keep best candidates
            candidates = self._sort_models(candidates)
            models = candidates[:self.width]
            
            # Set model levels and IDs
            for i, model in enumerate(models, start=len(all_models)+1):
                model.level = level + 1
                model.id = i
                
            all_models.extend(models)
            
        # Print final report
        self._print_report(all_models)
        
        return all_models
        
    def _print_header(self):
        """Print search header matching C++ format exactly"""
        print("Option settings:")
        print(f"Search width: {self.width}")
        print(f"Search levels: {self.levels}")
        print(f"Search direction: {self.search_dir}")
        print(f"Sort by: {self.sort_by}")
        print(f"\nState Space Size: {self.state_space.get_state_space_size()}")
        print(f"Sample Size: {self.state_space.n}")
        if self.varlist.get_dv_abbrev():
            ivs = self.varlist.get_iv_abbrevs()
            print(f"IVs in use ({len(ivs)}) {' '.join(ivs)}")
            print(f"DV {self.varlist.get_dv_abbrev()}")
            
    def _print_report(self, models: List[Model]):
        """Print final report matching C++ format"""
        # Print model table header
        print("\nID MODEL Level H dDF dLR Alpha Inf %dH(DV) dAIC dBIC Inc.Alpha Prog. %C(Data) %cover")
        
        # Print each model's statistics
        for model in models:
            stats = model.statistics
            print(f"{model.id:2d}{'*' if model.is_loopless() else ' '} "
                  f"{model.get_name():20s} "
                  f"{model.level:2d} "
                  f"{stats['h']:8.4f} "
                  f"{stats['ddf']:4d} "
                  f"{stats['dlr']:8.4f} "
                  f"{stats['alpha']:8.4f} "
                  f"{stats['information']:8.4f} "
                  f"{stats['dh_dv']:8.4f} "
                  f"{stats['aic']:8.4f} "
                  f"{stats['bic']:8.4f} "
                  f"{model.incremental_alpha:8.4f} "
                  f"{model.progenitor_id if model.progenitor_id else 0:4d} "
                  f"{self._compute_pct_correct(model):8.4f} "
                  f"{100.0000:8.4f}")
                  
        # Print best models summary
        print("\nBest Model(s) by dBIC:")
        best_bic = max(models, key=lambda m: m.statistics['bic'])
        print(f"{best_bic.get_name()} ({best_bic.statistics['bic']:.4f})")
        
        print("\nBest Model(s) by dAIC:")
        best_aic = max(models, key=lambda m: m.statistics['aic'])
        print(f"{best_aic.get_name()} ({best_aic.statistics['aic']:.4f})")
        
        print("\nBest Model(s) by Information:")
        best_info = max(models, key=lambda m: m.statistics['information'])
        print(f"{best_info.get_name()} ({best_info.statistics['information']:.4f})")
        
        # py_occam/search.py (continued)

    def _generate_models(self, model: Model) -> List[Model]:
        """Generate next level models exactly matching VBMManager::searchOneLevel"""
        new_models = []
        
        # For each component except IV component
        for i, comp in enumerate(model.components):
            if set(comp) == set(self.varlist.get_iv_abbrevs()):
                continue
                
            # Try adding each unused variable to component
            for var in self.varlist.get_active_abbrevs():
                if var not in comp:
                    # Create new model with expanded component
                    new_comps = [c.copy() for c in model.components]
                    new_comps[i] = sorted(comp + [var])
                    
                    # Skip if not allowed by search filter
                    if self.search_filter == 'loopless':
                        if len([c for c in new_comps if len(c) > 2]) > 1:
                            continue
                            
                    # Create new model
                    new_model = Model(
                        components=new_comps,
                        variables=self.varlist.get_active_abbrevs(),
                        dv_col=self.varlist.get_dv_abbrev(),
                        state_space=self.state_space
                    )
                    
                    # Compute statistics with reference model
                    new_model.compute_model_statistics(self.ref_model)
                    
                    # Set progenitor for tracking ancestry
                    new_model.set_progenitor(model)
                    
                    new_models.append(new_model)
                    
        return new_models
        
    def _sort_models(self, models: List[Model]) -> List[Model]:
        """Sort models exactly as VBMManager.cpp does"""
        # Define sort key functions to match C++ ordering
        key_funcs = {
            'information': lambda m: (-m.statistics['information'], m.statistics['h']),
            'aic': lambda m: (-m.statistics['aic'], m.statistics['h']),
            'bic': lambda m: (-m.statistics['bic'], m.statistics['h']),
            'h': lambda m: (m.statistics['h'], m.get_name())
        }
        
        # Get sort key function
        key_func = key_funcs.get(self.sort_by, key_funcs['h'])
        
        # Sort using stable sort
        return sorted(models, key=key_func)
        
    def _remove_duplicates(self, models: List[Model]) -> List[Model]:
        """Remove duplicate models matching VBMManager behavior"""
        unique_models = []
        seen = set()
        
        for model in models:
            # Create model signature from sorted components
            sig = tuple(sorted(
                tuple(sorted(comp)) for comp in model.components
            ))
            
            if sig not in seen:
                seen.add(sig)
                unique_models.append(model)
                
        return unique_models
        
    def _compute_pct_correct(self, model: Model) -> float:
        """Compute percent correct predictions matching VBMManager"""
        q = model.get_fitted_probs()
        total_correct = 0
        total_cases = 0
        
        # Group states by IV values
        iv_groups = defaultdict(list)
        for state in q:
            iv_state = state.project(self.varlist.get_iv_abbrevs())
            iv_groups[iv_state].append(state)
            
        # For each IV state
        for iv_state in iv_groups:
            states = iv_groups[iv_state]
            freq = sum(self.state_space.state_frequencies[s] for s in states)
            
            if freq > 0:
                # Get predicted DV value (highest probability)
                dv_probs = defaultdict(float)
                for s in states:
                    dv_val = s.get_variable_values([self.varlist.get_dv_abbrev()])[0]
                    dv_probs[dv_val] += q[s]
                    
                # Get prediction and count correct cases
                pred_dv = max(dv_probs.items(), key=lambda x: x[1])[0]
                correct = sum(self.state_space.state_frequencies[s] 
                            for s in states 
                            if s.get_variable_values([self.varlist.get_dv_abbrev()])[0] == pred_dv)
                            
                total_correct += correct
                total_cases += freq
                
        return (total_correct / total_cases * 100) if total_cases > 0 else 0.0
        
    def _make_independence_model(self) -> Model:
        """Create independence model (IV:Z)"""
        # Get IV and DV components
        iv_comp = sorted(self.varlist.get_iv_abbrevs())
        dv_comp = [self.varlist.get_dv_abbrev()]
        
        # Create model
        model = Model(
            components=[iv_comp, dv_comp],
            variables=self.varlist.get_active_abbrevs(),
            dv_col=self.varlist.get_dv_abbrev(),
            state_space=self.state_space
        )
        
        # Initialize model level and ID
        model.level = 0
        model.id = 1
        
        return model