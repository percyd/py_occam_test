# py_occam/fit.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path

from .core import Model, StateSpace, VariableList

@dataclass
class Fit:
    """Detailed model fitting matching VBMManager.cpp behavior"""
    data: pd.DataFrame
    varlist: VariableList
    state_space: StateSpace = field(init=False)
    model_name: str
    default_model: Optional[str] = None
    skip_ipf: bool = False
    skip_trained_table: bool = False
    
    def __post_init__(self):
        """Initialize state space and validate model spec"""
        # Initialize state space
        self.state_space = StateSpace(self.data, self.varlist)
        
        # Validate model specification
        self.varlist.verify_model_spec(self.model_name)
        
        # Create reference models
        self.ref_top = self._make_top_model()
        self.ref_bottom = self._make_independence_model()
        
    def run(self) -> Dict:
        """Run fit analysis matching VBMManager fit output exactly"""
        # Print header
        self._print_header()
        
        # Create model to fit
        model = self._make_model(self.model_name)
        
        # Print model structure 
        print(f"\nModel {self.model_name} (Directed System)")
        self._print_model_structure(model)
        
        # Compute and print basic statistics
        stats = self._compute_basic_stats(model)
        self._print_basic_stats(stats)
        
        # Get fitted probabilities
        if not self.skip_ipf:
            q = model.get_fitted_probs()
        
        # Print reference comparisons
        print("\nREFERENCE = TOP")
        self._print_ref_comparison(model, self.ref_top)
        
        print("\nREFERENCE = BOTTOM")
        self._print_ref_comparison(model, self.ref_bottom)
        
        # Print conditional probability tables
        if not self.skip_trained_table:
            self._print_conditional_tables(model, q)
            
        # For each predicting component, print component tables
        if len(model.components) > 2:  # More than IV and one predicting component
            for comp in model.components[1:]:  # Skip IV component
                print(f"\nConditional DV (D) (%) for each IV composite state for the Relation {''.join(comp)}.")
                print("(For component relations, the Data and Model parts of the table are equal, so only one is given)")
                self._print_component_table(comp)
                
        return {
            'model': model,
            'statistics': stats,
            'q': q if not self.skip_ipf else None
        }
        
    def _print_header(self):
        """Print fit header matching C++ format"""
        print("Option settings:")
        print("\nData Lines Read", len(self.data))
        print("State Space Size", self.state_space.get_state_space_size())
        print("Sample Size", self.state_space.n)
        if self.varlist.get_dv_abbrev():
            ivs = self.varlist.get_iv_abbrevs()
            print(f"IVs in use ({len(ivs)}) {' '.join(ivs)}")
            print(f"DV {self.varlist.get_dv_abbrev()}")
            
    def _print_model_structure(self, model: Model):
        """Print model structure exactly as VBMManager does"""
        # Print IV component
        iv_comp = model.components[0]
        print(f"IV Component: {'; '.join(iv_comp)} ({'+'.join(iv_comp)})")
        
        # Print predicting components
        for comp in model.components[1:]:
            print(f"Model Component: {'; '.join(comp)} ({'+'.join(comp)})")
            
    def _compute_basic_stats(self, model: Model) -> Dict:
        """Compute basic model statistics matching VBMManager"""
        stats = {}
        
        # Compute degrees of freedom
        stats['df'] = model.compute_df()
        
        # Check if model has loops
        stats['loops'] = not model.is_loopless()
        
        # Get entropy and information
        model.compute_model_statistics(self.ref_bottom)
        stats['h'] = model.statistics['h']
        stats['information'] = model.statistics['information']
        
        # Compute transmission
        stats['t'] = abs(model.statistics['dlr'] / 
                        (2 * self.state_space.n * np.log(2)))
                        
        return stats
        
    def _print_basic_stats(self, stats: Dict):
        """Print basic stats matching VBMManager format"""
        print(f"\nDegrees of Freedom (DF): {stats['df']}")
        print(f"Loops: {'YES' if stats['loops'] else 'NO'}")
        print(f"Entropy(H): {stats['h']:.6f}")
        print(f"Information captured (%): {stats['information']*100:.4f}")
        print(f"Transmission (T): {stats['t']:.6f}")
        
    def _print_ref_comparison(self, model: Model, ref_model: Model):
        """Print reference model comparison table"""
        model.compute_model_statistics(ref_model)
        print("Value Prob. (Alpha)")
        print(f"Log-Likelihood (LR) {model.statistics['dlr']:8.4f} "
              f"{model.statistics['alpha']:8.4f}")
        print(f"Delta DF (dDF) {model.statistics['ddf']}")
        
    def _print_conditional_tables(self, model: Model, q: Dict):
        """Print conditional probability tables matching VBMManager"""
        print("\nConditional DV (D) (%) for each IV composite state")
        print(f"IV order: {', '.join(model.components[0])} ({', '.join(self.varlist.get_iv_abbrevs())})")
        
        # Print header
        print("IV | Data | Model")
        print("| obs. p(DV|IV) | calc. q(DV|IV)")
        
        # Group states by IV values
        iv_groups = self._group_by_ivs(q)
        
        total_correct = 0
        total_cases = 0
        
        # For each IV state
        for iv_state in sorted(iv_groups.keys()):
            states = iv_groups[iv_state]
            self._print_conditional_row(iv_state, states, q, 
                                     total_correct, total_cases)
            
        # Print summary row
        self._print_conditional_summary(total_correct, total_cases)
        
    def _group_by_ivs(self, q: Dict) -> Dict:
        """Group states by IV values"""
        groups = defaultdict(list)
        for state in q:
            iv_state = state.project(self.varlist.get_iv_abbrevs())
            groups[iv_state].append(state)
        return groups
        
    def _print_conditional_row(self, iv_state, states: List, q: Dict,
                             total_correct: int, total_cases: int):
        """Print single row of conditional table"""
        # Get frequencies and probabilities
        freq = sum(self.state_space.state_frequencies[s] for s in states)
        
        # Calculate observed DV probabilities
        obs_probs = self._calc_observed_probs(states, freq)
        
        # Calculate model DV probabilities
        model_probs = self._calc_model_probs(states, q)
        
        # Get prediction rule and accuracy
        pred_dv, correct = self._get_prediction(states, model_probs, freq)
        
        # Update totals
        total_correct += correct
        total_cases += freq
        
        # Print row
        iv_vals = ' '.join(str(v) for v in iv_state.values)
        print(f"{iv_vals:10s} | {freq:6.3f} {obs_probs[0]:6.3f} {obs_probs[1]:6.3f} | "
              f"{model_probs[0]:6.3f} {model_probs[1]:6.3f} {pred_dv:4d} "
              f"{correct:8.0f} {(correct/freq*100 if freq>0 else 0):8.2f}")
              
    def _print_conditional_summary(self, total_correct: int, total_cases: int):
        """Print summary row of conditional table"""
        total_pct = (total_correct / total_cases * 100) if total_cases > 0 else 0
        print(f"{'TOTAL':10s} | {total_cases:6.0f} | | | | {total_pct:8.2f}")
        print("| freq Z=0 Z=1| Z=0 Z=1 rule #correct %correct")
        
    def _print_component_table(self, comp: List[str]):
        """Print component relation table"""
        # Create component model
        comp_model = Model(
            components=[self.varlist.get_iv_abbrevs(), comp],
            variables=self.varlist.get_active_abbrevs(),
            dv_col=self.varlist.get_dv_abbrev(),
            state_space=self.state_space
        )
        
        # Print table
        q = comp_model.get_fitted_probs()
        self._print_conditional_tables(comp_model, q)
        
    def _make_model(self, model_name: str) -> Model:
        """Create model from specification"""
        # Parse components
        components = []
        model_parts = model_name.split(':')
        
        # Always put IV component first
        iv_comp = sorted(self.varlist.get_iv_abbrevs())
        components.append(iv_comp)
        
        # Add other components
        for part in model_parts:
            if part != 'IV':
                comp = self.varlist._parse_component(part)
                if set(comp) != set(iv_comp):
                    components.append(sorted(comp))
                    
        return Model(
            components=components,
            variables=self.varlist.get_active_abbrevs(),
            dv_col=self.varlist.get_dv_abbrev(),
            state_space=self.state_space
        )
        
        # py_occam/fit.py (additional calculation methods)

    def _calc_entropy(self, freqs: Dict[State, float], base: float = 2.0) -> float:
        """
        Compute entropy exactly matching VBMManager.cpp computeH().
        Uses stable log calculation to avoid drift.
        """
        total = sum(freqs.values())
        if total <= 0:
            return 0.0
            
        entropy = 0.0
        for freq in freqs.values():
            if freq > 0:  # Skip zero frequencies as C++ does
                prob = freq / total
                # Use log2 directly rather than log/log(2) for stability
                entropy -= prob * np.log2(prob)
                
        return entropy

    def _compute_l2_stats(self, model: Model, ref_model: Model) -> Dict[str, float]:
        """
        Compute L2 statistics matching VBMManager computeL2Statistics().
        Handles reference model comparison exactly as C++ does.
        """
        # Get model and reference frequencies
        q = model.get_fitted_probs()
        q_ref = ref_model.get_fitted_probs()
        
        # Compute h values
        h = self._calc_entropy(q)
        h_ref = self._calc_entropy(q_ref)
        
        # Calculate df values  
        df = model.compute_df()
        df_ref = ref_model.compute_df()
        ddf = abs(df - df_ref)
        
        # L2 statistics - match C++ exactly
        dlr = 2.0 * self.state_space.n * (h_ref - h) * np.log(2)
        
        # Compute alpha (p-value)
        alpha = 1.0
        if ddf > 0:
            alpha = 1 - stats.chi2.cdf(dlr, ddf)
            
        # Calculate AIC and BIC as in VBMManager
        aic = dlr - 2.0 * ddf
        bic = dlr - np.log(self.state_space.n) * ddf
        
        return {
            'h': h,
            'df': df,
            'ddf': ddf,
            'dlr': dlr,
            'alpha': alpha,
            'aic': aic,
            'bic': bic
        }

    def _compute_dependent_stats(self, model: Model) -> Dict[str, float]:
        """
        Compute DV uncertainty reduction stats matching VBMManager.
        This follows computeDependentStatistics() exactly.
        """
        if not self.varlist.get_dv_abbrev():
            return {}  # Return empty dict for neutral systems
            
        # Get conditional DV distributions
        q = model.get_fitted_probs()
        dv_obs = self.state_space.project_frequencies(
            self.state_space.state_frequencies, 
            [self.varlist.get_dv_abbrev()]
        )
        dv_bottom = self.state_space.project_frequencies(
            self.ref_bottom.get_fitted_probs(),
            [self.varlist.get_dv_abbrev()]
        )
            
        # Calculate entropies
        h_dv = self._calc_entropy(dv_obs)
        h_dv_bottom = self._calc_entropy(dv_bottom)
        
        # Calculate uncertainty reduction
        dh_dv = 0.0
        if h_dv_bottom > 0:
            dh_dv = 100.0 * (h_dv_bottom - h_dv) / h_dv_bottom
            
        return {
            'dh_dv': dh_dv,
            'h_dv': h_dv,
            'h_dv_bottom': h_dv_bottom
        }

    def _compute_ipf(self, model: Model, max_iter: int = 266, 
                    epsilon: float = 0.25) -> Dict[State, float]:
        """
        Compute IPF exactly matching VBMManager behavior.
        Uses stable normalization to avoid drift.
        """
        # Get component projections
        projections = []
        for comp in model.components:
            proj = self.state_space.project_frequencies(
                self.state_space.state_frequencies,
                comp
            )
            projections.append(proj)
            
        # Initialize with uniform distribution
        states = sorted(list(self.state_space.state_frequencies.keys()))
        q = {state: 1.0/len(states) for state in states}
        
        # IPF main loop - match C++ iteration exactly
        iteration = 0
        while iteration < max_iter:
            max_dev = 0.0
            
            # Update for each component
            for i, comp in enumerate(model.components):
                # Project current q
                q_proj = defaultdict(float)
                for state, prob in q.items():
                    proj_state = state.project(comp)
                    q_proj[proj_state] += prob
                        
                # Update q - use stable division
                for state, prob in q.items():
                    proj_state = state.project(comp)
                    if prob > 0 and q_proj[proj_state] > 0:
                        if proj_state in projections[i]:
                            mult = projections[i][proj_state] / q_proj[proj_state]
                            diff = abs((mult - 1.0) * prob)
                            q[state] *= mult
                            max_dev = max(max_dev, diff)
                            
                # Normalize after each component 
                total = sum(q.values())
                if total > 0:
                    q = {state: prob/total for state, prob in q.items()}
                    
            iteration += 1
            if max_dev <= epsilon:
                break
                
        return q