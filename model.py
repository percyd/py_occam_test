"""Implementation of Model class for OCCAM variable-based reconstructability analysis"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import scipy.stats as stats
from .definitions import Model, State, Relation

class ModelImplementation:
    """Implementation class for Model operations"""
    
    @staticmethod
    def validate_model(model: Model) -> None:
        """Validate model components"""
        if not model.components:
            raise ValueError("Model must have at least one component")
                
        # Verify all variables used are valid
        all_vars = set(model.variables)
        for comp in model.components:
            if not set(comp).issubset(all_vars):
                raise ValueError(f"Invalid variables in component {comp}")
            
        # For directed systems, verify DV in predicting components
        if model.dv_col:
            for comp in model.components[1:]:  # Skip IV component
                if model.dv_col not in comp:
                    raise ValueError(f"Component {comp} missing DV")

    @staticmethod
    def get_name(model: Model) -> str:
        """Get model name in OCCAM format"""
        parts = []
        iv_vars = set(v for v in model.variables if v != model.dv_col)
        
        for comp in model.components:
            if set(comp) == iv_vars:
                parts.append("IV")
            else:
                parts.append("".join(sorted(comp)))
        return ":".join(parts)

    @staticmethod
    def get_predicting_components(model: Model) -> List[List[str]]:
        """Get components containing DV"""
        return [comp for comp in model.components 
                if model.dv_col in comp]
                
    @staticmethod
    def get_iv_component(model: Model) -> Optional[List[str]]:
        """Get IV component if present"""
        iv_vars = set(v for v in model.variables if v != model.dv_col)
        for comp in model.components:
            if set(comp) == iv_vars:
                return comp
        return None

    @staticmethod
    def is_loopless(model: Model) -> bool:
        """Check if model is loopless"""
        # For directed systems, loopless means single predicting component
        if model.dv_col:
            pred_comps = model.get_predicting_components()
            return len(pred_comps) == 1
            
        # For neutral systems, check for variable overlap
        for i, comp1 in enumerate(model.components):
            for comp2 in model.components[i+1:]:
                if len(set(comp1) & set(comp2)) > 1:
                    return False
        return True

    @staticmethod
    def get_fitted_probs(model: Model) -> Dict[State, float]:
        """Get fitted probabilities using IPF matching C++ VBMManager exactly"""
        # Get frequencies projected onto each component
        comp_tables = []
        for comp in model.components:
            comp_tables.append(model.state_space.project_frequencies(
                model.state_space.state_frequencies,
                comp
            ))
                
        # Initialize with all states from frequency data
        states = sorted(list(model.state_space.state_frequencies.keys()))
        
        # Initialize uniform distribution exactly as C++ does   
        q = {state: 1.0/len(states) for state in states}
        
        # IPF main loop - Modified to match C++ exactly
        MAX_ITERATIONS = 266  # From C++ defaults
        EPSILON = 0.25  # From C++ ipf-maxdev default
        iteration = 0
        
        while iteration < MAX_ITERATIONS:
            max_dev = 0.0
            
            # Iterate through components
            for i, comp in enumerate(model.components):
                # Project current q onto component variables
                q_proj = defaultdict(float)
                for state, prob in q.items():
                    proj_state = state.project(comp)
                    q_proj[proj_state] += prob
                        
                # Update q based on ratio - Match C++ calculation
                for state, prob in q.items():
                    proj_state = state.project(comp)
                    if prob > 0 and q_proj[proj_state] > 0 and proj_state in comp_tables[i]:
                        mult = comp_tables[i][proj_state] / q_proj[proj_state]
                        diff = abs((mult - 1.0) * prob)  # Match C++ calculation
                        q[state] *= mult
                        max_dev = max(max_dev, diff)
                
                # Normalize after each component (match C++)
                total = sum(q.values())
                if total > 0:
                    q = {state: prob/total for state, prob in q.items()}
                    
            iteration += 1
            if max_dev <= EPSILON:
                break
                
        return q

    @staticmethod
    def compute_model_statistics(model: Model, ref_model: Optional["Model"] = None):
        """Compute model statistics matching C++ VBMManager"""
        # Create relations for this model 
        if not hasattr(model, '_relations'):
            model._relations = model.create_relations()
            
        # Get reference model if needed
        if ref_model is None:
            # Create independence model
            ref_components = [
                sorted(model.state_space.iv_cols),
                [model.dv_col]
            ]
            ref_model = Model(
                components=ref_components,
                variables=model.variables,
                dv_col=model.dv_col,
                state_space=model.state_space
            )
        
        # Get reference relations
        ref_relations = ref_model.create_relations()
        
        # Get fitted probabilities
        q = model.get_fitted_probs()
        
        # Get reference model probabilities
        ref_q = ref_model.get_fitted_probs()
        
        # Calculate model df and entropy
        df = sum(rel.compute_df(model.state_space) 
                for rel in model._relations)
        h = model.compute_entropy(q)
        
        # Get reference values
        ref_df = sum(rel.compute_df(model.state_space) 
                    for rel in ref_relations)
        ref_h = ref_model.compute_entropy(ref_q)
        
        # Compute statistics matching C++ exactly 
        ddf = abs(df - ref_df)
        dlr = 2.0 * model.state_space.n * (ref_h - h) * np.log(2)
        
        # Alpha
        alpha = 1.0
        if ddf > 0:
            alpha = 1 - stats.chi2.cdf(dlr, ddf)
            
        # Information
        top_h, bottom_h, iv_h = model.state_space.compute_reference_entropies()
        if abs(bottom_h - top_h) > 1e-10:
            information = (bottom_h - h) / (bottom_h - top_h)
            information = max(0, min(1, information))
        else:
            information = 0.0
            
        # DV uncertainty reduction
        dh_dv = 0.0
        if model.dv_col:
            # Get DV marginals
            dv_obs = model.state_space.project_frequencies(
                model.state_space.state_frequencies, 
                [model.dv_col]
            )
            dv_bottom = model.state_space.project_frequencies(
                ref_q,
                [model.dv_col]
            )
            
            h_dv = model.state_space.compute_entropy(dv_obs)
            h_dv_bottom = model.state_space.compute_entropy(dv_bottom)
            
            if h_dv_bottom > 0:
                dh_dv = 100.0 * (h_dv_bottom - h_dv) / h_dv_bottom
                
        # AIC and BIC
        aic = dlr - 2.0*ddf
        bic = dlr - np.log(model.state_space.n)*ddf
        
        # Update model statistics
        model.statistics.update({
            'h': h,
            'df': df,
            'ddf': ddf,
            'dlr': dlr,
            'alpha': alpha,
            'information': information,
            'dh_dv': dh_dv,
            'aic': aic,
            'bic': bic
        })

    @staticmethod 
    def create_relations(model: Model) -> List[Relation]:
        """Create relations from components matching C++ VBMManager"""
        relations = []
        
        # First convert each component to a relation
        for comp in model.components:
            relation = model.state_space.make_relation(comp)
            relations.append(relation)
                
        return relations

    @staticmethod
    def compute_incremental_alpha(model: Model, progenitor: "Model") -> float:
        """Compute incremental alpha between model and progenitor"""
        # Get change in df and likelihood ratio
        ddf = abs(model.compute_df() - progenitor.compute_df())
        dlr = abs(model.statistics['dlr'] - progenitor.statistics['dlr'])
        
        # Compute p-value
        if ddf > 0:
            return 1 - stats.chi2.cdf(dlr, ddf)
        return 1.0

# Add implementation methods to Model class
def _model_post_init(self):
    """Post-init initialization"""
    ModelImplementation.validate_model(self)
    
def _model_get_name(self):
    """Get model name"""
    return ModelImplementation.get_name(self)
    
def _model_get_predicting_components(self):
    """Get DV components"""
    return ModelImplementation.get_predicting_components(self)
    
def _model_get_iv_component(self):
    """Get IV component"""
    return ModelImplementation.get_iv_component(self)
    
def _model_is_loopless(self):
    """Check if loopless"""
    return ModelImplementation.is_loopless(self)
    
def _model_get_fitted_probs(self):
    """Get fitted probabilities"""
    return ModelImplementation.get_fitted_probs(self)
    
def _model_compute_statistics(self, ref_model=None):
    """Compute statistics"""
    return ModelImplementation.compute_model_statistics(self, ref_model)
    
def _model_create_relations(self):
    """Create relations"""
    return ModelImplementation.create_relations(self)
    
def _model_compute_incremental_alpha(self, progenitor):
    """Compute incremental alpha"""
    return ModelImplementation.compute_incremental_alpha(self, progenitor)

def _model_str(self):
    """String representation"""
    return f"Model({self.get_name()})"

# Add implementation methods to Model class
Model.__post_init__ = _model_post_init
Model.get_name = _model_get_name
Model.get_predicting_components = _model_get_predicting_components
Model.get_iv_component = _model_get_iv_component
Model.is_loopless = _model_is_loopless
Model.get_fitted_probs = _model_get_fitted_probs
Model.compute_model_statistics = _model_compute_statistics
Model.create_relations = _model_create_relations
Model.compute_incremental_alpha = _model_compute_incremental_alpha
Model.__str__ = _model_str
Model.__repr__ = _model_str