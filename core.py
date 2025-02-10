# Updated core.py after refactoring to remove duplicate functionality

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
import scipy.stats as stats
from collections import defaultdict
import yaml
from pathlib import Path

# Import modularized components
from .variable_list import VariableList
from .state_space import StateSpace
from .model import Model
from .relation import Relation

# If needed, define only essential utility functions here

@dataclass
class Core:
    """
    Central OCCAM core functionality that is not handled by modularized components.
    """
    varlist: VariableList
    state_space: StateSpace
    models: List[Model] = field(default_factory=list)
    
    def add_model(self, model: Model):
        """Add a model to the core structure"""
        self.models.append(model)
    
    def get_models(self) -> List[Model]:
        """Retrieve all stored models"""
        return self.models
    
    def process_relation(self, variables: List[str]) -> Relation:
        """Process a relation given a list of variables"""
        return Relation(variables=variables)
