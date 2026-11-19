"""Simulator-independent diagnostics with explicit uncertainty and coverage."""
from .adapters import Adapter, Blowup, Unsupported
from .checks import check_assignment, check_observation, check_order
from .energy import check_energy
from .release import compare_releases
from .report import Feature, Modification, Report, Result, Status

from .workflow import WorkflowInputs, diagnose, plan
from .fit import EnergyFit, fit_parameter
from .laws import Law, LawStudy, LogArm, screen_laws
from .attribution import Hypothesis, Recovery

__version__ = '0.2.0a1'
__all__ = [
    'Adapter', 'Blowup', 'Unsupported', 'Feature', 'Modification', 'Report', 'Result', 'Status',
    'check_assignment', 'check_observation', 'check_order', 'check_energy', 'compare_releases',
    'WorkflowInputs', 'plan', 'diagnose', 'EnergyFit', 'fit_parameter', 'Law', 'LogArm',
    'LawStudy', 'screen_laws', 'Hypothesis', 'Recovery',
]
