"""Simulator-independent diagnostics with explicit uncertainty and coverage."""
from .adapters import Adapter, Blowup, Unsupported
from .checks import check_assignment, check_observation, check_order
from .release import compare_releases
from .report import Feature, Modification, Report, Result, Status

__version__ = '0.1.0a2'
__all__ = ['Adapter','Blowup','Unsupported','check_assignment','check_observation',
           'check_order','compare_releases','Feature','Modification','Report','Result','Status']
