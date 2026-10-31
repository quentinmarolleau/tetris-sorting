"""Tetris algorithm for optical tweezer array sorting.

Reproduction of Wang et al., Phys. Rev. Applied 19, 054032 (2023): a
stochastically loaded array of optical tweezers is rearranged into a
defect-free target by moving whole rows of atoms, then whole columns, in
parallel. The package re-exports the readable implementation
(`AtomsConfiguration`), the target geometry and acceptance rule
(`tetris.acceptance`), and the fast cost computation the benchmarks use
(`tetris.fast`).
"""

from tetris.acceptance import TargetWindow, configuration_kept, target_geometry
from tetris.config import AtomsConfiguration, OccupationMatrix
from tetris.fast import (
    DEFAULT_RUN_SELECTION,
    RearrangementMetrics,
    RunCriterion,
    parallel_displacements,
    rearrangement_metrics,
)

__all__ = [
    "DEFAULT_RUN_SELECTION",
    "AtomsConfiguration",
    "OccupationMatrix",
    "RearrangementMetrics",
    "RunCriterion",
    "TargetWindow",
    "configuration_kept",
    "parallel_displacements",
    "rearrangement_metrics",
    "target_geometry",
]
