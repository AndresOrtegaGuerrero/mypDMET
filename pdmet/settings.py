"""
pDMET Settings

Tunable parameters, group by responsibility:
    - scf for self consistency parameters
    - emb for embedding parameters
    - solver for the impurity solver options
    Enum-based options to facilitate new options and avoid typos


Example
    dmet = pDMET(cell, kmf, w90, solver=Solver.CASSCF)

    dmet.scf.method     = SCFMethod.BFGS
    dmet.scf.threshold  = 1e-5
    dmet.emb.impCluster = [1, 2]
    dmet.emb.OEH_type   = OEHType.FOCK
    dmet.solver.cas     = (6, 5)

"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

# All Enums shoulb be defined here


class SCFMethod(str, Enum):
    """Scipy optimization method for SCF convergence"""

    LBFGS_B = "L-BFGS-B"
    BFGS = "BFGS"
    CG = "CG"
    NEWTON_CG = "Newton-CG"


class CFType(str, Enum):
    """Type of cost function for DMET self-consistency"""

    F = "F"  # fragment block only
    DIAGF = "diagF"  # diagonal of fragment block
    FB = "FB"  # fragment + bath block
    DIAGFB = "diagFB"  # diagonal of fragment + bath block


class OEHType(str, Enum):
    FOCK = "FOCK"
    OEI = "OEI"


class CASType(str, Enum):
    FCI = "FCI"
    CheMPS2 = "CheMPS2"
    Block = "Block"


class Solver(str, Enum):
    # Single-reference
    HF = "HF"
    MP2 = "MP2"
    RCCSD = "RCCSD"
    RCCSD_T = "RCCSD_T"
    RCCSD_TSlow = "RCCSD_TSlow"
    # FCI-like
    FCI = "FCI"
    DMRG = "DMRG"
    SHCI = "SHCI"
    # CASCI family
    CASCI = "CASCI"
    DMRG_CI = "DMRG-CI"
    # CASSCF family
    CASSCF = "CASSCF"
    DMRG_SCF = "DMRG-SCF"
    SS_CASSCF = "SS-CASSCF"
    SA_CASSCF = "SA-CASSCF"
    SS_DMRG_SCF = "SS-DMRG-SCF"
    SA_DMRG_SCF = "SA-DMRG-SCF"
    # MC-PDFT family
    CASPDFT = "CASPDFT"
    SS_CASPDFT = "SS-CASPDFT"
    SA_CASPDFT = "SA-CASPDFT"


# Settings Classes


@dataclass
class SCFSettings:
    """
    Settings for the SCF optimization in DMET self-consistency
    """

    method: SCFMethod = SCFMethod.BFGS
    threshold: float = 1e-4
    maxcycle: int = 200
    CF_type: CFType = CFType.F
    damping: float = 1.0  # no damping
    use_DIIS: bool = False
    DIIS_start: int = 1
    DIIS_nvector: int = 8
    alt_CF: Optional[bool] = (
        False  # Whether to use the alternative cost function in the DMET self-consistency
    )

    def validate(self):
        if not (0.0 <= self.damping <= 1):
            raise ValueError("Damping factor should be between 0 and 1.")

        if self.use_DIIS:
            if self.DIIS_start < 1:
                raise ValueError("DIIS_start must be >= 1")


@dataclass
class EmbeddingSettings:
    """
    Setting for the embedding (impurity + bath) construction
    OEH_type  :  One-electron Hamiltonian used in the bath construction
    impCluster        : list[int]  — 1-based atom indices, None Use for Gamma DMET
    impOrbs_threshold : float - Threhold for selecting close-distance in Angstrom impurity orbitals for impCluster
    impOrbs_rmlist     : list[int]  — 1-based orbital indices to be removed from the impurity space, None
    impOrbs_addlist    : list[int]  — 1-based orbital indices to be added to the impurity space, None
    num_bath :  int -  Used to keep the no. of baths are the same as in the 1st cycle of SCF
    bath_truncation : bool  -Whether to use bath_truncatio or not.
    use_GDF : bool - Whether to use GDF for ERI transformation.
    xc : str, DFT the functional for GDF, e.g., "PBE0"
    xc_range : float range separation (auto 0.2 for RSH-PBE0)
    dft_CF : bool - Whether to use DF like cost function


    """

    OEH_type: OEHType = OEHType.FOCK
    impCluster: Optional[list] = None
    impOrbs_threshold: float = 1.0
    impOrbs_rmlist: Optional[list] = None
    impOrbs_addlist: Optional[list] = None
    num_bath: Optional[int] = None
    bath_truncation: bool = True
    use_GDF: bool = True
    xc: Optional[str] = None
    xc_omega: Optional[float] = None
    dft_CF: Optional[bool] = False
    dft_CF_constraint: Optional[int] = 1
    dft_HF: Optional[bool] = None


@dataclass
class SolverSettings:
    """
    Setting for the impurity solver (QCSolver)
    """

    name: Solver = Solver.HF
    twoS: Optional[int] = None
    nroots: int = 1
    state_percent: Optional[list] = None
    e_shift: Optional[float] = None
    cas: Optional[tuple] = None  # (n_orb, n_ele)
    molist: Optional[list] = None  # list of 1-based orbital indices for active space
    state_specific_: Optional[int] = 0
    state_average_: Optional[list] = None  # field(default_factory=lambda: [0.5, 0.5])
    state_average_mix_: Optional[tuple] = (
        None  # Wrapper can take different type of CI solvers and mix the solutions in given weights
    )
    nevpt2_roots: Optional[list] = None  # field(default_factory=list)
    nevpt2_nroots: Optional[int] = None  # int = 10
    nevpt2_spin: Optional[int] = None
    mc_dup: Optional[bool] = (
        None  # Placeholder, I need to figure out how this variable is used.
    )
    verbose: int = 0
    max_memory: int = 4000  # For impurity solver in MB
    cas_solver: CASType = CASType.FCI
    otxc: str = (
        "tPBE"  # To use for CASPDFT (Check if we can generalize to use ftPBE and tPBE0)
    )

    def validate(self):
        if self.name == Solver.RCCSD and self.twoS != 0:
            raise Exception("RCCSD solver does not support ROHF wave function")
        if "SS" in self.name:
            assert self.nroots > self.state_specific_, (
                "Number of roots should be greater than state-specific index for state-specific solvers"
            )
        if "SA" in self.name:
            assert self.nroots == len(self.state_average_), (
                "Number of roots should be equal to the length of state-average weights for state-average solvers"
            )
        if self.nevpt2_roots is not None:
            assert self.nevpt2_nroots >= len(self.nevpt2_roots), (
                "Increase the number of roots in the FCI solver"
            )
        else:
            self.nevpt2_spin = self.twoS

        if self.nroots > 1:
            if self.state_percent is not None:
                assert abs(sum(self.state_percent) - 1.0) < 1.0e-10, (
                    "The total percent has be 1"
                )
                assert len(self.state_percent) == self.nroots, (
                    "Length of state_percent should be equal to nroots"
                )
            else:
                # Set percentage
                self.state_percent = [1 / self.nroots] * self.nroots

    def to_qcsolver_kwargs(self, is_KROHF):
        return dict(
            solver=self.name,
            twoS=self.twoS,
            is_KROHF=is_KROHF,
            e_shift=self.e_shift,
            nroots=self.nroots,
            state_percent=self.state_percent,
            verbose=self.verbose,
            memory=self.max_memory,
        )

    def build_solver(self, is_KROHF=False):
        from pdmet.qcsolvers.registry import build

        return build(self, is_KROHF=is_KROHF)
