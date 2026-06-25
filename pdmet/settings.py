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

from dataclasses import dataclass, field
from typing import Optional, List, Union
from enum import Enum
from pyscf import lib
import os


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
    Block2 = "Block2"


class LOMethod(str, Enum):
    WANNIER = "wannier"
    IAO_PAO = "iao+pao"


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
class LocalBasisSettings:
    """
    Settings for the local-orbital basis used by the embedding.

    method      : which LO construction to use ("wannier" or "iao+pao")
    kmesh       : (3,) tuple of ints. Always required (single source of truth).
                  For Wannier, must equal w90.mp_grid_loc.
    w90         : pyWannier90 object. Required if method == "wannier".
    minao       : reference basis for IAO valence (IAO+PAO only).
                  Either a built-in name string ("gth-szv-molopt-sr", ...)
                  or a per-element dict matching ``cell.basis`` style:
                  ``{"Er": gto.basis.parse(er_szv_str), "Mo": "gth-szv-molopt-sr", ...}``
                  Use the dict form when one of your elements has no
                  built-in minimal-valence basis registered in PySCF.
    orth_virt   : Lowdin-orthogonalize PAO virtuals at every k (IAO+PAO only)
    w90_chkfile : optional path to a Wannier-90 chkfile.
    lo_chkfile  : optional path to a saved IAO+PAO chkfile

    allow_smearing  : if True, IAO uses density-weighted projection and
                      get_occ_rhf uses Fermi-Dirac smearing
    smearing_sigma  : Hartree, used when allow_smearing is True
    nocc            : optional explicit occupation count (overrides mo_occ
                      threshold). float or array (spin, nkpts).
    frozen_core_per_atom : dict[str, int], e.g. {"Ni": 5} freezes 1s2s2p3s3p
                          on every Ni atom. None means no freezing.
    """

    method: LOMethod = LOMethod.WANNIER
    w90: Optional[object] = None
    minao: Union[str, dict] = "minao"
    orth_virt: bool = True
    w90_chkfile: Optional[str] = None
    lo_chkfile: Optional[str] = None

    # Placeholder
    # until the smearing path is implemented
    allow_smearing: bool = False
    smearing_sigma: Optional[float] = None
    nocc: Optional[Union[float, list]] = None
    frozen_core_per_atom: Optional[dict] = None

    def validate(self):
        if self.method == LOMethod.WANNIER:
            if self.w90 is None:
                raise ValueError("Wannier method requires w90.")
            if self.lo_chkfile is not None:
                raise ValueError("lo_chkfile is for IAO+PAO method only.")

        else:
            if self.w90_chkfile is not None:
                raise ValueError("w90_chkfile is for Wannier method only.")
        # To be implemented
        if self.allow_smearing or self.frozen_core_per_atom is not None:
            raise NotImplementedError("Smearing / frozen-core comming soon.")


@dataclass
class StateConfig:
    spin: int
    roots: int
    weights: Union[float, List[float]]

    def __post_init__(self):
        if isinstance(self.weights, float):
            self.weights = [self.weights] * self.roots
        elif isinstance(self.weights, list):
            if len(self.weights) != self.roots:
                raise ValueError("Length of weights should match the number of roots")
        else:
            raise ValueError("Weights should be either a float or a list of floats")

    def total_weight(self):
        return sum(self.weights)


@dataclass
class DMRGSettings:
    memory: int = 4  # Memory in GB for DMRG solver
    maxM: int = 500  # Maximum bond dimension for DMRG solver
    tol: float = 1e-7  # DMRG convergence tolerance
    conv_tol: float = 1e-10  # DMRG convergence tolerance for the converged check
    BLOCKEXE: str = (
        os.popen("which block2main").read().strip()
    )  # Executable for Block DMRG, if using Block as the DMRG solver
    MPIPREFIX: Optional[str] = (
        ""  # MPI prefix for running DMRG in parallel, e.g., "mpirun -np 4"
    )
    nevpt2_maxM: int = 200  # Maximum bond dimension for DMRG-NEVPT2 solver
    use_compress_nevpt2: bool = False  # Whether to use compression in DMRG-NEVPT2
    threads: int = 1  # Number of threads for DMRG solver
    runtime_dir: Optional[str] = field(default_factory=lambda: lib.param.TMPDIR)
    scratch_dir: Optional[str] = field(default_factory=lambda: lib.param.TMPDIR)
    det_cutoff: float = (
        0.05  # Determinant cutoff for printing CI coefficients in DMRG solver
    )


@dataclass
class SCFSettings:
    """
    Settings for the SCF optimization in DMET self-consistency
    """

    method: SCFMethod = SCFMethod.BFGS
    threshold: float = 1e-4
    maxcycle: int = 100
    CF_type: CFType = CFType.F
    damping: float = 1.0  # no damping
    use_DIIS: bool = False
    DIIS_start: int = 1
    DIIS_nvector: int = 8
    alt_CF: Optional[bool] = (
        False  # alternative cost function in the DMET self-consistency
    )
    chkfile_freq: int = 5  # Save a restart checkpoint every N cycles

    def validate(self):
        if not (0.0 <= self.damping <= 1):
            raise ValueError("Damping factor should be between 0 and 1.")

        if self.use_DIIS:
            if self.DIIS_start < 1:
                raise ValueError("DIIS_start must be >= 1")
        if self.chkfile_freq < 1:
            raise ValueError("chkfile_freq must be >= 1")


@dataclass
class EmbeddingSettings:
    """
    Setting for the embedding (impurity + bath) construction
    OEH_type  :  One-electron Hamiltonian used in the bath construction
    impCluster        : list[int]  — 1-based atom indices, None Use for Gamma DMET
    impOrbs_threshold : float - Threhold for selecting close-distance in Angstrom impurity orbitals for impCluster
    impOrbs_rmlist     : list[int]  — 1-based orbital indices to be removed from the impurity space, None
    impOrbs_addlist    : list[int]  — 1-based orbital indices to be added to the impurity space, None
    imp_orbital_filter: Optional[dict] = None  Restrict the impurity to specific orbital shells per atom.
        Examples:
            {"Ni": ["3d"], "O": ["2p"]}        # by element
            {1: ["3d"], 2: ["2p"]}             # by atom index
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
    imp_orbital_filter: Optional[dict] = None
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
    mo_restart: Optional[object] = None  # Restart orbitals for CASSCF/DMRG-SCF
    state_specific_: Optional[int] = 0
    state_average_: Optional[list] = None  # field(default_factory=lambda: [0.5, 0.5])
    state_average_mix_: Optional[List[Union[StateConfig, dict]]] = (
        None  # list of (root1, root2) pairs to mix in state-average CASSCF
    )
    nevpt2_roots: Optional[list] = None  # field(default_factory=list)
    nevpt2_nroots: Optional[int] = None  # int = 10
    nevpt2_spin: Optional[int] = None
    nto: bool = False  # compute NTOs from the multi-root transition densities
    nto_export: bool = False  # also write NTO cubes at the end of one_shot()/run()?
    nto_npairs: int = 2  # how many top (donor, acceptor) pairs per root
    nto_lambda_floor: float = 1e-3  # skip pairs below this weight even if asked
    verbose: int = 0
    max_memory: int = 4000  # For impurity solver in MB
    cas_solver: CASType = CASType.FCI
    otxc: str = (
        "tPBE"  # To use for CASPDFT (Check if we can generalize to use ftPBE and tPBE0)
    )
    dmrg: Optional[DMRGSettings] = None  # DMRG-specific settings

    def validate(self):
        if self.name == Solver.RCCSD and self.twoS != 0:
            raise Exception("RCCSD solver does not support ROHF wave function")
        if "SS-" in self.name:
            self._validatate_state_specific()

        if "SA-" in self.name:
            self._validate_state_average()
            self._validate_state_average_mix()

        if self.nevpt2_roots is not None:
            if self.state_average_mix_ is not None:
                assert len(self.nevpt2_roots) == len(self.state_average_mix_), (
                    "Length of nevpt2_roots should match the number of states in state_average_mix_"
                )

            if self.state_average_ is not None:
                assert len(self.nevpt2_roots) == self.nroots, (
                    "Length of nevpt2_roots should match nroots for state-average solvers"
                )
            if self.nevpt2_spin is None:
                self.nevpt2_spin = self.twoS

        if self.nto or self.nto_export:
            # NTOs need a multi-root wavefunction (ground + >=1 excited root);
            # NEVPT2 is not required.
            multi_root = (
                self.nevpt2_roots is not None
                or self.nroots > 1
                or self.state_average_ is not None
                or self.state_average_mix_ is not None
            )
            if not multi_root:
                raise ValueError(
                    "nto/nto_export require a multi-root calculation (set "
                    "nevpt2_roots, nroots>1, state_average_, or state_average_mix_)."
                )
            if self.nto_npairs < 1:
                raise ValueError("nto_npairs must be >= 1.")
            if self.nto_lambda_floor < 0:
                raise ValueError("nto_lambda_floor must be non-negative.")

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

    def _validatate_state_specific(self):
        assert self.nroots > self.state_specific_, (
            "Number of roots should be greater than state-specific index for state-specific solvers"
        )
        assert self.state_average_ is None, (
            "State-average weights should be None for state-specific solvers"
        )
        assert self.state_average_mix_ is None, (
            "State-average mixing should be None for state-specific solvers"
        )

    def _validate_state_average(self):
        assert self.nroots > 1, (
            "Number of roots should be greater than 1 for state-average solvers"
        )
        assert self.state_specific_ is not None, (
            "State-specific index should be None for state-average solvers"
        )
        assert (self.state_average_ is not None) ^ (
            self.state_average_mix_ is not None
        ), "Exactly one of state_average_ or state_average_mix_ must be provided"

        if self.state_average_ is not None:
            assert len(self.state_average_) == self.nroots, (
                "Length of state-average weights should be equal to nroots for state-average solvers"
            )
            assert abs(sum(self.state_average_) - 1.0) < 1.0e-10, (
                "State-average weights should sum to 1 for state-average solvers"
            )

    def _validate_state_average_mix(self):
        if self.state_average_mix_ is not None:
            clean_mix = []
            for item in self.state_average_mix_:
                if isinstance(item, StateConfig):
                    clean_mix.append(item)
                elif isinstance(item, dict):
                    clean_mix.append(StateConfig(**item))
                else:
                    raise ValueError(
                        "Items in state_average_mix_ must be either StateConfig or dict"
                    )
            self.state_average_mix_ = clean_mix

            total_weights = sum(
                config.total_weight() for config in self.state_average_mix_
            )
            assert abs(total_weights - 1.0) < 1e-8, (
                "Total weights across all states must sum to 1"
            )

            assert len(self.state_average_mix_) > 1, (
                "There should be at least two states to mix in state_average_mix_"
            )

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
