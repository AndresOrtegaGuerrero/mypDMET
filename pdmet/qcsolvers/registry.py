from pdmet.settings import Solver
from pdmet.qcsolvers.hf import HFSolver
from pdmet.qcsolvers.mp2 import MP2Solver
from pdmet.qcsolvers.rccsd import RCCSDSolver, RCCSD_TSolver, RCCSD_TSlowSolver
from pdmet.qcsolvers.fci import FCISolver
from pdmet.qcsolvers.shci import SHCISolver
from pdmet.qcsolvers.casci import CASCISolver
from pdmet.qcsolvers.casscf import CASSCFSolver
from pdmet.qcsolvers.caspdft import CASPDFTSolver
from pdmet.qcsolvers.dmrgblock2 import DMRGBlock2Solver

_REGISTRY = {
    Solver.HF: {
        "cls": HFSolver,
    },
    Solver.MP2: {
        "cls": MP2Solver,
    },
    Solver.RCCSD: {
        "cls": RCCSDSolver,
    },
    Solver.RCCSD_T: {
        "cls": RCCSD_TSolver,
    },
    Solver.RCCSD_TSlow: {
        "cls": RCCSD_TSlowSolver,
    },
    Solver.FCI: {
        "cls": FCISolver,
    },
    Solver.SHCI: {
        "cls": SHCISolver,
    },
    Solver.CASCI: {
        "cls": CASCISolver,
        "family": "CASCI",
    },
    Solver.DMRG_CI: {
        "cls": CASCISolver,
        "family": "CASCI",
        "fci_solver": "CheMPS2",
    },
    Solver.CASSCF: {
        "cls": CASSCFSolver,
        "family": "CASSCF",
    },
    Solver.DMRG_SCF: {
        "cls": CASSCFSolver,
        "family": "CASSCF",
        "fci_solver": "CheMPS2",
    },
    Solver.SS_CASSCF: {
        "cls": CASSCFSolver,
        "family": "CASSCF",
        "state": "SS",
    },
    Solver.SA_CASSCF: {
        "cls": CASSCFSolver,
        "family": "CASSCF",
        "state": "SA",
    },
    Solver.SS_DMRG_SCF: {
        "cls": CASSCFSolver,
        "family": "CASSCF",
        "state": "SS",
        "fci_solver": "CheMPS2",
    },
    Solver.SA_DMRG_SCF: {
        "cls": CASSCFSolver,
        "family": "CASSCF",
        "state": "SA",
        "fci_solver": "CheMPS2",
    },
    #  Solver.DMRG       : (DMRGSolver,   {}),
    Solver.CASPDFT: {
        "cls": CASPDFTSolver,
        "family": "CASPDFT",
        # "fci_solver": "CheMPS2",
    },
    Solver.SS_CASPDFT: {
        "cls": CASPDFTSolver,
        "family": "CASPDFT",
        "state": "SS",
        # "fci_solver": "CheMPS2",
    },
    Solver.SA_CASPDFT: {
        "cls": CASPDFTSolver,
        "family": "CASPDFT",
        "state": "SA",
        # "fci_solver": "CheMPS2",
    },
    Solver.SA_DMRG_SCF: {
        "cls": DMRGBlock2Solver,
        "family": "CASSCF",
        "state": "SA",
    },
    Solver.SS_DMRG_SCF: {
        "cls": DMRGBlock2Solver,
        "family": "CASSCF",
        "state": "SS",
    },
}


def build(settings, is_KROHF=False):
    """Build the correct solver class using SolverSettings."""
    entry = _REGISTRY.get(settings.name)
    if entry is None:
        raise ValueError(f"Solver {settings.name!r} not in registry.")

    return entry["cls"](settings, is_KROHF=is_KROHF)


def dispatch(solver_instance, settings, pdft_context=None):
    """Call kernel() with the correct kwargs from SolverSettings."""
    entry = _REGISTRY[settings.name]
    kwargs = {}

    family = entry.get("family")
    state = entry.get("state")

    # # State-specific / state-average for CASSCF family
    if state == "SS":
        kwargs["state_specific_"] = settings.state_specific_
    elif state == "SA":
        kwargs["state_average_"] = settings.state_average_
        kwargs["state_average_mix_"] = settings.state_average_mix_

    # CASPDFT - pass extra dictionary of inputs
    if family == "CASPDFT":
        if pdft_context is None:
            raise ValueError(f"{settings.name} requires pdft_context")
        kwargs["pdft_context"] = pdft_context

    if "fci_solver" in entry:
        kwargs["fci_solver"] = entry["fci_solver"]

    return solver_instance.kernel(**kwargs)
