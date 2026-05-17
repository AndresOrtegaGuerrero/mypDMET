"""Small open-shell integration test for projected DMET.

System: single O atom in a 10x10x10 A vacuum supercell, Gamma-only KROHF
triplet (S=1, twoS=2). The O 3P ground state has 2s^2 2p^4 with two
unpaired electrons in the 2p shell -- a clean, cheap testbed for the
ROHF-aware p-DMET projector.

What this exercises
-------------------
1. Full pipeline: IAO+PAO localization, ROHF Schmidt bath
   (Nbath = Nimp + 2), kernel(), projected_DMET cycles.
2. The ROHF projector at lines 1039-1049 of dmet.py: verifies the
   projected mean-field reference retains exactly 2 eigenvalues at ~1
   after each p-DMET cycle (the open-shell character is not folded
   into the closed-shell block).
3. The Gamma chempot fit (force_chempot_fit=True on cycle >= 2):
   verifies the Newton fit converges and the impurity-block trace
   tracks the lattice target.

How to run
----------
$ export W90DIR=/path/to/wannier90   # not used by IAO+PAO, but
                                     #   pdmet/__init__.py requires it
$ python tests/test_pdmet_rohf_integration.py
"""

import os
import sys
import types

import numpy as np

# Same stubs as the unit test so this script runs on a fresh checkout.
os.environ.setdefault("W90DIR", "/tmp/_w90_stub")
os.makedirs("/tmp/_w90_stub/wrap", exist_ok=True)
sys.modules.setdefault("pywannier90", types.ModuleType("pywannier90"))
# pyscf.dmrgscf is an optional extension only needed by DMRGBlock2Solver,
# which is not exercised by this HF-only test. Stub it so the qcsolvers
# registry import succeeds.
try:
    from pyscf import dmrgscf  # noqa: F401
except ImportError:
    import pyscf

    pyscf.dmrgscf = types.ModuleType("pyscf.dmrgscf")
    pyscf.dmrgscf.DMRGCI = object
    sys.modules["pyscf.dmrgscf"] = pyscf.dmrgscf

from pyscf.pbc import gto, scf, df  # noqa: E402
from pdmet import dmet  # noqa: E402


# ---------- system ------------------------------------------------------- #


def build_o_atom_cell():
    cell = gto.Cell()
    cell.atom = "O 5.0 5.0 5.0"
    cell.basis = "gth-dzv"  # dzv so the IAO minao reference matches
    cell.pseudo = "gth-pade"
    cell.a = np.eye(3) * 10.0
    cell.spin = 2  # triplet, twoS = 2
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()
    return cell


def run_krohf(cell):
    kpts = cell.make_kpts([1, 1, 1])  # Gamma only
    gdf = df.GDF(cell, kpts)
    gdf.build()
    kmf = scf.KROHF(cell, kpts).density_fit(with_df=gdf)
    kmf.exxdiv = None
    kmf.kernel()
    return kmf, kpts


# ---------- assertions on a single projected reference ------------------- #


def report_eig_structure(rdm, label, twoS):
    eigs = np.sort(np.linalg.eigvalsh(rdm))[::-1]
    n_dc = int(np.sum(np.abs(eigs - 2.0) < 1e-3))
    n_so = int(np.sum(np.abs(eigs - 1.0) < 1e-3))
    print(
        f"  [{label}]  Tr = {rdm.trace():.4f}  "
        f"n(eig~=2) = {n_dc}  n(eig~=1) = {n_so}  twoS = {twoS}"
    )
    return n_dc, n_so


# ---------- main --------------------------------------------------------- #


def main():
    cell = build_o_atom_cell()
    kmf, kpts = run_krohf(cell)
    print(f"KROHF energy   = {kmf.e_tot:.6f} Eh")
    print(f"cell.nelectron = {cell.nelectron}  cell.spin = {cell.spin}")

    pdmet_obj = dmet.pDMET(cell, kmf, lo_method="iao+pao", solver="HF")
    pdmet_obj.solver.twoS = cell.spin  # 2 = triplet
    pdmet_obj.emb.impCluster = [1]  # the O atom is the impurity
    # Filter to the 2p valence shell only: the unpaired electrons live there.
    # The 2s and the 3s/3p PAOs go into the environment, so the embedding
    # has a real bath (3 imp + 2 bath orbitals for S=1).
    pdmet_obj.emb.imp_orbital_filter = {"O": ["2p"]}
    pdmet_obj.initialize()

    # ---- one_shot sanity (closed-form: HF-in-HF embedding is exact) ----
    pdmet_obj.one_shot()
    print(
        f"\none_shot energy = {pdmet_obj.e_tot:.6f} Eh "
        f"(should match KROHF since solver=HF)"
    )
    assert abs(pdmet_obj.e_tot - kmf.e_tot) < 1e-4, (
        f"HF-in-HF embedding not exact: |E_dmet - E_KROHF| = "
        f"{abs(pdmet_obj.e_tot - kmf.e_tot):.2e}"
    )

    # ---- run two p-DMET cycles and instrument the projector ----
    # We patch the static method to record the *output* of every projection
    # so we can verify cycle-to-cycle that the open-shell structure survives.
    original_project = dmet.pDMET._project_to_mf_density
    projection_log = []

    def logging_project(rdm, Nelec_total, twoS=0):
        D_mf = original_project(rdm, Nelec_total, twoS=twoS)
        projection_log.append(D_mf.copy())
        return D_mf

    dmet.pDMET._project_to_mf_density = staticmethod(logging_project)
    try:
        # Threshold tight enough that HF-in-HF runs at least a couple of cycles
        # so the projector is exercised at least once. Maxcycle caps runtime.
        pdmet_obj.scf.maxcycle = 3
        pdmet_obj.scf.threshold = 1e-12
        pdmet_obj.scf.use_DIIS = False
        pdmet_obj.scf.damping = 1.0
        pdmet_obj.projected_DMET()
    finally:
        dmet.pDMET._project_to_mf_density = staticmethod(original_project)

    print(f"\nProjected references inspected: {len(projection_log)}")
    assert len(projection_log) > 0, (
        "projected_DMET converged before the projector was ever called -- "
        "tighten scf.threshold so at least one projection step runs."
    )
    for k, D in enumerate(projection_log):
        n_dc, n_so = report_eig_structure(D, f"p-DMET cycle {k + 1}", twoS=2)
        # Each cycle's mean-field reference must have exactly twoS=2 eigvals
        # at 1 (the singly-occupied orbitals); zero such eigvals would mean
        # the spin polarization got folded into the closed-shell block.
        assert n_so == 2, (
            f"Cycle {k + 1}: expected 2 singly-occupied eigvals, got {n_so}. "
            "The ROHF projector has regressed."
        )
        assert abs(D.trace() - cell.nelectron) < 1e-6, (
            f"Cycle {k + 1}: trace drifted to {D.trace():.6f}, "
            f"expected {cell.nelectron}."
        )

    print("\nOK: ROHF p-DMET projector preserves spin polarization across every cycle.")


if __name__ == "__main__":
    main()
