"""Energy-consistency test for an exxdiv != None starting guess (Step 3).

System: H2 molecule in a 6 A box, Gamma-only KRHF. impCluster picks ONE H as
the impurity; the other H provides a genuine Schmidt bath (the textbook
1-impurity / 1-bath embedding). This avoids the lone-atom trap, where a
closed-shell atom in vacuum produces ZERO bath and the embedding is trivial.

What this proves
----------------
We ingest an exxdiv='ewald' SCF as the DMET starting guess. The Madelung term
is a pure constant shift on the occupied subspace -- it leaves every MO, the
density, and the bath untouched -- so the embedding is built in bare Coulomb
(kmf.exxdiv stripped to None internally) and the exact energy constant is
re-added at the end. At the HF level, embedded HF == lattice HF per cell, so:

    DMET(ewald guess).e_tot  ==  KRHF(ewald).e_tot          (constant re-added)
    DMET(None).e_tot         ==  KRHF(None).e_tot           (control)

A real (nonzero) bath must be present, else the test is vacuous.

How to run
----------
$ export W90DIR=/tmp/_w90_stub
$ python tests/test_exxdiv_consistency.py
"""

import os
import sys
import types

import numpy as np

# Same stubs as the other integration tests so this runs on a fresh checkout.
os.environ.setdefault("W90DIR", "/tmp/_w90_stub")
os.makedirs("/tmp/_w90_stub/wrap", exist_ok=True)
sys.modules.setdefault("pywannier90", types.ModuleType("pywannier90"))
try:
    from pyscf import dmrgscf  # noqa: F401
except ImportError:
    import pyscf

    pyscf.dmrgscf = types.ModuleType("pyscf.dmrgscf")
    pyscf.dmrgscf.DMRGCI = object
    sys.modules["pyscf.dmrgscf"] = pyscf.dmrgscf

from pyscf.pbc import df, gto, scf  # noqa: E402

from pdmet import dmet  # noqa: E402

TOL = 1e-4  # HF-in-HF embedding is exact to SCF convergence


# ---------- system ------------------------------------------------------- #


def build_h2_cell():
    cell = gto.Cell()
    cell.atom = "H 3.0 3.0 2.63; H 3.0 3.0 3.37"  # ~0.74 A bond, centered
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.a = np.eye(3) * 6.0
    cell.spin = 0
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()
    return cell


def run_krhf(cell, exxdiv):
    """Gamma-only KRHF with an explicit exxdiv choice."""
    kpts = cell.make_kpts([1, 1, 1])
    gdf = df.GDF(cell, kpts)
    gdf.build()
    kmf = scf.KRHF(cell, kpts).density_fit(with_df=gdf)
    kmf.exxdiv = exxdiv
    kmf.kernel()
    return kmf


# ---------- one-shot HF-in-HF energy for a given exxdiv ------------------ #


def dmet_one_shot(cell, kmf):
    """Run one HF-in-HF cycle; return (e_tot, Nbath, e_madelung)."""
    obj = dmet.pDMET(cell, kmf, lo_method="iao+pao", solver="HF")
    obj.emb.impCluster = [1]  # first H is the impurity
    obj.emb.imp_orbital_filter = {"H": ["1s"]}  # 1s valence; the other H is bath
    obj.initialize()
    obj.one_shot()
    return obj.e_tot, obj.Nbath, obj.e_madelung


# ---------- main --------------------------------------------------------- #


def main():
    cell = build_h2_cell()

    # --- control: exxdiv=None must reproduce its own kmf ---
    kmf_none = run_krhf(cell, exxdiv=None)
    e_none, nbath_none, _ = dmet_one_shot(cell, kmf_none)
    print(
        f"[None ]  KRHF = {kmf_none.e_tot:.8f}  DMET = {e_none:.8f}  Nbath = {nbath_none}"
    )
    assert nbath_none > 0, "no bath formed -- embedding is trivial, test is vacuous"
    assert abs(e_none - kmf_none.e_tot) < TOL, (
        f"None: HF-in-HF not exact, |dE| = {abs(e_none - kmf_none.e_tot):.2e}"
    )

    # --- target: exxdiv='ewald' guess, embedded bare + constant re-added ---
    kmf_ewald = run_krhf(cell, exxdiv="ewald")
    e_ewald, nbath_ewald, e_madelung = dmet_one_shot(cell, kmf_ewald)
    print(
        f"[ewald]  KRHF = {kmf_ewald.e_tot:.8f}  DMET = {e_ewald:.8f}  "
        f"Nbath = {nbath_ewald}  e_madelung = {e_madelung:.8f}"
    )
    assert nbath_ewald > 0, "no bath formed for the ewald guess"
    assert abs(e_ewald - kmf_ewald.e_tot) < TOL, (
        f"ewald guess: total not reproduced after re-adding the Madelung "
        f"constant, |dE| = {abs(e_ewald - kmf_ewald.e_tot):.2e}"
    )

    # --- the correction is nonzero: the two SCF references must differ ---
    shift = kmf_ewald.e_tot - kmf_none.e_tot
    print(
        f"Madelung shift (ewald - none) = {shift:.8f} Eh ; e_madelung = {e_madelung:.8f}"
    )
    assert abs(shift) > 1e-6, "exxdiv='ewald' produced no shift -- check setup"
    assert abs(e_madelung - shift) < TOL, (
        f"re-added constant != SCF Madelung shift: {e_madelung:.6f} vs {shift:.6f}"
    )

    print("\nPASS: exxdiv='ewald' starting guess ingested and energy consistent.")


if __name__ == "__main__":
    main()
