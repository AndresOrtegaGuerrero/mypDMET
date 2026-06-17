"""Energy-consistency test for an exxdiv != None KROHF starting guess.

Open-shell twin of test_exxdiv_consistency.py. System: OH radical (doublet,
spin=1) in a 10 A box, Gamma-only KROHF. O is the impurity; the O-H sigma bond
(O 2p_z <-> H 1s) provides the entanglement that yields a genuine Schmidt bath.

A *bonded* open-shell molecule is required, not a lone atom: a lone atom's
complete shell is already a set of natural orbitals (integer occupations), so
it does not entangle with the environment and produces ZERO bath -- the trap a
single O atom falls into, regardless of box size.

What this proves
----------------
The ingest-and-strip path works per spin channel: a KROHF exxdiv='ewald' SCF
is ingested, embedded in bare Coulomb, and the exact Madelung constant
(computed via kmf.energy_tot(), which handles alpha/beta) is re-added. At the
HF level, embedded ROHF == lattice ROHF per cell, so:

    DMET(ewald guess).e_tot  ==  KROHF(ewald).e_tot          (constant re-added)
    DMET(None).e_tot         ==  KROHF(None).e_tot           (control)

How to run
----------
$ export W90DIR=/tmp/_w90_stub
$ python tests/test_exxdiv_rohf_consistency.py
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


def build_oh_cell():
    cell = gto.Cell()
    cell.atom = "O 5.0 5.0 5.0; H 5.0 5.0 5.97"  # ~0.97 A O-H bond
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.a = np.eye(3) * 10.0
    cell.spin = 1  # doublet, twoS = 1
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()
    return cell


def run_krohf(cell, exxdiv):
    """Gamma-only KROHF with an explicit exxdiv choice."""
    kpts = cell.make_kpts([1, 1, 1])
    gdf = df.GDF(cell, kpts)
    gdf.build()
    kmf = scf.KROHF(cell, kpts).density_fit(with_df=gdf)
    kmf.exxdiv = exxdiv
    kmf.kernel()
    return kmf


# ---------- one-shot HF-in-HF energy for a given exxdiv ------------------ #


def dmet_one_shot(cell, kmf):
    """Run one ROHF-in-ROHF cycle; return (e_tot, Nbath, e_madelung)."""
    obj = dmet.pDMET(cell, kmf, lo_method="iao+pao", solver="HF")
    obj.solver.twoS = cell.spin  # 1 = doublet
    obj.emb.impCluster = [1]  # the O atom is the impurity
    obj.emb.imp_orbital_filter = {"O": ["2p"]}  # O 2p; O-H bond -> real bath
    obj.initialize()
    obj.one_shot()
    return obj.e_tot, obj.Nbath, obj.e_madelung


# ---------- main --------------------------------------------------------- #


def main():
    cell = build_oh_cell()

    # --- control: exxdiv=None must reproduce its own kmf ---
    kmf_none = run_krohf(cell, exxdiv=None)
    e_none, nbath_none, _ = dmet_one_shot(cell, kmf_none)
    print(
        f"[None ]  KROHF = {kmf_none.e_tot:.8f}  DMET = {e_none:.8f}  Nbath = {nbath_none}"
    )
    assert nbath_none > 0, "no bath formed -- embedding is trivial, test is vacuous"
    assert abs(e_none - kmf_none.e_tot) < TOL, (
        f"None: ROHF-in-ROHF not exact, |dE| = {abs(e_none - kmf_none.e_tot):.2e}"
    )

    # --- target: exxdiv='ewald' guess, embedded bare + constant re-added ---
    kmf_ewald = run_krohf(cell, exxdiv="ewald")
    e_ewald, nbath_ewald, e_madelung = dmet_one_shot(cell, kmf_ewald)
    print(
        f"[ewald]  KROHF = {kmf_ewald.e_tot:.8f}  DMET = {e_ewald:.8f}  "
        f"Nbath = {nbath_ewald}  e_madelung = {e_madelung:.8f}"
    )
    assert nbath_ewald > 0, "no bath formed for the ewald guess"
    assert abs(e_ewald - kmf_ewald.e_tot) < TOL, (
        f"ewald guess: total not reproduced after re-adding the Madelung "
        f"constant, |dE| = {abs(e_ewald - kmf_ewald.e_tot):.2e}"
    )

    # --- the correction is nonzero and matches the SCF shift ---
    shift = kmf_ewald.e_tot - kmf_none.e_tot
    print(
        f"Madelung shift (ewald - none) = {shift:.8f} Eh ; e_madelung = {e_madelung:.8f}"
    )
    assert abs(shift) > 1e-6, "exxdiv='ewald' produced no shift -- check setup"
    assert abs(e_madelung - shift) < TOL, (
        f"re-added constant != SCF Madelung shift: {e_madelung:.6f} vs {shift:.6f}"
    )

    print("\nPASS: KROHF exxdiv='ewald' starting guess ingested and energy consistent.")


if __name__ == "__main__":
    main()
