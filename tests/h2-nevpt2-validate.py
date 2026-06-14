"""Validate pDMET NEVPT2 against bare pyscf on the same H2 / Gamma Hamiltonian.

pDMET with the whole molecule as the impurity is an EXACT embedding, so its
CASSCF + NEVPT2 must reproduce a plain pyscf CASSCF + NEVPT2 on the identical
gamma-point mean field. Both sides use the SAME recipe pDMET uses internally:
state-specific CASSCF(2,2) ground -> 3-root CASCI on those orbitals -> per-root
NEVPT2 (with the same fix_spin shift). gth-dzv gives 2 external orbitals, so
NEVPT2 is non-zero.

    python tests/h2-nevpt2-validate.py

Excitation/correlation energies should match to ~uHa (validates NEVPT2). If the
absolute totals also match, pDMET's embedding energy accounting is right too.
No NEVPT2/embedding bug if all the deltas are ~0.
"""

import os
import numpy as np
from pyscf.pbc import gto, scf, df
from pyscf import mcscf, mrpt

from pdmet import dmet
from pdmet.tools import tchkfile

D_HH, A_X, VAC = 1.0, 3.0, 20.0
CAS = (2, 2)
N_ROOTS = 3
E_SHIFT = 0.2
HARTREE2EV = 27.211386245988


def build_cell():
    cell = gto.Cell()
    cell.atom = [
        ["H", (0.0, 0.5 * VAC, 0.5 * VAC)],
        ["H", (D_HH, 0.5 * VAC, 0.5 * VAC)],
    ]
    cell.a = np.diag([A_X, VAC, VAC])
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.spin = 0
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()
    return cell


def reference(cell):
    """Bare pyscf: gamma RHF -> CASSCF(2,2) ground -> 3-root CASCI -> NEVPT2."""
    mf = scf.RHF(cell, exxdiv=None).density_fit()
    mf.kernel()

    mc = mcscf.CASSCF(mf, *CAS)
    mc.fix_spin_(shift=E_SHIFT, ss=0)
    mc.kernel()  # ground-state orbitals

    mc_ci = mcscf.CASCI(mf, *CAS)
    mc_ci.fcisolver.nroots = N_ROOTS
    mc_ci.fix_spin_(shift=E_SHIFT, ss=0)
    mc_ci.kernel(mc.mo_coeff)
    e_casci = np.atleast_1d(mc_ci.e_tot)

    e_nevpt2 = np.array(
        [e_casci[r] + mrpt.NEVPT(mc_ci, r).kernel() for r in range(N_ROOTS)]
    )
    return e_casci, e_nevpt2


def run_pdmet(cell, kmf):
    """pDMET with the whole H2 as impurity (exact embedding) + NEVPT2."""
    obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="CASSCF")
    obj.lobasis.minao = {"H": "gth-szv"}
    obj.emb.impCluster = [1, 2]  # full molecule
    obj.solver.twoS = 0
    obj.solver.cas = CAS
    obj.solver.e_shift = E_SHIFT
    obj.solver.nevpt2_roots = list(range(N_ROOTS))
    obj.solver.nevpt2_nroots = N_ROOTS
    obj.solver.nroots = 1
    obj.initialize()
    obj.one_shot()
    return np.asarray(obj.e_casci_tot), np.asarray(obj.e_nevpt2_tot)


def table(name, ref, pdm):
    print(f"\n{name}  (Hartree)")
    print("  root        pyscf            pDMET           delta")
    for r in range(len(ref)):
        print(f"  {r:3d}   {ref[r]:15.8f}  {pdm[r]:15.8f}  {ref[r] - pdm[r]:+.2e}")
    # excitation energies relative to root 0 (offset-free comparison)
    print("  excitation (root k - root 0):  pyscf vs pDMET (eV)")
    for r in range(1, len(ref)):
        ex_r = (ref[r] - ref[0]) * HARTREE2EV
        ex_p = (pdm[r] - pdm[0]) * HARTREE2EV
        print(f"    {r}: {ex_r:10.5f}  vs {ex_p:10.5f}   d={ex_r - ex_p:+.2e} eV")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    gdf_file = os.path.join(here, "gdf-h2val.h5")
    chk_file = os.path.join(here, "chk_HF_h2val")

    cell = build_cell()
    kpts = cell.make_kpts([1, 1, 1])
    if not os.path.exists(gdf_file):
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_file
        gdf.build()

    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.with_df._cderi = gdf_file
    kmf.exxdiv = None
    kmf.run()
    tchkfile.save_kmf(kmf, chk_file)
    kmf = tchkfile.load_kmf(kmf, chk_file)

    print("Running bare-pyscf reference ...")
    ref_cas, ref_pt2 = reference(cell)
    print("Running pDMET (full-embedded) ...")
    pdm_cas, pdm_pt2 = run_pdmet(cell, kmf)

    table("CASCI totals", ref_cas, pdm_cas)
    table("NEVPT2 totals", ref_pt2, pdm_pt2)

    print("\nNEVPT2 correlation per root (E_nevpt2 - E_casci):  pyscf vs pDMET")
    for r in range(N_ROOTS):
        cref, cpdm = ref_pt2[r] - ref_cas[r], pdm_pt2[r] - pdm_cas[r]
        print(f"  root {r}: {cref:12.8f}  vs {cpdm:12.8f}   d={cref - cpdm:+.2e}")

    dmax = max(np.abs(ref_cas - pdm_cas).max(), np.abs(ref_pt2 - pdm_pt2).max())
    print(f"\nmax |delta| (absolute totals) = {dmax:.2e} Hartree")
    print("  ~1e-6 or below  -> pDMET NEVPT2 matches bare pyscf.")


if __name__ == "__main__":
    main()
