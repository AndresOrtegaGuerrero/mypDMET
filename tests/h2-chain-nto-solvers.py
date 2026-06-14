"""H2-chain NTOs via the `nto` flag (NO NEVPT2) for the FCI multi-root solvers.

Same sigma_g -> sigma_u physics as h2-chain-nto-print.py, but NTOs come straight
from the multi-root wavefunction (solver.nto = True), proving they no longer
depend on NEVPT2.

    python tests/h2-chain-nto-solvers.py casci      # multi-root CASCI
    python tests/h2-chain-nto-solvers.py sacasscf   # state-averaged CASSCF

For DMRG see h4-dmrg-nto.py: the (2,2) toy is too small for a stable block2
DMRG-SCF, so that script uses the working h4-average cas=(4,4) setup. Expect,
for the real transition: asym ~1.4, hole occupied (sigma_g), particle virtual.
"""

import os
import sys
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf, df

from pdmet import dmet
from pdmet.tools import tchkfile

lib.logger.TIMER_LEVEL = lib.logger.INFO


def build_h2_chain_cell(n_units=1, d_hh=1.0, cell_len=3.0, vacuum=20.0):
    """Periodic chain of H2 molecules along x, centered in the y/z vacuum."""
    atoms = []
    for n in range(n_units):
        x0 = n * cell_len
        atoms += [["H", (x0, 0.0, 0.0)], ["H", (x0 + d_hh, 0.0, 0.0)]]
    ys = [a[1][1] for a in atoms]
    zs = [a[1][2] for a in atoms]
    yc = 0.5 * vacuum - 0.5 * (min(ys) + max(ys))
    zc = 0.5 * vacuum - 0.5 * (min(zs) + max(zs))
    atoms = [[s, (x, y + yc, z + zc)] for s, (x, y, z) in atoms]

    cell = gto.Cell()
    cell.atom = atoms
    cell.a = np.diag([n_units * cell_len, vacuum, vacuum])
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.spin = 0
    cell.unit = "Angstrom"
    cell.verbose = 4
    cell.build()
    return cell


def transition_asymmetry(pdmet_obj, state):
    """||T - T^T|| / ||T||: ~1.41 real hole!=particle, ~0 symmetric."""
    T = pdmet_obj.t_dm1s[state]
    nrm = np.linalg.norm(T)
    return float(np.linalg.norm(T - T.T) / nrm) if nrm else 0.0


def occ_virt_character(pdmet_obj, vec):
    """Fraction of an EO-basis orbital in the OCCUPIED vs VIRTUAL MF space."""
    mf = pdmet_obj.qcsolver.mf
    C = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ) > 0
    n2 = float(np.vdot(vec, vec).real)

    def frac(Cs):
        proj = Cs.conj().T @ vec
        return float(np.vdot(proj, proj).real) / n2

    return frac(C[:, occ]), frac(C[:, ~occ])


def pick_sigma_state(pdmet_obj, floor=1e-3):
    """Index of the most asymmetric (transition-like) excited state, or None."""
    best, best_a = None, 0.0
    for s in range(1, len(pdmet_obj.ntos_per_root)):
        lam = pdmet_obj.ntos_per_root[s]["lambdas"]
        if len(lam) and lam[0] >= floor:
            a = transition_asymmetry(pdmet_obj, s)
            if a > best_a:
                best, best_a = s, a
    return best


def make_pdmet(cell, kmf, mode, n_units, n_roots):
    """Configure pDMET for one of the three multi-root solvers, nto flag on."""
    n_active = 2 * n_units
    common = dict(w90=None, lo_method="iao+pao")

    if mode == "casci":
        obj = dmet.pDMET(cell, kmf, solver="CASCI", **common)
    elif mode == "sacasscf":
        obj = dmet.pDMET(cell, kmf, solver="SA-CASSCF", **common)
        obj.solver.state_average_ = [1.0 / n_roots] * n_roots
    else:
        raise SystemExit(
            f"unknown mode {mode!r}; use casci | sacasscf (DMRG -> h4-dmrg-nto.py)"
        )

    obj.lobasis.minao = {"H": "gth-szv"}
    obj.emb.impCluster = list(range(1, 2 * n_units + 1))
    obj.solver.twoS = 0
    obj.solver.cas = (n_active, n_active)
    obj.solver.e_shift = 0.2  # bias toward singlets
    obj.solver.nroots = n_roots
    obj.solver.nto = True  # <-- NTOs without NEVPT2
    obj.solver.nto_npairs = 1
    obj.solver.nto_lambda_floor = 1e-3
    return obj


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "casci"
    n_units = 1
    n_roots = 3
    here = os.path.dirname(os.path.abspath(__file__))
    gdf_file = os.path.join(here, "gdf-h2solv.h5")
    chk_file = os.path.join(here, "chk_HF_h2solv")
    cube_dir = os.path.join(here, f"h2_{mode}_nto_cubes")

    cell = build_h2_chain_cell(n_units)
    print(f"\nH2 chain NTOs via flag -- solver mode: {mode}, CAS=(2,2)\n")

    kpts = cell.make_kpts([1, 1, 1])
    if not os.path.exists(gdf_file):
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_file
        gdf.build()

    khf = scf.KRHF(cell, kpts).density_fit()
    khf.with_df._cderi = gdf_file
    khf.exxdiv = None
    khf.run()
    tchkfile.save_kmf(khf, chk_file)
    kmf = tchkfile.load_kmf(khf, chk_file)

    pdmet_obj = make_pdmet(cell, kmf, mode, n_units, n_roots)
    pdmet_obj.initialize()
    pdmet_obj.one_shot()

    assert pdmet_obj.ntos_per_root is not None, "no NTOs produced (flag/solver?)"

    print("\n" + "=" * 60)
    print(f"NTOs from {mode} (nto flag, no NEVPT2)")
    print("=" * 60)
    for s in range(len(pdmet_obj.ntos_per_root)):
        pdmet_obj.get_ntos(s, outdir=None)

    state = pick_sigma_state(pdmet_obj)
    if state is None:
        print("\nNo transition-like state found (try more roots / spacing).")
        return

    info = pdmet_obj.ntos_per_root[state]
    ho, _ = occ_virt_character(pdmet_obj, info["V_hole"][:, 0])
    _, pv = occ_virt_character(pdmet_obj, info["U_part"][:, 0])
    print(
        f"\nState {state}: asym={transition_asymmetry(pdmet_obj, state):.3f}, "
        f"hole {ho * 100:.0f}% occ (sigma_g), particle {pv * 100:.0f}% virt (sigma_u)"
    )
    pdmet_obj.get_ntos(state, n_pairs=1, outdir=cube_dir, grid=(60, 60, 60))
    print(f"\nDonor/acceptor cubes -> {cube_dir}")


if __name__ == "__main__":
    main()
