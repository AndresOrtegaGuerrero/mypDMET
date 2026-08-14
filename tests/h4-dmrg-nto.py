"""H4 NTOs via the `nto` flag (no NEVPT2), block2 backend -- DMRG-CI vs SA-DMRG-SCF.

H4 (= two H2) with IAO+PAO LOs and cas=(4,4), 3 roots, solver.nto = True. Runs
each solver in SOLVERS (DMRG-CI = CASCI at fixed orbitals; SA-DMRG-SCF = state-
averaged CASSCF), printing the NTO lambda table + asym per state and writing
donor/acceptor cubes for the most asymmetric (real hole != particle) transition.

    python tests/h4-dmrg-nto.py

No Wannier90 (lo_method='iao+pao'). Needs a working block2 (BLOCKEXE); runs
single-threaded to dodge the macOS/conda duplicate-libomp segfault.
"""

import os

# Set BEFORE numpy/pyscf/block2 load OpenMP. Single-threading block2 also avoids
# the duplicate-libomp crash that KMP_DUPLICATE_LIB_OK only papers over.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf, df

from pdmet import dmet
from pdmet.tools import tchkfile
from pdmet.settings import DMRGSettings

lib.logger.TIMER_LEVEL = lib.logger.INFO

SOLVERS = ["DMRG-CI", "SA-DMRG-SCF"]  # CASCI (fixed orbitals) vs SA-CASSCF


def transition_asymmetry(pdmet_obj, state):
    T = pdmet_obj.t_dm1s[state]
    nrm = np.linalg.norm(T)
    return float(np.linalg.norm(T - T.T) / nrm) if nrm else 0.0


def occ_virt_character(pdmet_obj, vec):
    mf = pdmet_obj.qcsolver.mf
    C = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ) > 0
    n2 = float(np.vdot(vec, vec).real)

    def frac(Cs):
        proj = Cs.conj().T @ vec
        return float(np.vdot(proj, proj).real) / n2

    return frac(C[:, occ]), frac(C[:, ~occ])


def pick_transition_state(pdmet_obj, floor=1e-3):
    best, best_a = None, 0.0
    for s in range(1, len(pdmet_obj.ntos_per_root)):
        lam = pdmet_obj.ntos_per_root[s]["lambdas"]
        if len(lam) and lam[0] >= floor:
            a = transition_asymmetry(pdmet_obj, s)
            if a > best_a:
                best, best_a = s, a
    return best


def run_solver(cell, kmf, solver_name, scratch_root, cube_root):
    print("\n" + "#" * 60 + f"\n# SOLVER = {solver_name}\n" + "#" * 60)
    scratch = os.path.join(scratch_root, solver_name)
    os.makedirs(scratch, exist_ok=True)

    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver=solver_name)
    pdmet_obj.lobasis.minao = {"H": "gth-szv"}
    pdmet_obj.emb.impCluster = [1, 2, 3, 4]  # all H -> 4 valence IAOs for cas=(4,4)
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (4, 4)

    pdmet_obj.solver.dmrg = DMRGSettings()
    pdmet_obj.solver.dmrg.scratch_dir = scratch
    pdmet_obj.solver.dmrg.runtime_dir = scratch
    pdmet_obj.solver.dmrg.threads = 1  # single-threaded block2

    n_roots = 3
    weights = [1.0 / n_roots] * n_roots
    pdmet_obj.solver.nroots = n_roots
    pdmet_obj.solver.state_percent = weights  # DMRG-CI / CASCI weight field
    pdmet_obj.solver.state_average_ = weights  # SA-DMRG-SCF weight field
    pdmet_obj.solver.nto = True  # <-- NTOs without NEVPT2
    pdmet_obj.solver.nto_npairs = 1
    pdmet_obj.solver.nto_lambda_floor = 1e-3

    pdmet_obj.initialize()
    pdmet_obj.one_shot()

    assert pdmet_obj.ntos_per_root is not None, "no NTOs produced (DMRG/flag?)"

    print(f"\nNTOs from {solver_name} (nto flag, no NEVPT2)")
    for s in range(len(pdmet_obj.ntos_per_root)):
        pdmet_obj.get_ntos(s, outdir=None)

    state = pick_transition_state(pdmet_obj)
    if state is None:
        print("No transition-like state found among the roots.")
        return

    info = pdmet_obj.ntos_per_root[state]
    ho, _ = occ_virt_character(pdmet_obj, info["V_hole"][:, 0])
    _, pv = occ_virt_character(pdmet_obj, info["U_part"][:, 0])
    print(
        f"State {state}: asym={transition_asymmetry(pdmet_obj, state):.3f}, "
        f"hole {ho * 100:.0f}% occupied, particle {pv * 100:.0f}% virtual"
    )
    cube_dir = os.path.join(cube_root, solver_name)
    pdmet_obj.get_ntos(state, n_pairs=1, outdir=cube_dir, grid=(50, 50, 50))
    print(f"Donor/acceptor cubes -> {cube_dir}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    gdf_file = os.path.join(here, "gdf-h4.h5")
    chk_file = os.path.join(here, "chk_HF_h4")
    scratch_root = os.path.join(here, "tmp_h4dmrg")
    cube_root = os.path.join(here, "h4_dmrg_nto_cubes")
    os.makedirs(scratch_root, exist_ok=True)

    cell = gto.Cell()
    cell.atom = """
    H 1.0 1.0 1.0
    H 1.0 1.0 2.0
    H 2.0 1.0 1.0
    H 2.0 1.0 2.0
    """
    cell.basis = "gth-dzv"
    cell.spin = 0
    cell.a = np.eye(3) * 10
    cell.verbose = 4
    cell.build()

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
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

    for solver_name in SOLVERS:
        run_solver(cell, kmf, solver_name, scratch_root, cube_root)


if __name__ == "__main__":
    main()
