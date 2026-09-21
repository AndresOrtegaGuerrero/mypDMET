"""H4 NTOs + NDOs via the `nto`/`ndo` flags (no NEVPT2), block2 backend --
DMRG-CI vs SA-DMRG-SCF.

H4 (= two H2 at 1.0 A) with IAO+PAO LOs and cas=(4,4), 3 roots. Runs each
solver in SOLVERS, printing NTO lambda and NDO kappa tables per state and
writing cubes for the most asymmetric transition. NOTE: this near-square H4
is a DIRADICAL, so the expected physics is lam ~ 1 with |kappa| << lam
(open-shell recoupling, not charge transfer) -- the NTO/NDO disagreement is
the diagnostic. Ends with a DMRG-CI vs SA-DMRG-SCF summary table: the two
solvers agreeing on kappa is the canary for the natorb/noreorder file basis.

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
    pdmet_obj.solver.ndo = True  # <-- NDOs from the same solve (block2 1pdm files)
    pdmet_obj.solver.ndo_npairs = 2
    pdmet_obj.solver.ndo_kappa_floor = 1e-3

    pdmet_obj.initialize()
    pdmet_obj.one_shot()

    assert pdmet_obj.ntos_per_root is not None, "no NTOs produced (DMRG/flag?)"
    assert pdmet_obj.ndos_per_root is not None, "no NDOs produced (DMRG/flag?)"
    assert len(pdmet_obj.ndos_per_root) == len(pdmet_obj.ntos_per_root), (
        "NTO and NDO lists misaligned"
    )

    print(f"\nNTOs + NDOs from {solver_name} (flags, no NEVPT2)")
    for s in range(len(pdmet_obj.ntos_per_root)):
        pdmet_obj.get_ntos(s, outdir=None)
        pdmet_obj.get_ndos(s, outdir=None)

    # global NDO bookkeeping: Tr(Delta) = 0 and p_A = -p_D for every root
    for s in range(1, len(pdmet_obj.ndos_per_root)):
        info = pdmet_obj.ndos_per_root[s]
        assert abs(np.trace(pdmet_obj.d_dm1s[s])) < 1e-6, f"Tr(Delta) != 0 (state {s})"
        assert abs(info["p_A"] + info["p_D"]) < 1e-6, f"p_A != -p_D (state {s})"

    state = pick_transition_state(pdmet_obj)
    if state is None:
        print("No transition-like state found among the roots.")
        return None

    info = pdmet_obj.ntos_per_root[state]
    ho, _ = occ_virt_character(pdmet_obj, info["V_hole"][:, 0])
    _, pv = occ_virt_character(pdmet_obj, info["U_part"][:, 0])
    print(
        f"State {state}: asym={transition_asymmetry(pdmet_obj, state):.3f}, "
        f"hole {ho * 100:.0f}% occupied, particle {pv * 100:.0f}% virtual"
    )
    lam0 = float(pdmet_obj.ntos_per_root[state]["lambdas"][0])
    ndo = pdmet_obj.ndos_per_root[state]
    kap0 = float(abs(ndo["kappa"][0])) if len(ndo["kappa"]) else 0.0
    print(
        f"State {state}: top lambda={lam0:.3f}, top |kappa|={kap0:.3f}, "
        f"p={ndo['p_A']:.3f}  (diradical: expect |kappa| << lambda)"
    )

    cube_dir = os.path.join(cube_root, solver_name)
    pdmet_obj.get_ntos(state, n_pairs=1, outdir=cube_dir, grid=(50, 50, 50))
    if kap0 >= pdmet_obj.solver.ndo_kappa_floor:
        pdmet_obj.get_ndos(state, n_orbs=2, outdir=cube_dir, grid=(50, 50, 50))
    print(f"NTO/NDO cubes -> {cube_dir}")
    return {"state": state, "lam0": lam0, "kap0": kap0, "p": float(ndo["p_A"])}


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

    summaries = {}
    for solver_name in SOLVERS:
        summaries[solver_name] = run_solver(
            cell, kmf, solver_name, scratch_root, cube_root
        )

    # cross-solver canary: same invariants from CASCI (fixed orbitals) and
    # SA-CASSCF (natorb basis) -- a large kappa gap here means the 1pdm files
    # and mo_coeff disagree (noreorder / natorb re-solve issue).
    print("\n" + "=" * 60 + "\n Summary (transition state per solver)\n" + "=" * 60)
    print(f"{'solver':<14} {'state':>5} {'lambda0':>9} {'|kappa0|':>9} {'p':>7}")
    for name, r in summaries.items():
        if r is None:
            print(f"{name:<14}  no transition-like state")
            continue
        print(
            f"{name:<14} {r['state']:>5d} {r['lam0']:>9.3f} {r['kap0']:>9.3f} {r['p']:>7.3f}"
        )


if __name__ == "__main__":
    main()
