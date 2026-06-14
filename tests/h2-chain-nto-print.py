"""Periodic H2 chain NTO demo (CASSCF + NEVPT2, IAO+PAO).

The sigma_g -> sigma_u excitation is rank-1: hole = bonding sigma_g (occupied),
particle = antibonding sigma_u (virtual). The script prints the lambda table,
asym = ||T-T^T||/||T|| (~1.41 = real transition, ~0 = symmetric/identical cubes)
and the occupied/virtual character, then writes donor/acceptor cubes. lambda_0~2
(total a+b density); rank-1 means lambda_0/sum ~= 1.

    python tests/h2-chain-nto-print.py [n_units] [state]   # no Wannier90 needed

See h2-chain-nto-solvers.py for the same physics via the nto flag (no NEVPT2)
on CASCI / SA-CASSCF / SA-DMRG.
"""

import os
import sys
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf, df

from pdmet import dmet
from pdmet.tools import tchkfile

lib.logger.TIMER_LEVEL = lib.logger.INFO


def build_h2_chain_cell(n_units, d_hh=1.0, cell_len=3.0, vacuum=20.0):
    """Chain of H2 along x, centered in the y/z vacuum. d_hh=1.0 (bound,
    closed-shell); cell_len-d_hh=2.0 A gap => weak coupling, clean sigma_g/u.
    """
    atoms = []
    for n in range(n_units):
        x0 = n * cell_len
        atoms.append(["H", (x0, 0.0, 0.0)])
        atoms.append(["H", (x0 + d_hh, 0.0, 0.0)])

    # center the chain in the vacuum (y, z) directions so cubes sit mid-box
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
    cell.max_memory = 10000
    cell.build()
    return cell


def transition_asymmetry(pdmet_obj, state):
    """asym = ||T - T^T||_F / ||T||_F for the embedding-basis transition density.
    ~1.41 => real hole != particle transition; ~0 => symmetric (identical cubes).
    """
    T = pdmet_obj.t_dm1s[state]
    nrm = np.linalg.norm(T)
    return float(np.linalg.norm(T - T.T) / nrm) if nrm else 0.0


def occ_virt_character(pdmet_obj, vec):
    """(occupied, virtual) fraction of an EO-basis orbital vs the MF MOs.
    hole(sigma_g) -> mostly occupied; particle(sigma_u) -> mostly virtual.
    """
    mf = pdmet_obj.qcsolver.mf
    C = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ) > 0
    norm2 = float(np.vdot(vec, vec).real)

    def frac(C_sub):
        proj = C_sub.conj().T @ vec
        return float(np.vdot(proj, proj).real) / norm2

    return frac(C[:, occ]), frac(C[:, ~occ])


def excited_state_diagnostics(pdmet_obj, floor=1e-3):
    """Per excited state: (state, top_lambda, sum_lambda, asym) or asym=None."""
    rows = []
    for state in range(1, len(pdmet_obj.ntos_per_root)):
        lam = pdmet_obj.ntos_per_root[state]["lambdas"]
        if not len(lam) or lam[0] < floor:
            rows.append((state, float(lam[0]) if len(lam) else 0.0, 0.0, None))
            continue
        rows.append(
            (
                state,
                float(lam[0]),
                float(lam.sum()),
                transition_asymmetry(pdmet_obj, state),
            )
        )
    return rows


def pick_sigma_state(pdmet_obj, floor=1e-3, min_asym=0.7):
    """Most asymmetric (transition-like) excited state = the sigma_g -> sigma_u."""
    rows = [r for r in excited_state_diagnostics(pdmet_obj, floor) if r[3] is not None]
    if not rows:
        return None, False
    rows.sort(key=lambda r: -r[3])
    state, _, _, asym = rows[0]
    return state, (asym > min_asym)


def main():
    n_units = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    forced = int(sys.argv[2]) if len(sys.argv) > 2 else None

    here = os.path.dirname(os.path.abspath(__file__))
    tag = f"h2chain{n_units}"
    gdf_file = os.path.join(here, f"gdf-{tag}.h5")
    chk_file = os.path.join(here, f"chk_HF_{tag}")
    cube_dir = os.path.join(here, f"{tag}_nto_cubes")

    cell = build_h2_chain_cell(n_units)
    n_active = 2 * n_units
    print(
        f"\nH2 chain: {n_units} molecule(s), CAS = ({n_active},{n_active}) "
        f"(sigma_g / sigma_u space)\n"
    )

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)

    if not os.path.exists(gdf_file):
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_file
        gdf.build()

    khf = scf.KRHF(cell, kpts).density_fit()
    khf.with_df._cderi = gdf_file
    khf.exxdiv = None
    khf.max_cycle = 80
    khf.run()
    print(f"\nKRHF energy = {khf.e_tot:.8f}\n")

    tchkfile.save_kmf(khf, chk_file)
    kmf = tchkfile.load_kmf(khf, chk_file)

    # ---- pDMET + CASSCF + NEVPT2 on the sigma space ------------------------
    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="CASSCF")
    pdmet_obj.lobasis.minao = {"H": "gth-szv"}
    pdmet_obj.emb.impCluster = list(range(1, 2 * n_units + 1))  # all H atoms
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (n_active, n_active)
    pdmet_obj.solver.e_shift = 0.2  # keep singlets (push the triplet up)

    n_roots = min(4, n_active + 1)
    pdmet_obj.solver.nevpt2_roots = list(range(n_roots))
    pdmet_obj.solver.nevpt2_nroots = n_roots
    pdmet_obj.solver.nroots = 1
    pdmet_obj.solver.nto = True  # compute NTOs (implied by nevpt2_roots, explicit here)
    pdmet_obj.solver.nto_npairs = 1  # sigma_g -> sigma_u is a single pair
    pdmet_obj.solver.nto_lambda_floor = 1e-3

    pdmet_obj.initialize()
    pdmet_obj.one_shot()

    # ---- NTO report --------------------------------------------------------
    print("\n" + "=" * 64)
    print("Natural Transition Orbitals (ground -> excited roots)")
    print("=" * 64)
    for state in range(len(pdmet_obj.ntos_per_root)):
        pdmet_obj.get_ntos(state, outdir=None)  # lambda table per state

    print(
        "\nExcited-state character "
        "(a real sigma_g->sigma_u has asym~1.41, rank-1, hole occ / part virt):"
    )
    print(
        "  state  top_lambda  sum_lambda   asym    rank1   hole(occ/virt)  part(occ/virt)"
    )
    for s, top, ssum, asym in excited_state_diagnostics(pdmet_obj):
        if asym is None:
            print(f"  {s:3d}    {top:9.4f}    (below floor)")
            continue
        info = pdmet_obj.ntos_per_root[s]
        rank1 = top / ssum
        ho, hv = occ_virt_character(pdmet_obj, info["V_hole"][:, 0])
        po, pv = occ_virt_character(pdmet_obj, info["U_part"][:, 0])
        print(
            f"  {s:3d}    {top:9.4f}   {ssum:9.4f}  {asym:6.3f}  {rank1:5.2f}   "
            f"{ho:4.2f}/{hv:4.2f}        {po:4.2f}/{pv:4.2f}"
        )

    state = forced if forced is not None else pick_sigma_state(pdmet_obj)[0]
    if state is None:
        print(
            "\nNo transition-like excited state found -- try a different bond "
            "length / spacing, more roots, or a longer chain."
        )
        return

    info = pdmet_obj.ntos_per_root[state]
    lam = info["lambdas"]
    ho, hv = occ_virt_character(pdmet_obj, info["V_hole"][:, 0])
    po, pv = occ_virt_character(pdmet_obj, info["U_part"][:, 0])
    print(
        f"\nChosen state {state}: lambda_0 = {lam[0]:.4f}, sum = {lam.sum():.4f}, "
        f"rank-1 fraction = {lam[0] / lam.sum():.3f}, asym = "
        f"{transition_asymmetry(pdmet_obj, state):.3f}"
    )
    print(f"  hole NTO  : {ho * 100:5.1f}% occupied  -> bonding sigma_g  (expected)")
    print(f"  particle  : {pv * 100:5.1f}% virtual   -> antibonding sigma_u (expected)")

    print(f"\nWriting donor/acceptor cubes for state {state} -> {cube_dir}")
    pdmet_obj.get_ntos(state, n_pairs=1, outdir=cube_dir, grid=(60, 60, 60))

    print(f"\nDone. Open the .cube files in {cube_dir} with VMD / Avogadro:")
    print("  ..._donor_...    = hole     = bonding sigma_g  (no node between the H)")
    print(
        "  ..._acceptor_... = particle = antibonding sigma_u (one node between the H)"
    )


if __name__ == "__main__":
    main()
