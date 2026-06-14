"""Trans-polyacetylene pi->pi* NTO demo (CASSCF + NEVPT2, IAO+PAO).

A Peierls-DIMERIZED N-unit chain (alternating short/long C-C, so a real pi gap)
at Gamma is a 2N-carbon polyene: the lowest bright excitation is ~rank-1 HOMO
(pi) -> LUMO (pi*). The script prints lambda + asym per state and writes
donor/acceptor cubes for the most asymmetric (real transition) state. asym~0 with
two equal lambdas means a symmetric/biradical density (undimerized chain).

    python tests/tpa-nto-print.py [n_units] [state]   # no Wannier90 needed
"""

import os
import sys
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf, df

from pdmet import dmet
from pdmet.tools import tchkfile

lib.logger.TIMER_LEVEL = lib.logger.INFO

# --- trans-polyacetylene geometry: PROPERLY DIMERIZED (Peierls) ------------- #
# Real trans-PA alternates short C=C (~1.36 A) and long C-C (~1.44 A) bonds. That
# bond alternation OPENS the pi gap. A uniform chain (all bonds equal) is
# metallic -> degenerate frontier -> biradical ground state (occupancies near
# [.., 1.0, 1.0, ..]) -> symmetric transition densities -> donor == acceptor.
# Alternation is therefore essential for a clean pi->pi* NTO.
_D_DOUBLE = 1.36  # C=C (short)
_D_SINGLE = 1.44  # C-C (long)
_CCC = np.deg2rad(124.5)  # C-C-C backbone angle
_CH = 1.09


def _tpa_geom():
    """Planar zigzag with ALTERNATING bonds -> (h, p, q, a).

    Carbons sit on two y-levels +/- h/2. The double bond advances x by p, the
    single bond by q, so the repeat length is a = p + q. Solved from the two bond
    lengths and the C-C-C angle (closed form; verified numerically).
    """
    d1, d2 = _D_DOUBLE, _D_SINGLE
    c = d1 * d2 * np.cos(_CCC)
    h2 = (d1**2 * d2**2 - c**2) / (d1**2 + d2**2 - 2 * c)
    h = np.sqrt(h2)
    p = np.sqrt(d1**2 - h2)  # x-advance of the double bond
    q = np.sqrt(d2**2 - h2)  # x-advance of the single bond
    return h, p, q, p + q


_H, _P, _Q, A_CHAIN = _tpa_geom()  # A_CHAIN ~ 2.478 A


def build_tpa_cell(n_units, vacuum=20.0):
    """Build an n_units-long trans-PA supercell, periodic along x.

    x is the periodic chain axis (left as-is so the chain tiles seamlessly).
    y and z are vacuum directions, so the molecule is shifted to sit in the
    MIDDLE of the box there -- otherwise it hugs the y=0, z=0 corner and the
    NTO cubes render off to one side.
    """
    h, p, _, a = _tpa_geom()
    yU, yL = 0.5 * h, -0.5 * h
    atoms = []
    for n in range(n_units):
        x0 = n * a
        # Upper-sublattice C (H up) and lower-sublattice C (H down). The
        # intra-cell C-C (advance p) is the DOUBLE bond; the bond to the next
        # cell (advance q) is the SINGLE bond -> alternation along the chain.
        atoms.append(["C", (x0, yU, 0.0)])
        atoms.append(["C", (x0 + p, yL, 0.0)])
        atoms.append(["H", (x0, yU + _CH, 0.0)])
        atoms.append(["H", (x0 + p, yL - _CH, 0.0)])

    # Center the molecule in the vacuum (y, z) directions; leave periodic x.
    ys = [a[1][1] for a in atoms]
    zs = [a[1][2] for a in atoms]
    y_shift = 0.5 * vacuum - 0.5 * (min(ys) + max(ys))
    z_shift = 0.5 * vacuum - 0.5 * (min(zs) + max(zs))
    atoms = [[sym, (x, y + y_shift, z + z_shift)] for sym, (x, y, z) in atoms]

    cell = gto.Cell()
    cell.atom = atoms
    cell.a = np.diag([n_units * A_CHAIN, vacuum, vacuum])
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.spin = 0
    cell.unit = "Angstrom"
    cell.verbose = 4
    cell.max_memory = 10000
    cell.build()
    return cell


def transition_asymmetry(pdmet_obj, state):
    """asym = ||T - T^T||_F / ||T||_F  for the embedding-basis transition density.

    This is THE robust test for "is this a real hole != particle excitation?".
    The NTO SVD only earns its keep when T is non-symmetric (hole and particle
    live in different orbitals). Then:

        asym ~ sqrt(2) ~ 1.41  => clean transition (rank-1-ish |h><l|, h != l);
                                  donor and acceptor are DIFFERENT orbitals.
        asym ~ 0               => SYMMETRIC T (ground-state density, or a dark /
                                  "breathing" excitation like polyene 2Ag); the
                                  donor and acceptor orbital sets COINCIDE, so
                                  the cubes look identical.

    Why not the donor/acceptor pairwise overlap? When singular values are
    (near-)degenerate the SVD returns an arbitrary basis of that subspace, so the
    per-pair overlap is meaningless -- a symmetric T can even report overlap 0.
    The asymmetry of T is basis-free and unambiguous.
    """
    T = pdmet_obj.t_dm1s[state]
    nrm = np.linalg.norm(T)
    return float(np.linalg.norm(T - T.T) / nrm) if nrm else 0.0


def excited_state_diagnostics(pdmet_obj, floor=1e-3):
    """Per excited state: (state, top_lambda, sum_lambda, asym) or asym=None."""
    rows = []
    for state in range(1, len(pdmet_obj.ntos_per_root)):
        lam = pdmet_obj.ntos_per_root[state]["lambdas"]
        if not len(lam) or lam[0] < floor:
            rows.append((state, float(lam[0]) if len(lam) else 0.0, 0.0, None))
            continue
        asym = transition_asymmetry(pdmet_obj, state)
        rows.append((state, float(lam[0]), float(lam.sum()), asym))
    return rows


def pick_charge_transfer_state(pdmet_obj, floor=1e-3, min_asym=0.7):
    """Choose the excited state with the most asymmetric (cleanest) transition.

    A genuine pi->pi* has high asym (~1.41). We take the highest-asym above-floor
    state and flag whether it really looks like a transition (asym > min_asym).
    """
    rows = [r for r in excited_state_diagnostics(pdmet_obj, floor) if r[3] is not None]
    if not rows:
        return None, False
    rows.sort(key=lambda r: -r[3])  # most asymmetric (most transition-like) first
    state, _, _, asym = rows[0]
    return state, (asym > min_asym)


def main():
    n_units = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    here = os.path.dirname(os.path.abspath(__file__))
    tag = f"tpa{n_units}d"  # 'd' = dimerized geometry (avoids reusing old caches)
    gdf_file = os.path.join(here, f"gdf-{tag}.h5")
    chk_file = os.path.join(here, f"chk_HF_{tag}")
    cube_dir = os.path.join(here, f"{tag}_nto_cubes")

    cell = build_tpa_cell(n_units)
    n_carbon = 2 * n_units
    print(
        f"\ntrans-PA: {n_units} units, {n_carbon} carbons "
        f"(pi active space target = ({n_carbon},{n_carbon}))\n"
    )

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)

    # ---- GDF (cached on disk) ----------------------------------------------
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

    # ---- pDMET + CASSCF + NEVPT2 on the pi system (same recipe as the H2 demo)
    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="CASSCF")
    pdmet_obj.lobasis.minao = {"C": "gth-szv", "H": "gth-szv"}
    pdmet_obj.emb.impCluster = list(range(1, 4 * n_units + 1))  # whole chain
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (n_carbon, n_carbon)  # full pi space
    pdmet_obj.solver.e_shift = 0.2  # spin penalty -> keep singlets

    # Ground-optimized orbitals (nroots=1), then several CASCI/NEVPT2 roots so
    # the BRIGHT pi->pi* (1Bu) is in range, not just the dark 2Ag. The asym
    # picker below selects the genuine hole != particle transition.
    n_roots = min(5, n_carbon + 1)
    pdmet_obj.solver.nevpt2_roots = list(range(n_roots))
    pdmet_obj.solver.nevpt2_nroots = n_roots
    pdmet_obj.solver.nroots = 1

    pdmet_obj.solver.nto = True
    pdmet_obj.solver.nto_npairs = 2
    pdmet_obj.solver.nto_lambda_floor = 1e-3

    pdmet_obj.initialize()
    pdmet_obj.one_shot()

    # ---- NTO report --------------------------------------------------------
    print("\n" + "=" * 60)
    print("Natural Transition Orbitals (ground -> excited roots)")
    print("=" * 60)
    for state in range(len(pdmet_obj.ntos_per_root)):
        pdmet_obj.get_ntos(state, outdir=None)  # lambda table only

    # Diagnostic: top weight AND transition asymmetry per excited state.
    # asym ~1.41 => real hole!=particle excitation (donor/acceptor cubes differ);
    # asym ~0    => symmetric transition, donor/acceptor sets coincide (identical).
    print("\nExcited-state character (a real pi->pi* has asym near 1.41):")
    print("  state   top_lambda   sum_lambda   asym=||T-T^T||/||T||   character")
    for s, top, ssum, asym in excited_state_diagnostics(pdmet_obj):
        if asym is None:
            print(f"  {s:3d}     {top:9.4f}    (below floor)")
            continue
        kind = (
            "transition (hole != particle)"
            if asym > 0.7
            else "symmetric (identical cubes)"
        )
        print(
            f"  {s:3d}     {top:9.4f}   {ssum:9.4f}        {asym:7.4f}           {kind}"
        )

    # state override: `python tests/tpa-nto-print.py <n_units> <state>`
    forced = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if forced is not None:
        state = forced
        is_ct = transition_asymmetry(pdmet_obj, forced) > 0.7
    else:
        state, is_ct = pick_charge_transfer_state(pdmet_obj)

    if state is None:
        print(
            "\nNo excited state cleared the lambda floor -- try more units "
            "(python tests/tpa-nto-print.py 3) or more roots."
        )
        return

    info = pdmet_obj.ntos_per_root[state]
    lam = info["lambdas"]
    asym = transition_asymmetry(pdmet_obj, state)
    print(
        f"\nChosen state {state}: top lambda = {lam[0]:.4f}, "
        f"sum lambda = {lam.sum():.4f}, asym = {asym:.4f}"
    )
    if not is_ct:
        print(
            "  WARNING: no asymmetric (transition-like) state found -- every "
            "computed root has a near-symmetric transition density, so donor "
            "and acceptor cubes WILL look identical. This usually means the "
            "active space is NOT the clean pi system (frontier selection "
            "grabbed sigma/mixed orbitals) or the low roots are dark states. "
            "Fix: pin the pi orbitals with pdmet.solver.molist=[...], use more "
            "roots, or a longer chain. Paste the CI/orbital analysis and I can "
            "tell you which indices to use."
        )

    print(f"Writing donor/acceptor cubes for state {state} -> {cube_dir}")
    pdmet_obj.get_ntos(state, n_pairs=2, outdir=cube_dir, grid=(60, 60, 60))

    print(f"\nDone. Open the .cube files in {cube_dir} with VMD / Avogadro.")
    print("  donor    = where the electron leaves  (pi  HOMO for a clean CT)")
    print("  acceptor = where the electron arrives (pi* LUMO for a clean CT)")
    print("  If the two cubes look identical, see the WARNING / overlap above.")


if __name__ == "__main__":
    main()
