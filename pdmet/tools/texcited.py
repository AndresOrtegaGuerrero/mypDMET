"""Excited-state analysis tools for pDMET: NTOs and NDOs.

Supports CASSCF, CASCI, and DMRG wavefunctions.
"""

import os
import numpy as np
from pdmet.tools import tplot


def _require(pdmet, attr, flag_hint):
    """Return a computed analysis attribute or raise a helpful error."""
    val = getattr(pdmet, attr, None)

    if val is None:
        raise ValueError(
            f"{attr} is None -- nothing was computed. "
            f"Set {flag_hint} before running the calculation."
        )

    return val


def _eo_to_lo(pdmet):
    """Return the Gamma-cell EO -> LO transformation matrix."""
    if pdmet.emb_orbs is None:
        raise ValueError(
            "emb_orbs not built yet -- call one_shot() before excited-state analysis."
        )

    return pdmet.emb_orbs[0]


def _check_gamma(pdmet, what):
    """Cube export is Gamma-only; tables work at any k-mesh."""
    if pdmet.local.Nkpts > 1:
        raise NotImplementedError(
            f"{what} cube export is currently Gamma-only "
            f"(Nkpts={pdmet.local.Nkpts}). The weights and embedding-basis "
            f"coefficients are available on the *_per_root attributes for any "
            f"k-mesh; only the real-space cube generation is restricted. Use "
            f"kmesh=[1,1,1] or implement supercell plotting."
        )


def _dump_cubes(pdmet, rotate_mat, filenames, outdir, grid, fmt, tag):
    """Shared cube writer: plot every column of `rotate_mat` (LO basis) with
    tplot.plot_wf, then rename {prefix}-{col}.{fmt} to `filenames[col]`.
    """
    os.makedirs(outdir, exist_ok=True)
    tmp_prefix = os.path.join(outdir, f"_{tag}_tmp")
    tplot.plot_wf(
        pdmet.local,
        rotate_mat,
        tmp_prefix,
        supercell=pdmet.kmesh,
        grid=grid,
        fmt=fmt,
    )
    for col, name in enumerate(filenames):
        src = f"{tmp_prefix}-{col}.{fmt}"
        dst = os.path.join(outdir, f"{name}.{fmt}")
        if os.path.exists(src):
            os.replace(src, dst)
            print(f"  [{tag}] wrote {dst}")


def rotate_mat_nto(pdmet, state=0, n_pairs=None):
    """LO-basis rotation matrix (nlo, 2*n_kept): interleaved donor, acceptor,
    ... for the top n_pairs of `state`. NTOs live in the EO basis.
    """
    infos = _require(pdmet, "ntos_per_root", "solver.nto=True")
    info = infos[state]
    n = n_pairs if n_pairs is not None else pdmet.solver.nto_npairs
    n_kept = min(n, len(info["lambdas"]))
    assert n_kept > 0, f"state {state} has no NTO pairs above threshold."

    cols = []
    for i in range(n_kept):
        cols.append(info["V_hole"][:, i])  # donor
        cols.append(info["U_part"][:, i])  # acceptor
    return _eo_to_lo(pdmet) @ np.column_stack(cols)


def rotate_mat_ndo(pdmet, state=0, n_orbs=None):
    """LO-basis rotation matrix (nlo, n_kept): top-|kappa| NDOs of `state`.
    Columns are ordered by |kappa| descending; the SIGN of kappa (detachment
    vs attachment) is encoded in the table / cube file names, not the order.
    """
    infos = _require(pdmet, "ndos_per_root", "solver.ndo=True")
    info = infos[state]
    n = n_orbs if n_orbs is not None else pdmet.solver.ndo_npairs
    n_kept = min(n, len(info["kappa"]))
    assert n_kept > 0, (
        f"state {state} has no NDOs above threshold "
        f"(state 0 is the ground state: Delta^00 = 0)."
    )
    return _eo_to_lo(pdmet) @ info["W"][:, :n_kept]


def _cas_weights(vec_cas, top=3, tol=0.05):
    w = np.abs(vec_cas) ** 2
    tot = w.sum()
    if tot == 0.0:
        return "--"
    w = w / tot
    idx = np.argsort(-w)[:top]
    parts = [f"{w[i]:.2f} cas_{i:02d}" for i in idx if w[i] >= tol]
    return " + ".join(parts) if parts else "(delocalized)"


def _print_nto_table(state, info, n_kept, floor):
    lam = info["lambdas"]
    total = lam.sum()
    n_total = len(lam)

    if state == 0:
        print(
            "\nState 0 = ground density (T^00 = gamma_0), so lambda_i = n_i^2. "
            "Not a transition."
        )

    print(f"\nNTOs for state {state}  (top {n_kept} of {n_total} nonzero pairs)")
    print(f"  sum(lambda) = {total:.4f}   [~2 = clean single excitation]")
    print("  pair       lambda     % of transition")
    for i in range(n_kept):
        pct = lam[i] / total * 100 if total > 0 else 0.0
        print(f"  {i:3d}     {lam[i]:10.5f}     {pct:6.2f}%")
        if info.get("V_hole_cas") is not None:
            print(f"            hole: {_cas_weights(info['V_hole_cas'][:, i])}")
            print(f"            part: {_cas_weights(info['U_part_cas'][:, i])}")

    if n_kept > 1 and abs(lam[0] - lam[1]) < 0.05 * abs(lam[0]):
        print(
            "  [near-degenerate lambdas: pairs defined only up to a rotation in "
            "that subspace]"
        )
    if n_kept < n_total:
        print(
            f"  ...     (skipped {n_total - n_kept} below floor={floor:.1e} or "
            f"beyond n_pairs)"
        )
    if total > 0 and n_kept > 0:
        print(f"  sum(lambda) kept / all = {lam[:n_kept].sum() / total * 100:5.1f}%")


def get_ntos(
    pdmet,
    state=0,
    n_pairs=None,
    lambda_floor=None,
    outdir=None,
    grid=(50, 50, 50),
    fmt="cube",
):
    """Print the lambda table for NTO `state` and return its top pairs;
    if `outdir` is given (Gamma-only), also write donor/acceptor cubes.

    Returns a list of {'pair', 'lambda', 'donor', 'acceptor'} (EO basis).
    """
    infos = _require(pdmet, "ntos_per_root", "solver.nto=True")

    n = n_pairs if n_pairs is not None else pdmet.solver.nto_npairs
    floor = lambda_floor if lambda_floor is not None else pdmet.solver.nto_lambda_floor

    info = infos[state]
    lam = info["lambdas"]
    n_kept = min(n, int(np.sum(lam >= floor)))

    _print_nto_table(state, info, n_kept, floor)

    if outdir is not None:
        _check_gamma(pdmet, "NTO")
        if n_kept > 0:
            names = []
            for i in range(n_kept):
                for role in ("donor", "acceptor"):
                    names.append(f"state{state}_pair{i}_{role}_lam{lam[i]:.3f}")
            _dump_cubes(
                pdmet,
                rotate_mat_nto(pdmet, state=state, n_pairs=n_kept),
                names,
                outdir,
                grid,
                fmt,
                tag=f"nto_state{state}",
            )

    return [
        {
            "pair": i,
            "lambda": float(lam[i]),
            "donor": info["V_hole"][:, i],
            "acceptor": info["U_part"][:, i],
        }
        for i in range(n_kept)
    ]


def _print_ndo_table(state, info, n_kept, floor):
    kappa = info["kappa"]
    n_total = len(kappa)
    pr_d = f"{info['PR_D']:.2f}" if info["PR_D"] else "--"
    pr_a = f"{info['PR_A']:.2f}" if info["PR_A"] else "--"

    print(f"\nNDOs for state {state}  (top {n_kept} of {n_total} nonzero)")
    print(f"  p = {info['p_A']:.4f} electrons promoted  [~1 single, ~2 double]")
    print(f"  PR_D = {pr_d}  PR_A = {pr_a}   [orbitals sharing each side]")
    print("  [kappa < 0: detachment (D);  kappa > 0: attachment (A)]")
    print("  orb        kappa      type    % of side")
    for i in range(n_kept):
        k = kappa[i]
        role = "D" if k < 0 else "A"
        side = abs(info["p_D"]) if k < 0 else info["p_A"]
        pct = abs(k) / side * 100 if side > 0 else 0.0
        print(f"  {i:3d}     {k:10.5f}      {role}      {pct:6.2f}%")
        if info.get("W_cas") is not None:
            print(f"            {_cas_weights(info['W_cas'][:, i])}")
    if n_kept < n_total:
        print(
            f"  ...     (skipped {n_total - n_kept} below floor={floor:.1e} "
            f"or beyond n_orbs)"
        )


def get_ndos(
    pdmet,
    state=0,
    n_orbs=None,
    kappa_floor=None,
    outdir=None,
    grid=(50, 50, 50),
    fmt="cube",
):
    """Print the kappa table for NDO `state` and return its top orbitals;
    if `outdir` is given (Gamma-only), also write detachment/attachment cubes.

    Returns a list of {'orb', 'kappa', 'type', 'coeff'} (EO basis;
    type 'D' = detachment (kappa<0), 'A' = attachment (kappa>0)).
    """
    infos = _require(pdmet, "ndos_per_root", "solver.ndo=True")

    n = n_orbs if n_orbs is not None else pdmet.solver.ndo_npairs
    floor = kappa_floor if kappa_floor is not None else pdmet.solver.ndo_kappa_floor

    info = infos[state]
    kappa = info["kappa"]
    n_kept = min(n, int(np.sum(np.abs(kappa) >= floor)))

    _print_ndo_table(state, info, n_kept, floor)

    if outdir is not None:
        _check_gamma(pdmet, "NDO")
        if n_kept > 0:
            names = [
                f"state{state}_ndo{i}_"
                f"{'det' if kappa[i] < 0 else 'att'}_kap{kappa[i]:+.3f}"
                for i in range(n_kept)
            ]
            _dump_cubes(
                pdmet,
                rotate_mat_ndo(pdmet, state=state, n_orbs=n_kept),
                names,
                outdir,
                grid,
                fmt,
                tag=f"ndo_state{state}",
            )

    return [
        {
            "orb": i,
            "kappa": float(kappa[i]),
            "type": "D" if kappa[i] < 0 else "A",
            "coeff": info["W"][:, i],
        }
        for i in range(n_kept)
    ]


# --------------------------------------------------------------------------- #
#  Densities and observables                                                  #
# --------------------------------------------------------------------------- #


def get_trans_dipole(pdmet):
    """Transition dipoles <0|r|I> from the transported transition densities."""
    t_dm1s = _require(pdmet, "t_dm1s", "solver.nto=True")

    charges = pdmet.cell.atom_charges()
    coords = pdmet.cell.atom_coords()
    nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
    pdmet.cell.set_common_orig_(nuc_charge_center)
    dip_ints = pdmet.cell.intor("cint1e_r_sph", comp=3)
    ao2eo = pdmet.local.get_ao2eo(pdmet.emb_orbs)[0]

    dipoles = []
    for i, t_dm1_emb in enumerate(t_dm1s):
        t_dm1_ao = ao2eo @ t_dm1_emb @ ao2eo.T.conj()
        dip = np.einsum("xij,ji->x", dip_ints, t_dm1_ao).real
        dipoles.append(dip)
        print(
            "Transition dipole between |0> and |{0:d}>: {1:3.5f} {2:3.5f} "
            "{3:3.5f} | Norm: {4:3.5f}".format(i, *dip, np.linalg.norm(dip))
        )
    return dipoles


def get_attach_detach_density(pdmet, state):
    """Attachment/detachment densities of `state` in the AO basis (Eqs. 71/73
    of JCP 141, 024106), ready for density plotting or Mulliken partitioning.

    Returns (D_det_ao, D_att_ao); their traces are (p_D, p_A).
    """
    infos = _require(pdmet, "ndos_per_root", "solver.ndo=True")
    info = infos[state]
    assert info["D_det"] is not None, f"state {state}: empty NDO entry."

    ao2eo = pdmet.local.get_ao2eo(pdmet.emb_orbs)[0]

    def to_ao(M):
        return ao2eo @ M @ ao2eo.T.conj()

    return to_ao(info["D_det"]), to_ao(info["D_att"])


# --------------------------------------------------------------------------- #
#  Auto-export (end of one_shot()/run())                                      #
# --------------------------------------------------------------------------- #


def auto_export(pdmet):
    """Emit NTO and/or NDO tables + cubes as requested by settings.*_export.

    Tables print at any k-mesh. Cubes are Gamma-only; at k>1 we skip them
    with a note instead of aborting an already-expensive run.
    """
    gamma = pdmet.local.Nkpts == 1
    base = getattr(pdmet, "outdir", ".")

    jobs = []
    if pdmet.solver.nto_export and getattr(pdmet, "ntos_per_root", None) is not None:
        jobs.append(("nto", pdmet.ntos_per_root, get_ntos))
    if pdmet.solver.ndo_export and getattr(pdmet, "ndos_per_root", None) is not None:
        jobs.append(("ndo", pdmet.ndos_per_root, get_ndos))

    for tag, infos, getter in jobs:
        outdir = os.path.join(base, f"{tag}s") if gamma else None
        if not gamma:
            print(
                f"[{tag}_export] Nkpts={pdmet.local.Nkpts} > 1: printing "
                f"tables only; cube export is Gamma-only."
            )
        for state in range(len(infos)):
            getter(pdmet, state, outdir=outdir)
