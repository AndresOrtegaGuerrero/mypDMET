import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pdmet.tools.optional import to_numpy


HARTREE2EV = 27.211386245988


def find_ao_indices(cell, element: str, orbital_prefix: str) -> list[int]:
    """Return AO indices on `element` whose label starts with `orbital_prefix`
    orbital_prefix='3d' catches 3dxy, 3dxz , 3dyz, 3dz^2, 3dx2-y2, 3dz2
    Args:
        cell             : pbc.gto.Cell
        element          : 'Fe', 'Mo', 'S',
        orbital_prefix   : '3d', '4d', '3p', '2px',

    Returns:
        list of integer AO indices
    """
    indices = []
    for i, lbl in enumerate(cell.ao_labels()):
        parts = lbl.split()
        if len(parts) < 3:
            continue
        if parts[1] == element and parts[2].startswith(orbital_prefix):
            indices.append(i)
    return indices


# Projecting MOs onto AOs to analyze their caracter
# Example usage after a converged SCF:
#
# from pdmet.tools.mo_projection_plot import (
#     find_ao_indices, plot_mo_projections)
#
# projections = {
#     'S 3p':  find_ao_indices(cell, 'S',  '3p'),
#     'Mo 4d': find_ao_indices(cell, 'Mo', '4d'),
#     'Fe 3d': find_ao_indices(cell, 'Fe', '3d'),
# }
# plot_mo_projections(
#     kmf, projections,
#     kpt_idx=0,
#     energy_window_eV=(-20, 10),
#     title="Fe-MoS2 SCF MO projections",
#     savepath="mo_projections.png",
# )
def plot_mo_projections(
    kmf,
    projections: dict,
    *,
    kpt_idx: int = 0,
    energy_window_eV: tuple | None = None,
    fermi_eV: float | None = None,
    title: str | None = None,
    savepath: str | None = None,
    figsize: tuple | None = None,
    show: bool = False,
):
    """Plot MO energies coloured by projection weight onto chosen AO subsets.

    Args:
        kmf : converged SCF object. Reads mo_coeff_kpts, mo_energy_kpts,
              mo_occ_kpts, get_ovlp(). Works with CPU pyscf KROHF,
              gpu4pyscf KROHF, or a fake_kmf from tchkfile.load_kmf
        projections : dict mapping label -> list of AO indices
              e.g. {'S 3p': s_3p_idx, 'Mo 4d': mo_4d_idx, 'Fe 3d': fe_3d_idx}
              Empty lists are silently skipped.
        kpt_idx : which k-point to plot
        energy_window_eV : (emin, emax) tuple in eV to clip the y-axis,
              or None for auto.
        fermi_eV : draw a horizontal reference line at this energy.
              If None, defaults to the HOMO energy.
        title : optional plot title.
        savepath : if given, save the figure to this path. PDF / PNG / SVG
              by extension.
        figsize : matplotlib figsize; default scales with n_panels.
        show : if True, plt.show() at the end.

    Returns:
        matplotlib.figure.Figure
    """
    # Compatible if GPU4PySC is Used
    C = to_numpy(kmf.mo_coeff_kpts[kpt_idx])
    S = to_numpy(kmf.get_ovlp()[kpt_idx])
    e = to_numpy(kmf.mo_energy_kpts[kpt_idx])
    oc = to_numpy(kmf.mo_occ_kpts[kpt_idx])

    e_eV = np.real(e) * HARTREE2EV

    # Mulliken weights w[ao, mo] = Re(C* . S C)
    SC = S @ C
    w = np.real(np.conj(C) * SC)

    proj_weights = {}
    for label, idx in projections.items():
        if not idx:
            continue
        proj_weights[label] = w[idx, :].sum(axis=0)

    n_proj = len(proj_weights)
    n_panels = 1 + n_proj

    if figsize is None:
        figsize = (1.6 * n_panels + 1, 7.5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]

    # HOMO / LUMO for reference
    occ_mask = oc > 0
    homo_eV = float(e_eV[occ_mask].max()) if occ_mask.any() else None
    lumo_eV = float(e_eV[~occ_mask].min()) if (~occ_mask).any() else None
    if fermi_eV is None and homo_eV is not None:
        fermi_eV = homo_eV

    # All MOs levels
    ax0 = axes[0]
    for ei, oi in zip(e_eV, oc):
        color, alpha = ("k", 1.0) if oi > 0 else ("gray", 0.4)
        ax0.axhline(y=ei, xmin=0.15, xmax=0.85, color=color, alpha=alpha, linewidth=0.7)
    if homo_eV is not None:
        ax0.axhline(
            homo_eV, color="C3", linestyle="--", linewidth=0.8, alpha=0.6, label="HOMO"
        )
    if lumo_eV is not None:
        ax0.axhline(
            lumo_eV, color="C0", linestyle="--", linewidth=0.8, alpha=0.6, label="LUMO"
        )
    ax0.set_title("All MOs", fontsize=10)
    ax0.set_xticks([])
    ax0.set_ylabel("Energy (eV)")
    ax0.legend(loc="lower right", fontsize=7, framealpha=0.7)

    # One panel per projection
    cmap = plt.get_cmap("viridis")
    for ax, (label, weights) in zip(axes[1:], proj_weights.items()):
        # clip weights to [0,1] for color/length (some basis sets give
        # tiny negative Mulliken weights from non-orthogonal AOs)
        wc = np.clip(weights, 0.0, 1.0)

        segments = []
        colors = []
        for ei, oi, wi in zip(e_eV, oc, wc):
            if wi < 0.01:
                continue
            x0, x1 = 0.0, wi
            segments.append([(x0, ei), (x1, ei)])
            rgba = list(cmap(wi))
            rgba[-1] = 1.0 if oi > 0 else 0.4
            colors.append(rgba)

        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=1.4)
            ax.add_collection(lc)
            ax.set_xlim(0, 1.05)
        else:
            ax.text(
                0.5,
                0.5,
                "(no weight)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="gray",
            )

        if fermi_eV is not None:
            ax.axhline(fermi_eV, color="C3", linestyle="--", linewidth=0.6, alpha=0.4)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("weight")
        ax.set_xticks([0, 0.5, 1.0])

    if energy_window_eV is not None:
        for ax in axes:
            ax.set_ylim(*energy_window_eV)
    else:
        if homo_eV is not None and lumo_eV is not None:
            pad = max(2.0, 0.15 * (lumo_eV - homo_eV))
            ax0.set_ylim(homo_eV - 6 * pad, lumo_eV + 6 * pad)

    if title:
        fig.suptitle(title, fontsize=11, y=1.00)

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"  [plot_mo_projections] saved -> {savepath}")
    if show:
        plt.show()
    return fig
