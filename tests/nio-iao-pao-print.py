"""
NiO + pDMET + IAO+PAO smoke test.

Exercises:
  * KRHF on NiO primitive cell (GTH pseudopotentials)
  * pDMET with lo_method='iao+pao' (no Wannier90 needed)
  * imp_orbital_filter so only Ni 3d is the impurity
  * pdmet.initialize() prints the LO label table + impurity selection
  * pdmet.plot(orb='lo', ...) writes one XSF file per LO
"""

import os
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf, df

from pdmet import dmet
from pdmet.tools import tchkfile

lib.logger.TIMER_LEVEL = lib.logger.INFO


def build_nio_cell():
    """Rocksalt NiO primitive cell (Ni at origin, O at body-diagonal half)."""
    a = 4.17
    cell = gto.Cell()
    cell.atom = [
        ["Ni", (0.0, 0.0, 0.0)],
        ["O", (0.5 * a, 0.5 * a, 0.5 * a)],
    ]
    cell.a = (
        np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )
        * a
    )
    cell.basis = {"Ni": "gth-dzvp-molopt-sr", "O": "gth-dzvp"}
    cell.pseudo = "gth-pade"
    cell.spin = 0
    cell.unit = "Angstrom"
    cell.verbose = 4
    cell.max_memory = 10000
    cell.build()
    return cell


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    gdf_file = os.path.join(here, "gdf-nio.h5")
    chk_file = os.path.join(here, "chk_HF_nio")

    cell = build_nio_cell()

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)

    # ---- Build GDF (cached on disk) ----------------------------------
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

    # ---- pDMET driver with IAO+PAO basis ------------------------------
    pdmet_obj = dmet.pDMET(
        cell,
        kmf,
        w90=None,
        lo_method="iao+pao",
        solver="CASSCF",
    )
    pdmet_obj.lobasis.minao = {
        "Ni": "gth-szv-molopt-sr",
        "O": "gth-szv",
    }  # IAO reference (minimal)
    pdmet_obj.emb.impCluster = [1]  # Ni atom (1-indexed)
    pdmet_obj.emb.imp_orbital_filter = {"Ni": ["3d"]}  # only Ni 3d as impurity
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (2, 2)
    pdmet_obj.initialize()

    pdmet_obj.one_shot()
    out_dir = os.path.join(here, "nio_lo_xsf")
    os.makedirs(out_dir, exist_ok=True)
    pdmet_obj.plot(orb="lo", grid=[40, 40, 40], path=out_dir, fmt="xsf")

    print(f"\nLO XSF files in: {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
