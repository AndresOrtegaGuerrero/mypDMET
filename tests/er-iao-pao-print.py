"""
Erbium + IAO+PAO smoke test for *non-built-in* minao.

Why this exists
---------------
For lanthanides (Er, Z=68 here), PySCF does not ship a `gth-szv-molopt-sr`
file out of the box. The IAO reference basis ("minao") therefore has to
be passed as a per-element dict — exactly the same shape PySCF accepts
for `cell.basis`. This test exercises that path with one Er atom in a
big vacuum box and writes out the LOs as XSF.

Layout of dict-form minao
-------------------------
    minao = {
        "Er": gto.basis.parse(er_szv_str),   # custom (no built-in)
        "S":  "gth-szv-molopt-sr",           # built-in (if you had S)
        "Mo": "gth-szv-molopt-sr",           # built-in (if you had Mo)
    }

Replace `er_szv_str` with the real CP2K BASIS_MOLOPT SZV-MOLOPT-SR-GTH-q22
block for Er — this script ships only a placeholder.
"""

import os
import numpy as np
from pyscf.pbc import gto, scf

from pdmet.tools import tchkfile

from pdmet import dmet
from pdmet.localbasis import make_iao_pao_kbasis


# ---- Er DZVP-MOLOPT-SR-GTH-q22 (parsed cell basis) -------------------------
er_basis_gth = """
Er  DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q22
1
2 0 4 7 3 2 2 2 1
  6.882482040000 -0.000574800000 -0.010224030000 -0.005497100000  0.013538650000 -0.006244070000 -0.001910260000 -0.004983050000 -0.477609210000 -0.303449340000  0.000500880000
  4.326958650000 -0.027553420000  0.027571250000 -0.000826630000 -0.044244320000  0.026745230000  0.005117640000  0.013853330000 -0.516817160000 -0.245333810000 -0.002707510000
  2.222820950000  0.425605960000 -0.050849260000  0.416893650000 -0.213757410000 -0.113281080000 -0.052239440000 -0.027685660000 -0.413596980000 -0.239147390000 -0.010633820000
  1.027600450000 -0.631096360000  0.095005420000 -0.226240790000  0.656120530000  0.134326800000  0.366477120000  0.069056430000 -0.224718710000 -0.280322890000  0.077060330000
  0.428276640000 -0.462278630000 -0.433627130000 -0.122864000000  0.710122540000 -0.036607030000  0.845733320000  0.267956150000  0.223897560000 -0.539980340000  0.672495810000
  0.160552570000  0.327196660000  0.405567780000 -0.782199480000  0.004387480000  0.720379370000  0.379011530000 -0.765722080000  0.431766120000 -0.592083250000  0.592191430000
  0.044884870000 -0.314735200000  0.796872580000  0.384775650000 -0.131831310000  0.669389940000 -0.063412450000 -0.579756130000  0.216047230000 -0.264227340000  0.437035180000
"""

# ---- Er GTH-PADE-q22 pseudopotential ---------------------------------------
er_pseudo_gth = """
Er  GTH-PADE-q22 GTH-LDA-q22 GTH-PADE GTH-LDA
4 6 0 12
      0.50583333   2    17.10529281    -1.43095318
     4
      0.41994801   2     2.14450257     1.54307528
                                       -3.98420323
      0.41445530   2     0.05408736     0.96959342
                                       -2.29447680
      0.41838497   1    -0.99900632
      0.24912591   1   -26.69680936
"""

er_minao_szv = """
Er  SZV-MOLOPT-SR-GTH SZV-MOLOPT-SR-GTH-q22
1
2 0 3 7 1 1 1 1
  6.882482040000 -0.000574800000  0.013538650000 -0.001910260000 -0.477609210000
  4.326958650000 -0.027553420000 -0.044244320000  0.005117640000 -0.516817160000
  2.222820950000  0.425605960000 -0.213757410000 -0.052239440000 -0.413596980000
  1.027600450000 -0.631096360000  0.656120530000  0.366477120000 -0.224718710000
  0.428276640000 -0.462278630000  0.710122540000  0.845733320000  0.223897560000
  0.160552570000  0.327196660000  0.004387480000  0.379011530000  0.431766120000
  0.044884870000 -0.314735200000 -0.131831310000 -0.063412450000  0.216047230000
"""


def build_er_cell():
    """One Er in a 12 Å cubic box (gamma-only, vacuum reference)."""
    cell = gto.Cell()
    cell.atom = [["Er", (6.0, 6.0, 6.0)]]
    cell.a = np.eye(3) * 12.0
    cell.basis = {"Er": gto.basis.parse(er_basis_gth)}
    cell.pseudo = {"Er": gto.pseudo.parse(er_pseudo_gth)}
    cell.unit = "Angstrom"
    cell.spin = 4  # Er^3+ in atom: 4f^11; here we keep
    # neutral Er with q22 → adjust to taste
    cell.verbose = 4
    cell.ke_cutoff = 80
    cell.exp_to_discard = 0.1
    cell.build()
    return cell


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    cell = build_er_cell()

    kpts = cell.make_kpts([1, 1, 1])

    kmf = scf.KROHF(cell, kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 120
    kmf.run()
    print(f"\nKROHF energy = {kmf.e_tot:.8f}\n")

    tchkfile.save_kmf(kmf, "temp_chk_HF_er")
    kmf = tchkfile.load_kmf(cell, kmf, kpts, "temp_chk_HF_er")

    minao_dict = {
        "Er": gto.basis.parse(er_minao_szv),
        # "Mo": "gth-szv-molopt-sr",
        # "S":  "gth-szv-molopt-sr",
    }

    # Sanity check the IAO build standalone
    C_ao_lo, C_val, C_virt, lo_labels = make_iao_pao_kbasis(
        cell, kmf=kmf, minao=minao_dict
    )
    print(
        f"\n[standalone] nao={cell.nao}, "
        f"nval(IAO)={C_val.shape[-1]}, nvirt(PAO)={C_virt.shape[-1]}"
    )
    for i, lbl in enumerate(lo_labels):
        print(f"  {i:3d}  {lbl}")

    # Test pDMET
    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="HF")
    pdmet_obj.lobasis.minao = minao_dict
    pdmet_obj.emb.impCluster = [1]
    pdmet_obj.emb.imp_orbital_filter = {"Er": ["4f"]}

    pdmet_obj.initialize()

    # Plot the LOs.
    out_dir = os.path.join(here, "er_lo_xsf")
    os.makedirs(out_dir, exist_ok=True)
    pdmet_obj.plot(orb="lo", grid=[40, 40, 40], path=out_dir, fmt="xsf")
    print(f"\nLO XSF files in: {out_dir}")


if __name__ == "__main__":
    main()
