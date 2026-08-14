import os
import pywannier90
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf, df

from pdmet import dmet
from pdmet.tools import tchkfile
from pdmet.settings import DMRGSettings

lib.logger.TIMER_LEVEL = lib.logger.INFO


cell = gto.Cell()
cell.atom = """
H 1.0 1.0 1.0
H 1.0 1.0 2.0
H 2.0 1.0 1.0
H 2.0 1.0 2.0
"""
cell.basis = "gth-dzv"
cell.spin = 0
cell.verbose = 2
cell.max_memory = 8000
cell.a = np.eye(3) * 10
cell.verbose = 5
cell.build()

"""================================"""
""" Build GDF """
"""================================"""
kmesh = [1, 1, 1]
kpts = cell.make_kpts(kmesh)
if not os.path.exists("gdf.h5"):
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = "gdf.h5"
    gdf.build()

"""================================"""
""" Read the HF wave function"""
"""================================"""
kmesh = [1, 1, 1]
kpts = cell.make_kpts(kmesh)
khf = scf.KRHF(cell, kpts).density_fit()
khf.with_df._cderi = "gdf.h5"
khf.exxdiv = None
khf.run()
print("khf mo coeff", khf.mo_coeff)
tchkfile.save_kmf(khf, "chk_HF")


"""================================"""
""" Contruct MLWFs """
"""================================"""
kmf = tchkfile.load_kmf(khf, "chk_HF")
num_wann = cell.nao
keywords = """
num_iter = 5000
begin projections
random
H: s
end projections
guiding_centres = .true.
"""
w90 = pywannier90.W90(kmf, cell, kmesh, num_wann, other_keywords=keywords)
w90.kernel()
tchkfile.save_w90(w90, "chk_w90")
kmf = tchkfile.load_kmf(khf, "chk_HF")
"""================================"""
""" Run DMET -- DMRG-CI (fused NEVPT2) vs CASCI (exact FCI reference) """
"""================================"""
# Single-state (ground) + NEVPT2. DMRG-CI takes the fused path; CASCI is the
# exact FCI at the SAME fixed orbitals, so the two NEVPT2 totals should agree
# for cas(4,4). ("CHEMPS2-CI" / "SS-DMRG-SCF" are also valid drop-ins.)
SOLVERS = ["DMRG-CI", "CASCI"]


def run(solver_name):
    print("\n" + "#" * 60 + f"\n# SOLVER = {solver_name}\n" + "#" * 60)
    scratch = os.path.abspath(os.path.join("./tmp", solver_name))
    os.makedirs(scratch, exist_ok=True)

    pdmet = dmet.pDMET(cell, kmf, w90, solver=solver_name)
    pdmet.kmf_chkfile = "chk_HF"
    pdmet.lobasis.w90_chkfile = "chk_w90"
    pdmet.emb.impCluster = [1]
    pdmet.emb.impOrbs_threshold = 1.5
    pdmet.solver.twoS = 0
    pdmet.solver.cas = (4, 4)
    pdmet.solver.dmrg = DMRGSettings()
    pdmet.solver.dmrg.scratch_dir = scratch
    pdmet.solver.dmrg.runtime_dir = scratch

    pdmet.solver.nevpt2_roots = [0]
    pdmet.solver.nevpt2_nroots = 1
    pdmet.solver.nroots = 1

    pdmet.initialize()
    pdmet.one_shot()
    return pdmet


for _solver in SOLVERS:
    run(_solver)
