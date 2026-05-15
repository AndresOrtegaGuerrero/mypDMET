import os
import pywannier90
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf, df

from pdmet import dmet
from pdmet.tools import tchkfile


lib.logger.TIMER_LEVEL = lib.logger.INFO

cell = gto.Cell()
cell.atom = """H 5 5 4; H 5 5 5"""
cell.basis = "gth-dzv"
cell.spin = 0
cell.verbose = 2
cell.max_memory = 10000
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

kmf = tchkfile.load_kmf(khf, "chk_HF")
"""================================"""
""" Run DMET """
"""================================"""
pdmet = dmet.pDMET(
    cell,
    kmf,
    w90,
    lo_method="wannier",
    solver="CASCI",
)  # pass an hf object (scf.ROHF(cell).density_fit()), not a khf object i.e. scf.KROHF(cell, kpts).density_fit(). scf.KROHF(cell, kpts).density_fit() prints an output type not compatible with slicing.
pdmet.lobasis.minao = "gth-dzv"
pdmet.emb.impCluster = [1]
pdmet.emb.imp_orbital_filter = {"H": ["1s", "2s", "2px", "2py", "2pz"]}
pdmet.emb.impOrbs_threshold = 1.5
pdmet.solver.twoS = 0
pdmet.solver.cas = (2, 2)
pdmet.solver.e_shift = 0.5

# #Excitations SA-CASSCF
weight = 1.0 / 3
pdmet.solver.nroots = 3
pdmet.solver.state_average_ = [weight, weight, weight]

# pdmet.solver.nevpt2_roots = [0]
# pdmet.solver.nevpt2_nroots = 1

# #Excitation NEVPT2
pdmet.solver.nevpt2_roots = list(range(0, 3))
pdmet.solver.state_average_ = [weight, weight, weight]
pdmet.solver.nevpt2_nroots = 3

pdmet.initialize()
pdmet.one_shot()
pdmet.plot(orb="wfs", grid=[50, 50, 50], path="./", fmt="xsf")
