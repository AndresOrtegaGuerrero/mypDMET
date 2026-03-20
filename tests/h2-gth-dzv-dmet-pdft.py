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
kmf = tchkfile.load_kmf(cell, khf, kmesh, "chk_HF")
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

kmf = tchkfile.load_kmf(cell, khf, kmesh, "chk_HF")
"""================================"""
""" Run DMET """
"""================================"""
pdmet = dmet.pDMET(
    cell,
    kmf,
    w90,
    solver="CASCI",
)  # pass an hf object (scf.ROHF(cell).density_fit()), not a khf object i.e. scf.KROHF(cell, kpts).density_fit(). scf.KROHF(cell, kpts).density_fit() prints an output type not compatible with slicing.
pdmet.emb.impCluster = [1]
pdmet.emb.impOrbs_threshold = 1.5
pdmet.solver.twoS = 0
pdmet.solver.cas = (2, 2)
pdmet.solver.e_shift = 0.5
pdmet.initialize()
pdmet.one_shot()

"""

'''================================'''
''' Molecular-point MC-PDFT  '''
'''================================'''


mol2 = cell.to_mol()
hf = scf.ROHF(mol2).density_fit()
# hf.with_df._cderi = 'gdf.h5'
hf.verbose=5
hf_2=hf
hf.run()
print("hf mo coeff",hf.mo_coeff)
######################## print("mo coefficient of hfyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",hf.mo_coeff)
######################## randommc= mcscf.CASSCF(hf, 4, 2)

######################## print("randommc.mo_coeff",randommc.mo_coeff)
mc = mcpdft.CASSCF (hf, 'tPBE', 4, 2, grids_level=6)
mc = mc.fix_spin_(shift=0.5, ss=2)
print("mcpdft mo coeff is --------------------------------------",mc.mo_coeff)
############################## mc.fcisolver = csf_solver (cell, smult = 1)
mc.verbose = 3
Vnn = mc._scf.energy_nuc()
print("Vnn ----------------------------------------------------------------------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",Vnn)
mc.kernel ()
print("mc.mo_occ",mc.mo_occ)

"""
