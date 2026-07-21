"""DMRG-CASSCF (block2): orbital optimization + optional NEVPT2.

All logic lives in BaseDMRGBlock2Solver; this driver only picks the CASSCF mc.
SS / SA / SA-mix are runtime kwargs to the inherited kernel().
"""

from pyscf import mcscf
from pdmet.qcsolvers.basedmrg import BaseDMRGBlock2Solver


class DMRGBlock2Solver(BaseDMRGBlock2Solver):
    def _build_mc(self, cas_norb, cas_nelec):
        return mcscf.CASSCF(self.mf, cas_norb, cas_nelec)
