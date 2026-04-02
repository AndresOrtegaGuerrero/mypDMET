from pdmet.qcsolvers.base import BaseSolver
from pyscf import mp


class MP2Solver(BaseSolver):
    """Restricted open/close-shell Møller–Plesset perturbation theory (RMP2/ROMP2)"""

    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.mp2 = None

    def kernel(self):
        """Run MP2 in the embedding basis"""
        self._setup_mf()
        self.mp2 = mp.MP2(self.mf)
        ecorr, t2 = self.mp2.kernel()
        EMP2 = self.mf.e_tot + ecorr
        RDM1_mo = self.mp2.make_rdm1(t2=t2)[0]
        RDM2_mo = self.mp2.make_rdm2(t2=t2)[0]
        RDM1, RDM2 = self._mo_to_local(RDM1_mo, RDM2_mo)

        # Energy impurity
        E_imp = self._impurity_energy(RDM1, RDM2)
        e_cell = self.kmf_ecore + E_imp

        return e_cell, EMP2, RDM1
