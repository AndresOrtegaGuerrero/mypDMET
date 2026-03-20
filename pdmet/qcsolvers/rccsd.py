from pdmet.qcsolvers.base import BaseSolver
from pyscf import cc
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm


class RCCSDSolver(BaseSolver):
    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.t1 = None
        self.t2 = None
        self.cc = None

    def _create_cc(self):
        """Initialize CC object"""
        self.cc = cc.CCSD(self.mf)

    def _update_cc(self):
        """Sync the CC object with the current mol/mf state."""
        self.cc.mol = self.mol
        self.cc._scf = self.mf
        self.cc.mo_coeff = self.mf.mo_coeff
        self.cc.mo_occ = self.mf.mo_occ
        self.cc._nocc = self.Nel // 2
        self.cc._nmo = self.Norb
        self.cc.chkfile = self.mf.chkfile

    def _t_guess(self):
        """Return saved amplitudes if shapes match, else None"""
        if self.t1 is not None and self.t1.shape[0] == self.cc._nocc:
            return self.t1, self.t2
        return None, None

    def _compute_T_correction(self):
        """Energy correction beyond CCSD"""
        return 0.0

    def _compute_rdms(self, t1, t2):
        """Default CCSD RDMs"""
        rdm1 = self.cc.make_rdm1(t1=t1, t2=t2)
        rdm2 = self.cc.make_rdm2(t1=t1, t2=t2)
        return rdm1, rdm2

    def kernel(self):
        """Run CCSD in the embedding basis"""
        self._setup_mf()
        self._create_cc()
        self._update_cc()
        t1_0, t2_0 = self._t_guess()
        Ecorr, t1, t2 = self.cc.kernel(t1=t1_0, t2=t2_0)

        # Update t1 and t2
        self.t1, self.t2 = t1, t2
        if not self.cc.converged:
            print("WARNING: CCSD not converged")

        ET = self._compute_T_correction()

        RDM1_mo, RDM2_mo = self._compute_rdms(t1, t2)
        RDM1, RDM2 = self._mo_to_local(RDM1_mo, RDM2_mo)

        E_imp = self._impurity_energy(RDM1, RDM2)

        e_cell = self.kmf_ecore + E_imp
        E_total = self.mf.e_tot + Ecorr + ET
        return e_cell, E_total, RDM1


class RCCSD_TSolver(RCCSDSolver):
    """
    CCSD(T): CSSD energy + perturbative triplets correections
    RDMs are from CSCS - The Triplet correction only affect the energy
    """

    def _compute_T_correction(self):
        """Energy correction beyond CCSD"""
        return self.cc.ccsd_t()


class RCCSD_TSlowSolver(RCCSDSolver):
    """
    Couple-cluster Single-Double (T) with full CCSD(T)
    """

    def _compute_T_correction(self):
        """Energy correction beyond CCSD"""
        return self.cc.ccsd_t()

    def _compute_rdms(self, t1, t2):
        eris = self.cc.ao2mo()  # Consume too much memory
        l1, l2 = ccsd_t_lambda.kernel(self.cc, eris, t1, t2, verbose=self.verbose)[1:]
        RDM1_mo = ccsd_t_rdm.make_rdm1(self.cc, t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)
        RDM2_mo = ccsd_t_rdm.make_rdm2(self.cc, t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)

        return RDM1_mo, RDM2_mo
