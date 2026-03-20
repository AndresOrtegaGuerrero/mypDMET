"""
Semistochastic heat bath configuration interaction (SHCI)
"""

import numpy as np
from pyscf import lib
from pdmet.qcsolvers.base import BaseSolver


class SHCISolver(BaseSolver):
    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.ci = None
        self.mo_coeff = None

    def kernel(self):
        """
        Run SHCI in the embedding basis
        """
        self._self_mf()
        self._create_shci()
        if self.nroots == 1:
            self._single_root()
        else:
            self._multi_root()

    def _create_shci(self):
        """Create SHCI object"""
        try:
            from pyscf.shciscf import shci
        except ImportError:
            raise ImportError(
                "pyscf.shciscf not found. Install PySCF with SHCI support."
                "This typically means settings.py is missing in pyscf.shciscf."
            )
        # Note: In the future Settings should have a dedicated dictionary for its own setttings
        self.mch = shci.SHCISCF(self.mf, self.Norb, self.mol.nelectron)

        self.mch.fcisolver.mpiprefix = ""
        self.mch.fcisolver.nPTiter = 0
        self.mch.fcisolver.sweep_iter = [0, 3]
        self.mch.fcisolver.sweep_epsilon = [1e-3, 0.5e-3]

        # Warm start
        ci0 = getattr(self, "ci", None)
        mo0 = getattr(self, "mo_coeff", None)

        e_noPT, self.e_cas, fcivec, mo_coeff = self.mch.mc1step(mo_coeff=mo0, ci0=ci0)[
            :4
        ]
        self.ci = fcivec
        self.mo_coeff = mo_coeff
        self.ESHCI = self.e_cas  # Check if this is correct

    def _single_root(self):
        if not self.mch.converged:
            print("WARNING: The solver is not converged")
        fcivec = self.ci

        self.SS = self.mch.fcisolver.spin_square(fcivec, self.Norb, self.mol.nelec)[0]
        RDM1_mo, RDM2_mo = self.mch.fcisolver.make_rdm12(
            fcivec, self.Norb, self.mol.nelec
        )
        RDM1, RDM2 = self._mo_to_local(RDM1_mo, RDM2_mo)
        e_cell = self.kmf_ecore + self._impurity_energy(RDM1, RDM2)
        return e_cell, self.ESHCI, RDM1

    def _multi_root(self):
        if not self.mch.converged.any():
            print("WARNING: The solver is not converged")

        RDM1s, e_cells, ss_list = [], [], []
        fcivec = self.ci
        for i, vec in enumerate(fcivec):
            ss = self.mch.fcisolver.spin_square(fcivec, self.Norb, self.mol.nelec)[0]
            rdm1_mo, rdm2_mo = self.mch.fcisolver.make_rdm12(
                fcivec, self.Norb, self.mol.nelec
            )
            rdm1, rdm2 = self._mo_to_local(rdm1_mo, rdm2_mo)
            e_imp = self.kmf_ecore + self._impurity_energy(rdm1, rdm2)
            print(f"Root {i}: ESHCI = {self.ESHCI[i]:.6f}, S^2 = {ss:.4f}")
            RDM1s.append(rdm1)
            e_cells.append(e_imp)
            ss_list.append(ss)

        w = np.asarray(self.settings.state_percent)  # weights
        RDM1 = lib.einsum("i,ijk->jk", w, RDM1s)
        e_cell = lib.einsum("i,i->", w, e_cells)
        self.SS = np.average(ss_list)
        return e_cell, self.ESHCI, RDM1
