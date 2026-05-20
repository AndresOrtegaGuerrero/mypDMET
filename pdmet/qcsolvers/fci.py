from pdmet.qcsolvers.base import BaseSolver
from pyscf import fci, lib
import numpy as np


class FCISolver(BaseSolver):
    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.ci = None  # In case CI vectors are passed

    def _build_fs(self):
        """Build the FCI object"""
        fs = fci.FCI(self.mf, self.mf.mo_coeff)
        if self.settings.e_shift is not None:
            fs = fci.addons.fix_spin_(fs, self.settings.e_shift)  # Spin penalty
        fs.verbose = self.settings.verbose
        fs.conv_tol = 1e-10
        fs.conv_tol_residual = None
        fs.nroots = self.settings.nroots
        return fs

    def _single_root(self, EFCI, fcivec):
        if not self.fs.converged:
            print("WARNING: FCI not converged")
        self.SS = self.fs.spin_square(fcivec, self.Norb, self.mol.nelec)[0]
        RDM1_mo, RDM2_mo = self.fs.make_rdm12(fcivec, self.Norb, self.mol.nelec)
        RDM1, RDM2 = self._mo_to_local(RDM1_mo, RDM2_mo)
        e_cell = self.kmf_ecore + self._impurity_energy(RDM1, RDM2)
        return e_cell, EFCI, RDM1

    def _multi_root(self, EFCI, fcivec):
        if not self.fs.converged.any():
            print("WARNING: The solver is not converged")
        RDM1s, e_cells, ss_list = [], [], []
        for i, vec in enumerate(fcivec):
            ss = self.fs.spin_square(vec, self.Norb, self.mol.nelec)[0]
            fs_r1, fs_r2 = self.fs.make_rdm12(vec, self.Norb, self.mol.nelec)
            rdm1, rdm2 = self._mo_to_local(fs_r1, fs_r2)
            e_imp = self.kmf_ecore + self._impurity_energy(rdm1, rdm2)
            print(
                f"Root {i}: EFCI = {EFCI[i]:.6f}, E(Imp)= {e_imp:.6f}, S^2 = {ss:.4f}"
            )
            RDM1s.append(rdm1)
            e_cells.append(e_imp)
            ss_list.append(ss)

        w = np.asarray(self.settings.state_percent)  # weights
        RDM1 = lib.einsum("i,ijk->jk", w, RDM1s)
        e_cell = lib.einsum("i,i->", w, e_cells)
        self.SS = np.average(ss_list)

        return e_cell, EFCI, RDM1

    def kernel(self):
        self._setup_mf()
        # FCI has no DF formulation
        self._ensure_eri()
        self.fs = self._build_fs()
        EFCI, fcivec = self.fs.kernel(ci0=self.ci)

        self.ci = fcivec
        if self.settings.nroots == 1:
            return self._single_root(EFCI, fcivec)
        else:
            return self._multi_root(EFCI, fcivec)
