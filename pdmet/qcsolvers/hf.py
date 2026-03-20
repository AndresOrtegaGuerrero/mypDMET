import numpy as np
from pdmet.qcsolvers.base import BaseSolver


class HFSolver(BaseSolver):
    """Restricted open/close-shell Hartree-Fock (RHF/ROHF)"""

    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)

    def kernel(self):
        self._setup_mf()
        RDM1 = self.mf.make_rdm1()  # Density matrix
        JK = self.mf.get_veff(None, dm=RDM1)  # Coulomb-exchange matrix

        # Multiply by 0.5 to avoid double counting only on impurity
        def impurity_energy(dm, jk):
            dm_imp = dm[: self.Nimp, : self.Nimp]
            fock_imp = self.FOCK[: self.Nimp, : self.Nimp]
            oei_imp = self.OEI[: self.Nimp, : self.Nimp]
            jk_imp = jk[: self.Nimp, : self.Nimp]

            return 0.5 * np.sum(dm_imp * (fock_imp + oei_imp)) + 0.5 * np.sum(
                dm_imp * jk_imp
            )

        # Close-shell
        if self.mol.spin == 0 and not self.is_KROHF:
            E_imp = impurity_energy(RDM1, JK)
        else:
            E_imp = sum(impurity_energy(RDM1[spin], JK[spin]) for spin in range(2))
            RDM1 = RDM1.sum(axis=0)
        # Total energy is kmf_core + E_imp
        return self.kmf_ecore + E_imp, self.mf.e_tot, RDM1
