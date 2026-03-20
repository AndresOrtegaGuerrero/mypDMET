import numpy as np
from pyscf import gto, scf, lib
from pdmet.settings import SolverSettings


class BaseSolver:
    """
    BaseClass for all impurity solvers
    """

    def __init__(self, settings: SolverSettings, is_KROHF=False):
        self.settings = settings
        self.is_KROHF = is_KROHF
        self.solver = self.settings.name
        self.SS = 0.5 * self.settings.twoS * (0.5 * self.settings.twoS + 1)
        self._build_dummy_molecule()

    def _build_dummy_molecule(self):
        """
        Build a dummy PySCF Mole object for the impurity solver.
        A single atom is used as placeholder
        Build th mean-field based on spin
        """
        self.mol = gto.Mole()
        self.mol.build(verbose=0)
        self.mol.atom.append(("S", (0, 0, 0)))
        self.mol.nelectron = 2 + self.settings.twoS
        self.mol.spin = self.settings.twoS
        self.mol.incore_anyway = True
        self.mol.max_memory = self.settings.max_memory
        if self.mol.spin == 0 and not self.is_KROHF:
            self.mf = scf.RHF(self.mol)
        else:
            self.mf = scf.ROHF(self.mol)

    def initialize(
        self, kmf_ecore, OEI, TEI, JK, DMguess, Norb, Nel, Nimp, chempot=0.0
    ):
        """Load embedding integrals."""
        self.kmf_ecore = kmf_ecore
        self.OEI = OEI
        self.TEI = TEI
        self.FOCK = OEI + JK
        self.DMguess = DMguess
        self.Norb = Norb
        self.Nel = Nel
        self.Nimp = Nimp
        chempot_diag = np.zeros(Norb)
        chempot_diag[:Nimp] = chempot
        self.chempot = np.diag(chempot_diag)

    def _setup_mf(self):
        """Inject embedding Hamiltonian and run SCF. Called at start of every kernel()."""
        from pyscf import ao2mo

        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: self.FOCK - self.chempot
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)
        if not self.mf.converged:
            dm = self.mf.mo_coeff @ np.diag(self.mf.mo_occ) @ self.mf.mo_coeff.T
            self.mf.newton().kernel(dm0=dm)

    def _mo_to_local(self, RDM1_mo, RDM2_mo=None):
        """Transform RDM1 (and optionally RDM2) from MO to local basis."""
        C = self.mf.mo_coeff
        RDM1 = lib.einsum("ap,pq,bq->ab", C, RDM1_mo, C)
        if RDM2_mo is None:
            return RDM1
        RDM2 = lib.einsum("ap,bq,cr,ds,pqrs->abcd", C, C, C, C, RDM2_mo)
        return RDM1, RDM2

    def _impurity_energy(self, RDM1, RDM2):
        """Partition impurity energy from RDMs and embedding integrals."""
        N = self.Nimp
        one_body = 0.5 * lib.einsum(
            "ij,ij->", RDM1[:N, :], self.FOCK[:N, :] + self.OEI[:N, :]
        )
        two_body = 0.125 * (
            lib.einsum("ijkl,ijkl->", RDM2[:N, :, :, :], self.TEI[:N, :, :, :])
            + lib.einsum("ijkl,ijkl->", RDM2[:, :N, :, :], self.TEI[:, :N, :, :])
            + lib.einsum("ijkl,ijkl->", RDM2[:, :, :N, :], self.TEI[:, :, :N, :])
            + lib.einsum("ijkl,ijkl->", RDM2[:, :, :, :N], self.TEI[:, :, :, :N])
        )
        return one_body + two_body
