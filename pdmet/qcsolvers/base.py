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
        self,
        kmf_ecore,
        OEI,
        B,
        JK,
        DMguess,
        Norb,
        Nel,
        Nimp,
        chempot=0.0,
    ):
        """Load embedding integrals.

        Parameters
        ----------
        B : (naux, nemb, nemb) ndarray
            Embedding density-fitting 3-center tensor. The full 4-index
            ERI is V_ijkl = sum_L B_Lij B_Lkl, but we never materialise it.
        """
        self.kmf_ecore = kmf_ecore
        self.OEI = OEI
        self.B = B
        self.FOCK = OEI + JK
        self.DMguess = DMguess
        self.Norb = Norb
        self.Nel = Nel
        self.Nimp = Nimp
        chempot_diag = np.zeros(Norb)
        chempot_diag[:Nimp] = chempot
        self.chempot = np.diag(chempot_diag)

    def build_full_tei(self):
        """Reassemble the 4-index ERI from B. O(nemb^4) memory — debug only."""
        return lib.einsum("Lij,Lkl->ijkl", self.B, self.B, optimize=True)

    def _ensure_eri(self, mf=None):
        """Lazily assemble the embedding 4-index ERI on a mean-field object.

        Some pyscf paths (plain FCI, mrpt.NEVPT) have no DF integral
        formulation: they read `mf._eri` directly. When the mean-field is our
        DF-wrapped object, `_eri` is None and they fall back to
        `mf.mol.intor('int2e')` on the dummy molecule, which is wrong.

        This helper rebuilds V = sum_L B_Lij B_Lkl in the embedding space
        (small: nemb^4, not nao^4), packs it 8-fold, and assigns it to
        `mf._eri`. Idempotent — if a valid float64 _eri is already there,
        nothing happens.

        Parameters
        ----------
        mf : pyscf mean-field, optional
            Target mean-field to attach _eri to. Defaults to `self.mf`.
            Pass a different one for NEVPT2 helpers that create a fresh
            CASCI(self.mf, ...) and need _eri on that mc's _scf.
        """
        from pyscf import ao2mo

        if mf is None:
            mf = self.mf
        if (
            getattr(mf, "_eri", None) is not None
            and getattr(mf._eri, "dtype", None) == np.float64
        ):
            return  # already populated, nothing to do
        V = lib.einsum("Lij,Lkl->ijkl", self.B, self.B, optimize=True)
        mf._eri = ao2mo.restore(8, V, self.Norb)

    def _setup_mf(self):
        """Inject the embedding Hamiltonian and run the SCF using density fitting.

        Instead of rebuilding the full 4-index ERIs and assigning them to ``mf._eri``,
        we create a DF mean-field object and pass the precomputed embedding 3-center
        tensor ``B`` through ``mf.with_df._cderi`` (stored in lower-triangular packed
        form).
        """
        from pyscf import scf

        self.mol.nelectron = self.Nel
        base = (
            scf.ROHF(self.mol)
            if (self.mol.spin or self.is_KROHF)
            else scf.RHF(self.mol)
        )
        self.mf = base.density_fit()
        self.mf.get_hcore = lambda *args: self.FOCK - self.chempot
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)

        # Inject embedding DF tensor as the 3-center _cderi
        # Shape convention: (naux, nemb*(nemb+1)/2) — lower-triangular packed
        naux, nemb, _ = self.B.shape
        self.mf.with_df._cderi = lib.pack_tril(self.B)
        self.mf.with_df.auxcell = None
        self.mf.with_df.get_naoaux = lambda: naux

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
        """Partition impurity energy from RDMs and embedding integrals (DF form).

        The four impurity-permutation 2-body contractions are evaluated through
        the 3-center tensor B, mirroring `_impurity_energy_from_cas_df`:

            sum_{ijkl in imp-slice} RDM2_ijkl V_ijkl
              = sum_L (B_imp_ij RDM2_ijkl) (B_kl)
                          ^ contract through L, never form V
        """
        N = self.Nimp
        one_body = 0.5 * lib.einsum(
            "ij,ij->", RDM1[:N, :], self.FOCK[:N, :] + self.OEI[:N, :]
        )

        B = self.B  # (naux, nemb, nemb)
        B_imp = B[:, :N, :]  # (naux, Nimp, nemb)

        def perm(dm2, imp_left):
            # dm2 has the impurity index on either side; contract through L
            if imp_left:
                D = lib.einsum("pqrs,Lpq->Lrs", dm2, B_imp, optimize=True)
                return lib.einsum("Lrs,Lrs->", D, B, optimize=True)
            D = lib.einsum("pqrs,Lrs->Lpq", dm2, B_imp, optimize=True)
            return lib.einsum("Lpq,Lpq->", D, B, optimize=True)

        t2 = perm(RDM2, True)
        t2 += perm(RDM2.transpose(1, 0, 3, 2), True)
        t2 += perm(RDM2.transpose(2, 3, 0, 1), False)
        t2 += perm(RDM2.transpose(3, 2, 1, 0), False)

        return one_body + 0.125 * t2
