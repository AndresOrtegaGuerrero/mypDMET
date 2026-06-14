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
        """Ensure a mean-field object has embedding ERIs in ``mf._eri``.

        Some PySCF paths (e.g. FCI, NEVPT2) access ``mf._eri`` directly and, if it is
        missing, incorrectly evaluate ``int2e`` on the dummy embedding molecule. This
        helper lazily builds the embedding ERI

            (ij|kl) = Σ_L B_Lij B_Lkl

        from the DF factors and stores the 8-fold packed result in ``mf._eri``.

        The ERI is assembled directly in packed form (no dense ``nemb^4`` tensor),
        reducing peak memory to ~``nemb^4/4``. Construction remains O(``nemb^4``) in
        the embedding size. For large embeddings, prefer DMRG with compressed NEVPT2
        (``use_compress_nevpt2=True``), which avoids ``_eri`` entirely.

        The operation is idempotent: if ``mf._eri`` already contains a valid
        ``float64`` array, no work is done.

        Parameters
        ----------
        mf : pyscf mean-field, optional
            Mean-field object to receive ``_eri``. Defaults to ``self.mf``. Useful
            when attaching ERIs to temporary CASCI/NEVPT2 mean-field objects.
        """
        from pyscf import ao2mo

        if mf is None:
            mf = self.mf
        if (
            getattr(mf, "_eri", None) is not None
            and getattr(mf._eri, "dtype", None) == np.float64
        ):
            return  # already populated, nothing to do
        # Direct packed build: pack the symmetric (i,j) pair index of B once,
        # then (ij|kl) = B_sym^T @ B_sym is already the 4-fold-packed ERI; no
        # dense Norb^4 intermediate. restore(8) gives the 8-fold _eri pyscf
        # consumes. Identical numbers to the dense einsum, lower peak memory.
        B_sym = lib.pack_tril(np.asarray(self.B))  # (naux, npair=Norb(Norb+1)/2)
        eri_s4 = lib.dot(B_sym.T, B_sym)  # (npair, npair) 4-fold-packed ERI
        mf._eri = ao2mo.restore(8, eri_s4, self.Norb)

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
