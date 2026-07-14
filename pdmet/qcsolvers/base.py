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
        # pyscf computes nalpha = (Nel + spin)//2: a parity mismatch would
        # silently change 2S (or lose an electron). Fail loudly instead.
        assert (self.Nel - self.mol.spin) % 2 == 0, (
            f"Nelec_in_emb={self.Nel} incompatible with twoS={self.mol.spin}; "
            "check bath truncation / num_bath."
        )
        base = (
            scf.ROHF(self.mol)
            if (self.mol.spin or self.is_KROHF)
            else scf.RHF(self.mol)
        )
        self.mf = base.density_fit()
        h_emb = self.FOCK - self.chempot
        s_emb = np.eye(self.Norb)
        self.mf.get_hcore = lambda *args: h_emb
        self.mf.get_ovlp = lambda *args: s_emb

        # Inject embedding DF tensor as the 3-center _cderi
        # Shape convention: (naux, nemb*(nemb+1)/2) — lower-triangular packed
        naux, nemb, _ = self.B.shape
        # pack_tril silently discards any antisymmetric part of B
        assert abs(self.B - self.B.transpose(0, 2, 1)).max() < 1e-10, (
            "embedding DF tensor B is not symmetric in (m, n)"
        )
        self.mf.with_df._cderi = lib.pack_tril(self.B)
        self.mf.with_df.auxcell = None
        self.mf.with_df.get_naoaux = lambda: naux

        if not self.settings.run_emb_scf:
            name = str(self.solver)
            if not any(t in name for t in ("CAS", "DMRG", "SHCI", "FCI")):
                raise RuntimeError(
                    f"run_emb_scf=False reached the {name} solver: HF/MP2/"
                    f"CCSD require a converged embedded mean field "
                    f"(Brillouin's theorem). Set run_emb_scf=True for this "
                    f"solver -- was it changed after initialize()?"
                )
            # CAS-DMET container mode: no SCF. The mf only holds the
            # embedding integrals + orbitals; the CAS/DMRG solver does all
            # the solving and NEVPT2 corrects the energy from the KROHF
            # low level. Converging an embedded HF here would only rotate
            # away the designed embedding orbitals.
            self._setup_mf_container()
            return

        self.mf.scf(self.DMguess)
        if not self.mf.converged:
            # newton() returns a *copy* (its kernel writes results onto that
            # copy, never back onto self.mf) -- capture and copy back by hand.
            mf2 = self.mf.newton()
            mf2.kernel(mo_coeff=self.mf.mo_coeff, mo_occ=self.mf.mo_occ)
            self.mf.mo_coeff, self.mf.mo_occ = mf2.mo_coeff, mf2.mo_occ
            self.mf.mo_energy, self.mf.e_tot = mf2.mo_energy, mf2.e_tot
            self.mf.converged = mf2.converged
        if not self.mf.converged:
            raise RuntimeError("Embedded HF/ROHF did not converge (SCF + Newton).")

    def _setup_mf_container(self):
        """Populate the mf as a pure container (run_emb_scf=False).

        Orbitals (settings.emb_orbitals):
          "embedding" -- identity: the impurity+bath orbitals AS CONSTRUCTED.
                         molist indices == embedding orbital indices.
          "natural"   -- natural orbitals of the embedding guess density:
                         the same space (a rotation-free re-sort would not
                         change any physics the CAS can express), ordered by
                         occupation so CASSCF's positional core window is
                         automatically sensible.

        mo_occ is the ideal ROHF pattern (nb doubles, 2S singles) on the
        chosen ordering; mo_energy = diag(C^T FOCK C) for diagnostics only.
        mf.e_tot is the energy OF THE GUESS OCCUPATION -- bookkeeping only;
        the physical energy comes from the CAS solver (+NEVPT2).
        """
        nb = (self.Nel - self.mol.spin) // 2
        na = self.Nel - nb
        mo_occ = np.zeros(self.Norb)
        mo_occ[:nb] = 2
        mo_occ[nb:na] = 1

        if self.settings.emb_orbitals == "natural":
            n_occ, C = np.linalg.eigh(self.DMguess)
            order = np.argsort(n_occ)[::-1]
            C = np.ascontiguousarray(C[:, order])
            n_diag = n_occ[order]
        else:  # "embedding"
            C = np.eye(self.Norb)
            n_diag = np.diag(self.DMguess).copy()
            # CASSCF's core window is positional: warn when the first nb raw
            # embedding orbitals do not actually carry the occupied density.
            core_charge = float(n_diag[:nb].sum())
            if abs(core_charge - 2.0 * nb) > 0.5:
                print(
                    f"  WARNING(container): first {nb} embedding orbitals "
                    f"hold {core_charge:.2f} e (ideal {2 * nb}). The "
                    f"positional core window is wrong -- select the active "
                    f"space with molist (CASSCF's core-virtual rotations "
                    f"relax the rest), or set emb_orbitals='natural'."
                )

        self.mf.mo_coeff = C
        self.mf.mo_occ = mo_occ
        self.mf.mo_energy = lib.einsum("pi,pq,qi->i", C, self.FOCK, C)
        self.mf.e_tot = self.mf.energy_tot(dm=self.mf.make_rdm1())
        self.mf.converged = True  # container: no SCF was run (by design)
        head = np.round(n_diag[: min(10, self.Norb)], 3)
        print(
            f"  [container] embedded SCF skipped -- mf holds integrals + "
            f"{self.settings.emb_orbitals} orbitals "
            f"(guess occupations head: {head.tolist()})"
        )

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
