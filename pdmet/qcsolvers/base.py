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
        self.lo_view = (
            None  # (emb_orbs, lo_labels, impOrbs); set by dmet.py for iao+pao
        )
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
        """Ensure ``mf._eri`` contains the embedding ERIs.

        Some PySCF routines (e.g. FCI and NEVPT2) access ``mf._eri``
        directly. If missing, they may evaluate ``int2e`` on the dummy
        embedding molecule instead. This helper reconstructs the embedding
        ERIs from the DF 3-center tensor ``B`` and stores them in PySCF's
        packed format.

        The ERIs are built directly in packed form, avoiding a dense
        ``nemb^4`` intermediate. If ``mf._eri`` is already populated,
        no work is performed.

        Parameters
        ----------
        mf : pyscf mean-field, optional
            Mean-field object receiving ``_eri``. Defaults to ``self.mf``.
        """
        from pyscf import ao2mo

        if mf is None:
            mf = self.mf

        if (
            getattr(mf, "_eri", None) is not None
            and getattr(mf._eri, "dtype", None) == np.float64
        ):
            return  # already populated

        # Pack AO pairs once, then build:
        # (ij|kl) = B_sym.T @ B_sym in 4-fold packed form.
        # restore(8) converts it to PySCF's packed _eri format.
        B_sym = lib.pack_tril(np.asarray(self.B))  # (naux, npair)
        eri_s4 = lib.dot(B_sym.T, B_sym)  # (npair, npair)
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
            # CAS-DMET container mode: no SCF. The mean-field object only stores
            # embedding integrals and orbitals; CAS/DMRG and NEVPT2 handle the
            # electronic structure. SCF is skipped to preserve the embedding orbitals."""
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

        # DMguess is spin-resolved (2, Norb, Norb) for the ROHF embedding;
        # occupation analysis wants the spin-SUMMED density (0..2 scale).
        dm_tot = np.asarray(self.DMguess)
        if dm_tot.ndim == 3:
            dm_tot = dm_tot[0] + dm_tot[1]

        if self.settings.emb_orbitals == "natural":
            n_occ, C = np.linalg.eigh(dm_tot)
            order = np.argsort(n_occ)[::-1]
            C = np.ascontiguousarray(C[:, order])
            n_diag = n_occ[order]
        else:  # "embedding"
            C = np.eye(self.Norb)
            n_diag = np.diag(dm_tot).copy()
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
        # bookkeeping energy of the TRUE embedding-projected low-level density
        self.mf.e_tot = self.mf.energy_tot(dm=self.DMguess)
        self.mf.converged = True  # container: no SCF by design

        print(
            f"  [container] embedded SCF skipped -- mf holds integrals + "
            f"{self.settings.emb_orbitals} orbitals; "
            f"E(low-level, embedding) = {self.mf.e_tot:.10f}"
        )
        self._print_container_occupations(n_diag, nb, na)
        if self.settings.emb_orbitals == "natural":
            self._print_container_composition(C, n_diag)

    def _print_container_occupations(self, n, nb, na):
        """Guess occupations at the closed|open and open|virtual boundaries."""
        lo = max(0, nb - 2)
        hi = min(self.Norb, na + 2)
        row = "  ".join(f"n[{i}]={n[i]:.3f}" for i in range(lo, hi))
        print(
            f"  [container] occupations near the frontier "
            f"({self.Nel} e in {self.Norb} orbitals): {row}"
        )
        if 0 < nb < na and (n[nb - 1] - n[nb]) < 0.5:
            print("  [container] WARNING: closed|open gap < 0.5 -- ambiguous")
        if 0 < na < self.Norb and (n[na - 1] - n[na]) < 0.5:
            print("  [container] WARNING: open|virtual gap < 0.5 -- ambiguous")
        nfrac = int(((n > 0.05) & (n < 1.95)).sum())
        print(f"  [container] fractional occupations (0.05 < n < 1.95): {nfrac}")

    def _lo_view(self, C):
        """(C_lo, labels, impurity LO rows) from lo_view, or None."""
        if getattr(self, "lo_view", None) is None:
            return None
        try:
            emb, labels, imp_mask = self.lo_view
            emb = np.asarray(emb)
            if emb.ndim == 3:  # (nkpt, nlo, Nemb) -> Gamma
                emb = emb[0]
            labels = [" ".join(str(label).split()) for label in np.asarray(labels)]
            imp_lo = np.where(np.asarray(imp_mask).astype(bool))[0]
            return emb @ np.asarray(C), labels, imp_lo
        except Exception as e:
            print(f"  [container] lo_view failed ({e}); using embedding rows")
            return None

    def _print_container_composition(self, C, n, max_rows=200):
        """NO composition vs the impurity block -- for picking molist indices."""
        w = np.abs(C) ** 2
        w_imp = w[: self.Nimp, :].sum(axis=0)
        sel = np.where(((n > 0.02) & (n < 1.98)) | (w_imp > 0.5))[0]
        if sel.size == 0:
            return

        lo = self._lo_view(C)
        if lo is not None:
            import re

            C_lo, labels, imp_lo = lo
            W = np.abs(C_lo) ** 2
            # shell columns from the impurity selection itself, in its order
            shell_rows = {}
            for r in imp_lo:
                p = labels[r].split()
                sh = re.match(r"\d+[a-z]", p[2]).group(0)  # "3dxy" -> "3d"
                shell_rows.setdefault(f"{p[1]}{sh}", []).append(int(r))
            print(
                "  [container] NO composition in the IAO+PAO basis "
                "(fractional or imp-weight > 0.5) -- molist uses these NO "
                "indices (0-based):"
            )
            print(
                "      NO       n   w_imp "
                + "".join(f"{nm:>7}" for nm in shell_rows)
                + "   top LO components"
            )
            for row_i, i in enumerate(sel):
                if row_i >= max_rows:
                    print(f"      ... {sel.size - max_rows} more suppressed")
                    break
                cols = "".join(
                    f"{W[rows, i].sum():7.2f}" for rows in shell_rows.values()
                )
                top = np.argsort(W[:, i])[::-1][:3]
                comp = "  ".join(
                    f"{' '.join(labels[k].split()[1:])}({W[k, i]:.2f})"
                    for k in top
                    if W[k, i] > 1e-2
                )
                print(f"   {i:6d}  {n[i]:6.3f}  {w_imp[i]:5.2f} {cols}   {comp}")
            return

        print(
            "  [container] NO composition (fractional or imp-weight > 0.5) "
            "-- molist uses these NO indices (0-based):"
        )
        print("      NO       n   w_imp   top embedding components (* = impurity)")
        for row_i, i in enumerate(sel):
            if row_i >= max_rows:
                print(f"      ... {sel.size - max_rows} more suppressed")
                break
            top = np.argsort(w[:, i])[::-1][:3]
            comp = "  ".join(
                f"emb#{k}({w[k, i]:.2f})" + ("*" if k < self.Nimp else "") for k in top
            )
            print(f"   {i:6d}  {n[i]:6.3f}  {w_imp[i]:5.2f}   {comp}")

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
