from pdmet.qcsolvers.base import BaseSolver
from pyscf import lib
import numpy as np


class BaseCASSolver(BaseSolver):
    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.mc = None
        self.ci = None
        # NTO outputs (populated by _compute_ntos on any multi-root path).
        self.t_dm1s = None
        self.ntos_per_root = None
        # NDOS outputs
        self.d_dm1s = None
        self.ndos_per_root = None

    def _analyze_states(self, mc, fcivec, roots, allow_nto=True):
        """Entry point for excited-states analysis (NTOs/NDOs)"""

        if self.settings.nto and allow_nto:
            self._compute_ntos(mc, fcivec, roots)
        if self.settings.ndo:
            self._compute_ndos(mc, fcivec, roots)

    def _reset_analysis(self):
        """Clear NTO buffers at kernel start (avoid accumulation across DMET iters)."""
        self.t_dm1s = None
        self.ntos_per_root = None
        self.d_dm1s = None
        self.ndos_per_root = None

    def _compute_root_quantities(self, mc, fcivec, roots, compute, dm1_attr, info_attr):
        """Compute and accumulate root-dependent density quantities."""
        if not isinstance(fcivec, (list, tuple)):
            fcivec = [fcivec]

        dm1s = getattr(self, dm1_attr)
        infos = getattr(self, info_attr)

        if dm1s is None:
            dm1s, infos = [], []
            setattr(self, dm1_attr, dm1s)
            setattr(self, info_attr, infos)

        for root in roots:
            dm1, info = compute(mc, fcivec, root)
            dm1s.append(dm1)
            infos.append(info)

        return dm1s, infos

    def _compute_ntos(self, mc, fcivec, roots):
        """Compute transition densities and NTOs."""
        return self._compute_root_quantities(
            mc,
            fcivec,
            roots,
            self._transition_dm1,
            "t_dm1s",
            "ntos_per_root",
        )

    def _compute_ndos(self, mc, fcivec, roots):
        """Compute difference densities and NDOs."""
        return self._compute_root_quantities(
            mc,
            fcivec,
            roots,
            self._difference_dm1,
            "d_dm1s",
            "ndos_per_root",
        )

    def _mo_guess(self, mc):
        """Initial-guess MOs for CASCI/CASSCF, in priority order:
        1. settings.mo_restart -- MOs from a checkpoint. They already define the
            converged active space, so hand them back as-is (continue, not restart).
        2. settings.molist     -- user sorts chosen orbitals into the active space.
        3. self.mf.mo_coeff    -- plain mean-field guess (the default path).
        """
        if self.settings.mo_restart is not None:
            return self.settings.mo_restart

        if self.settings.molist is not None:
            from pyscf import mcscf

            return mcscf.sort_mo(mc, mc.mo_coeff, self.settings.molist, 0)

        return self.mf.mo_coeff

    def _apply_spin_constraint(self):
        """Apply spin constraint if requested."""
        if self.settings.e_shift is None:
            return

        target_SS = 0.5 * self.settings.twoS * (0.5 * self.settings.twoS + 1)

        # Only safe for standard FCI-like solvers
        if hasattr(self.mc, "fix_spin_"):
            self.mc.fix_spin_(shift=self.settings.e_shift, ss=target_SS)

    def _set_fci_solver(self, solver_name):
        """Configure the FCI solver (standard FCI or CheMPS2 for DMRG)."""
        if solver_name == "CheMPS2":
            try:
                from pyscf import dmrgscf

                self.mc.fcisolver = dmrgscf.CheMPS2(self.mol)
            except ImportError:
                raise ImportError("CheMPS2 not available — install pyscf-dmrgscf.")
        else:
            # default PySCF FCI
            pass

        self._apply_spin_constraint()

    def _cas_sizes(self):
        """Return (cas_nelec, cas_norb), defaulting to full embedding space."""
        if self.settings.cas is None:
            return self.Nel, self.Norb
        return self.settings.cas

    def _setup_cas_object(self, mc, cas_norb, cas_nelec):
        """Sync a CASCI/CASSCF mc object with the current mol/mf state.

        After `_setup_mf` the underlying mean-field is DF-wrapped, but the
        CAS object was constructed in __init__ from the plain mf and still
        carries the non-DF `ao2mo` path. We DF-wrap it here so that
        mcscf.mc_ao2mo routes through the 3-center tensor instead of
        looking for `mf._eri` (which no longer exists).
        """
        if hasattr(self.mf, "with_df") and self.mf.with_df is not None:
            if not getattr(mc, "with_df", None):
                mc = mc.density_fit(with_df=self.mf.with_df)
                self.mc = mc  # rebind so the caller sees the wrapped object
            else:
                # mf (and its B tensor) is rebuilt on every _setup_mf; without
                # this refresh the CAS integrals keep the previous cycle's B.
                mc.with_df = self.mf.with_df
        mc.mol = self.mol
        mc._scf = self.mf
        mc.ncas = cas_norb
        # parity guard: a mismatch would silently change the CAS spin
        assert (cas_nelec - self.mol.spin) % 2 == 0, (
            f"cas_nelec={cas_nelec} incompatible with twoS={self.mol.spin}: "
            "put the open shell(s) inside the active space."
        )
        nelecb = (cas_nelec - self.mol.spin) // 2
        mc.nelecas = (cas_nelec - nelecb, nelecb)
        ncorelec = self.mol.nelectron - sum(mc.nelecas)
        assert ncorelec % 2 == 0
        mc.ncore = ncorelec // 2
        mc.mo_coeff = self.mf.mo_coeff
        mc.mo_energy = self.mf.mo_energy

    def _cas_rdm1_to_local_from_dm(self, casdm1_mo, mc, cas_norb):
        """
        Transform CAS RDM1 (MO basis) → local AO basis
        """
        core_norb = mc.ncore
        mo = mc.mo_coeff

        core_MO = mo[:, :core_norb]
        active_MO = mo[:, core_norb : core_norb + cas_norb]
        # core contribution
        coredm1 = core_MO @ core_MO.T * 2
        # active contribution
        casdm1 = lib.einsum(
            "ap,pq,bq->ab", active_MO, casdm1_mo, active_MO, optimize=True
        )

        return coredm1 + casdm1

    def _print_ci_analysis(
        self, ci, cas_norb, neleca, nelecb, root, tol=None, max_det=None
    ):
        """
        Per-state CI summary:
            weight  |α,β>   (SCS: 2 / u / d / 0 per orbital)
        plus natural-orbital occupations from the 1-RDM.

        tol / max_det default to settings.ci_print_tol / settings.ci_max_det
        (explicit arguments still override). They truncate ONLY the printed
        determinant table; the natural occupancies always come from the FULL
        CI vector's 1-RDM and are unaffected by the truncation.

        Conventions
        -----------
        - PySCF: bit i of the determinant string == orbital i.
        - Bit strings printed with ORBITAL 0 ON THE LEFT,
          so reading left→right walks orbitals 0, 1, 2, ...
        - SCS code per orbital:
            (α=1, β=1) -> '2'   doubly occupied
            (α=1, β=0) -> 'u'   single α
            (α=0, β=1) -> 'd'   single β
            (α=0, β=0) -> '0'   empty
        """
        if tol is None:
            tol = getattr(self.settings, "ci_print_tol", 0.1)
        if max_det is None:
            max_det = getattr(self.settings, "ci_max_det", 8)
        from pyscf.fci import addons, direct_spin1

        # Natural-orbital occupations (eigenvalues of the 1-RDM, descending).
        rdm1 = direct_spin1.make_rdm1(ci, cas_norb, (neleca, nelecb))
        occ = np.linalg.eigvalsh(rdm1)[::-1]

        # Dominant determinants, sorted by |coeff|.
        dets = addons.large_ci(
            ci, cas_norb, (neleca, nelecb), tol=tol, return_strs=True
        )
        dets = sorted(dets, key=lambda x: -abs(x[0]))[:max_det]

        def _to_int(s):
            # pyscf may return '0b1010', '1010', or already an int — handle all.
            if isinstance(s, str):
                s = s[2:] if s.startswith("0b") else s
                return int(s, 2)
            return int(s)

        def _orb_bits(s):
            """List of 0/1 with orbital 0 first, length cas_norb."""
            x = _to_int(s)
            return [(x >> i) & 1 for i in range(cas_norb)]

        _scs = {(1, 1): "2", (1, 0): "u", (0, 1): "d", (0, 0): "0"}

        print(f"  State {root}")
        for coeff, stra, strb in dets:
            a, b = _orb_bits(stra), _orb_bits(strb)
            det = f"{''.join(map(str, a))},{''.join(map(str, b))}"
            scs = " ".join(_scs[(ai, bi)] for ai, bi in zip(a, b))
            print(f"    {coeff:+.4f} |{det}>   ({scs})")
        # Coverage: how much of the wavefunction the table accounts for.
        # ~0.99 => extra rows are noise; <<1 => strongly multireference,
        # lower ci_print_tol to see the rest.
        shown = sum(abs(c) ** 2 for c, _, _ in dets)
        print(
            f"    Sum|c|^2 shown = {shown:.4f} "
            f"({len(dets)} dets, tol {tol:g}, max_det {max_det})"
        )
        print(f"    Natural occupancies (full CI 1-RDM): {np.round(occ, 4).tolist()}")

    def _impurity_energy_from_cas_naive(self, mc, cas_norb, RDM1, casdm2):
        Nimp = self.Nimp
        ncore = mc.ncore
        mo = mc.mo_coeff
        core_MO = mo[:, :ncore]
        active_MO = mo[:, ncore : ncore + cas_norb]

        TEI = (
            self.build_full_tei()
        )  # For testing porpuses use since allocates the full TEI
        # only CAS 2-RDM needed
        casdm2_loc = lib.einsum(
            "ip,jq,kr,ls,pqrs->ijkl",
            active_MO,
            active_MO,
            active_MO,
            active_MO,
            casdm2,
            optimize=True,
        )
        # one body term
        one_body = 0.5 * lib.einsum(
            "ij,ij->",
            RDM1[:Nimp, :],
            self.FOCK[:Nimp, :] + self.OEI[:Nimp, :],
            optimize=True,
        )

        two_body = 0.125 * (
            lib.einsum(
                "ijkl,ijkl->",
                casdm2_loc[:Nimp, :, :, :],
                TEI[:Nimp, :, :, :],
                optimize=True,
            )
            + lib.einsum(
                "ijkl,ijkl->",
                casdm2_loc[:, :Nimp, :, :],
                TEI[:, :Nimp, :, :],
                optimize=True,
            )
            + lib.einsum(
                "ijkl,ijkl->",
                casdm2_loc[:, :, :Nimp, :],
                TEI[:, :, :Nimp, :],
                optimize=True,
            )
            + lib.einsum(
                "ijkl,ijkl->",
                casdm2_loc[:, :, :, :Nimp],
                TEI[:, :, :, :Nimp],
                optimize=True,
            )
        )

        if ncore > 0:
            coredm1 = core_MO @ core_MO.T * 2
            casdm1_loc = RDM1 - coredm1
            coredm2 = lib.einsum(
                "pq,rs->pqrs", coredm1, coredm1, optimize=True
            ) - 0.5 * lib.einsum("ps,rq->pqrs", coredm1, coredm1, optimize=True)

            effdm2 = 2 * lib.einsum(
                "pq,rs->pqrs", casdm1_loc, coredm1, optimize=True
            ) - lib.einsum("ps,rq->pqrs", casdm1_loc, coredm1, optimize=True)
            dm2_corr = coredm2 + effdm2

            two_body += 0.125 * (
                lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:Nimp, :, :, :],
                    TEI[:Nimp, :, :, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :Nimp, :, :],
                    TEI[:, :Nimp, :, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :, :Nimp, :],
                    TEI[:, :, :Nimp, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :, :, :Nimp],
                    TEI[:, :, :, :Nimp],
                    optimize=True,
                )
            )

        return one_body + two_body

    def _impurity_energy_from_cas_df(self, mc, cas_norb, RDM1, casdm2):
        """
        Fully DF-based impurity energy.
        No TEI, no dm2_corr, no O(N^4) memory.

        B: (naux, nemb, nemb)
        """

        Nimp = self.Nimp
        ncore = mc.ncore
        mo = mc.mo_coeff

        # MO spaces
        active_MO = mo[:, ncore : ncore + cas_norb]  # (nemb, ncas)
        imp_MO = active_MO[:Nimp, :]  # (Nimp, ncas)

        # One-body
        one_body = 0.5 * lib.einsum(
            "ij,ij->",
            RDM1[:Nimp, :],
            self.FOCK[:Nimp, :] + self.OEI[:Nimp, :],
            optimize=True,
        )

        # DF tensors in CAS basis
        B_cas = lib.einsum(
            "ip,Lij,jq->Lpq",
            active_MO,
            self.B,
            active_MO,
            optimize=True,
        )

        B_imp = lib.einsum(
            "ip,Lij,jq->Lpq",
            imp_MO,
            self.B[:, :Nimp, :],
            active_MO,
            optimize=True,
        )

        # CAS 2-body contribution
        def _perm(dm2, imp_left: bool):
            if imp_left:
                D = lib.einsum("pqrs,Lpq->Lrs", dm2, B_imp, optimize=True)
                return lib.einsum("Lrs,Lrs->", D, B_cas, optimize=True)
            else:
                D = lib.einsum("pqrs,Lrs->Lpq", dm2, B_imp, optimize=True)
                return lib.einsum("Lpq,Lpq->", D, B_cas, optimize=True)

        t2 = _perm(casdm2, True)
        t2 += _perm(casdm2.transpose(1, 0, 3, 2), True)
        t2 += _perm(casdm2.transpose(2, 3, 0, 1), False)
        t2 += _perm(casdm2.transpose(3, 2, 1, 0), False)

        two_body = 0.125 * t2

        # Core + core-active correction
        if ncore > 0:
            core_MO = mo[:, :ncore]

            Dc = core_MO @ core_MO.T * 2  # core density
            Da = RDM1 - Dc  # active density in AO basis

            # Transform densities to CAS basis
            Dc_cas = lib.einsum(
                "pi,pq,qj->ij",
                active_MO,
                Dc,
                active_MO,
                optimize=True,
            )

            Da_cas = lib.einsum(
                "pi,pq,qj->ij",
                active_MO,
                Da,
                active_MO,
                optimize=True,
            )

            # Coulomb
            rho_imp = lib.einsum("Lpq,pq->L", B_imp, Dc_cas, optimize=True)
            rho_cas = lib.einsum("Lrs,rs->L", B_cas, Dc_cas, optimize=True)
            E_coul = np.dot(rho_imp, rho_cas)

            # Exchange (Dc exchange)
            X_imp = lib.einsum("Lps,ps->L", B_imp, Dc_cas, optimize=True)
            X_cas = lib.einsum("Lrq,rq->L", B_cas, Dc_cas, optimize=True)
            E_exch = -0.5 * np.dot(X_imp, X_cas)

            #  Mixed Coulomb
            rho_imp_m = lib.einsum("Lpq,pq->L", B_imp, Da_cas, optimize=True)
            rho_cas_m = lib.einsum("Lrs,rs->L", B_cas, Dc_cas, optimize=True)
            E_mix = 2.0 * np.dot(rho_imp_m, rho_cas_m)

            # Mixed exchange
            X_imp_m = lib.einsum("Lps,ps->L", B_imp, Da_cas, optimize=True)
            X_cas_m = lib.einsum("Lrq,rq->L", B_cas, Dc_cas, optimize=True)
            E_mix -= np.dot(X_imp_m, X_cas_m)

            two_body += 0.125 * (E_coul + E_exch + E_mix)

        return one_body + two_body

    def _nevpt2_fci_roots(self, mc_ci, fcivec, roots, cas_norb):
        """NEVPT2 energies per root"""
        from pyscf import mrpt

        # mrpt.NEVPT has no DF integral path — make sure mc_ci._scf carries
        # the embedding ERI before pt.kernel() runs.
        self._ensure_eri(mc_ci._scf)

        e_casci_nevpt2 = []
        if not isinstance(fcivec, (list, tuple)):
            fcivec = [fcivec]

        print("=" * 45)
        print("NEVPT2 results:")
        print("=" * 45)
        for root in roots:
            ci = fcivec[root]
            ss = mc_ci.fcisolver.spin_square(ci, cas_norb, mc_ci.nelecas)[0]
            e_corr = mrpt.NEVPT(mc_ci, root).kernel()
            e_cas_root = (
                mc_ci.e_tot
                if not isinstance(mc_ci.e_tot, np.ndarray)
                else mc_ci.e_tot[root]
            )
            e_casci_nevpt2.append([ss, e_cas_root, e_cas_root + e_corr])
            self._print_ci_analysis(
                ci, cas_norb, mc_ci.nelecas[0], mc_ci.nelecas[1], root
            )

        return e_casci_nevpt2

    @staticmethod
    def _decompose_nto_cas(t_dm1_cas, thresh=1e-12):
        """SVD of a CAS transition density.

        Returns
        -------
        lambdas : ndarray
            NTO weights, s**2.
        V_hole : ndarray
            Hole/donor NTOs in the CAS basis.
        U_part : ndarray
            Particle/acceptor NTOs in the CAS basis.
        """
        U, s, Vh = np.linalg.svd(t_dm1_cas)
        lam = s**2
        keep = lam > thresh

        return lam[keep], Vh[keep].conj().T, U[:, keep]

    @staticmethod
    def _decompose_ndo_cas(d_dm1_cas, thresh=1e-12):
        """
        Eigendecompose Delta = D^II - D^00 (symmetric). Returns kappa sorted by
        |kappa| desc, W columns, and the scalar descriptors. (J. Chem. Phys. 141, 024106 (2014), Eq 72-76)
        """
        kappa, W = np.linalg.eigh(d_dm1_cas)

        order = np.argsort(-np.abs(kappa))
        kappa, W = kappa[order], W[:, order]
        keep = np.abs(kappa) > thresh
        kappa, W = kappa[keep], W[:, keep]
        d, a = np.minimum(kappa, 0), np.maximum(kappa, 0)

        D_det, D_att = (W * d) @ W.T, (W * a) @ W.T  # Eqs. 71, 73  (= W diag(.) W^T)
        assert np.allclose(
            D_det + D_att, d_dm1_cas
        )  # d + a = kappa ⇒ split exact (to thresh)
        stats = {
            "p_D": d.sum(),
            "p_A": a.sum(),
            "PR_D": d.sum() ** 2 / (d @ d) if np.any(d) else 0.0,
            "PR_A": a.sum() ** 2 / (a @ a) if np.any(a) else 0.0,
        }
        return kappa, W, D_det, D_att, stats

    def _build_nto(self, mc_ci, t_dm1_cas):
        """Transition 1-RDM (ground -> root) + NTOs. SVD in the CAS basis (where
        the rank is exact), then promote the vectors to the embedding (EO) basis
        via orbcas (lambdas are invariant under that isometry).

        Returns (t_dm1_emb, nto_info), nto_info = {'lambdas', 'V_hole', 'U_part'}
        with the orbitals in the (Norb) EO basis.
        """
        lam, V_cas, U_cas = self._decompose_nto_cas(t_dm1_cas)

        orbcas = mc_ci.mo_coeff[:, mc_ci.ncore : mc_ci.ncore + mc_ci.ncas]
        t_dm1_emb = orbcas @ t_dm1_cas @ orbcas.T

        V_eo, U_eo = self._fix_orbital_phases(orbcas @ V_cas, orbcas @ U_cas)
        info = {
            "lambdas": lam,
            "V_hole": V_eo,
            "U_part": U_eo,
            "V_hole_cas": V_cas,
            "U_part_cas": U_cas,
        }
        return t_dm1_emb, info

    def _build_ndo(self, mc_ci, d_dm1_cas):
        """Build EO difference density and EO NDOs from a CAS density."""

        # NDOs decomposition
        kappa, W_cas, D_det, D_att, stats = self._decompose_ndo_cas(d_dm1_cas)
        # CAS orbs in EO basis
        orbcas = mc_ci.mo_coeff[:, mc_ci.ncore : mc_ci.ncore + mc_ci.ncas]

        # CAS -> EO
        def to_eo(M):
            return orbcas @ M @ orbcas.T

        # NDO in EO basis
        W_eo = orbcas @ W_cas

        if W_eo.shape[1]:
            idx = np.abs(W_eo).argmax(axis=0)
            phase = np.sign(W_eo[idx, np.arange(W_eo.shape[1])])
            W_eo *= phase

        info = {
            "kappa": kappa,
            "W": W_eo,
            "D_det": to_eo(D_det),
            "D_att": to_eo(D_att),
            "W_cas": W_cas,
            **stats,
        }
        return to_eo(d_dm1_cas), info

    def _transition_dm1(self, mc_ci, fcivec, root):
        """Get a PySCF CAS transition density and build its NTOs."""
        t_dm1_cas = mc_ci.fcisolver.trans_rdm1(
            fcivec[0], fcivec[root], mc_ci.ncas, mc_ci.nelecas
        )

        return self._build_nto(mc_ci, t_dm1_cas)

    def _difference_dm1(self, mc_ci, fcivec, root):
        """Get PySCF CAS state densities and build their NDOs."""
        from pyscf.fci import direct_spin1

        nelecas = self._nelecas_per_state(len(fcivec))

        def rdm1(r):
            return direct_spin1.make_rdm1(fcivec[r], mc_ci.ncas, nelecas[r])

        return self._build_ndo(mc_ci, rdm1(root) - rdm1(0))

        # return self._build_ndo(mc_ci, dm1_I - dm1_0)

    @staticmethod
    def _fix_orbital_phases(V, U):
        """Anchor each hole column's largest entry real-positive; apply the SAME
        phase to its particle partner so U diag(sqrt(lam)) V^H still rebuilds T.
        """
        V, U = V.copy(), U.copy()
        for j in range(V.shape[1]):
            phase = V[np.argmax(np.abs(V[:, j])), j]
            if phase != 0:
                phase = np.conj(phase) / np.abs(phase)
                V[:, j] *= phase
                U[:, j] *= phase
        return V, U

    def _nelecas_per_state(self, n_states):
        """
        Return a list of (neleca, nelecb) — one per CI vector.

        state_average_mix_: each underlying fcisolver carries its own 2S,
        so the (na, nb) sector differs from state to state.
        plain state_average_ (one solver): all states share self.mc.nelecas.
        """
        cas_nelec = sum(self.mc.nelecas)
        fcisolvers = getattr(self.mc.fcisolver, "fcisolvers", None)

        if fcisolvers is None:
            # Single solver, same sector for every state.
            return [tuple(self.mc.nelecas)] * n_states

        out = []
        for solver in fcisolvers:
            two_s = getattr(solver, "spin", 0)
            neleca = (cas_nelec + two_s) // 2
            nelecb = cas_nelec - neleca
            out.extend([(neleca, nelecb)] * solver.nroots)
        return out
