from pdmet.qcsolvers.base import BaseSolver
from pyscf import lib
import numpy as np


class BaseCASSolver(BaseSolver):
    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.mc = None
        self.ci = None

    def _mo_guess(self, mc):
        """Return initial guess MOs for CASCI/CASSCF, defaulting to current MF MOs."""
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
        """Sync a CASCI/CASSCF mc object with the current mol/mf state."""
        mc.mol = self.mol
        mc._scf = self.mf
        mc.ncas = cas_norb
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
        self, ci, cas_norb, neleca, nelecb, root, tol=0.1, max_det=4
    ):
        """
        Per-state CI summary:
            weight  |α,β>   (SCS: 2 / u / d / 0 per orbital)
        plus natural-orbital occupations from the 1-RDM.

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
        print(f"    Natural occupancies: {np.round(occ, 4).tolist()}")

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
        from pyscf import mrpt

        e_casci_nevpt2, t_dm1s = [], []

        # Ensure fcivec is always a list of CI vectors
        if not isinstance(fcivec, (list, tuple)):
            fcivec = [fcivec]
        # Apply NEVPT2 correction

        # Print header
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
            t_dm1s.append(self._transition_dm1(mc_ci, fcivec, root, cas_norb))
            e_casci_nevpt2.append([ss, e_cas_root, e_cas_root + e_corr])
            self._print_ci_analysis(
                ci, cas_norb, mc_ci.nelecas[0], mc_ci.nelecas[1], root
            )

        return e_casci_nevpt2, t_dm1s

    def _transition_dm1(self, mc_ci, fcivec, root, cas_norb):
        """Transition 1-RDM between ground state and excited root."""
        t_dm1 = mc_ci.fcisolver.trans_rdm1(
            fcivec[0], fcivec[root], mc_ci.ncas, mc_ci.nelecas
        )
        orbcas = mc_ci.mo_coeff[:, mc_ci.ncore : mc_ci.ncore + mc_ci.ncas]
        return orbcas @ t_dm1 @ orbcas.T

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
