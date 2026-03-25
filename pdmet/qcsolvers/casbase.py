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
        from pyscf.fci import addons, direct_spin1

        # RDM1 , occupations
        rdm1 = direct_spin1.make_rdm1(ci, cas_norb, (neleca, nelecb))
        occ = np.linalg.eigvalsh(rdm1)[::-1]

        # determinants in string representation
        dominant = addons.large_ci(
            ci, cas_norb, (neleca, nelecb), tol=tol, return_strs=True
        )

        def _fmt_det(s, cas_norb):
            """Convert '0b11' → '0011' padded to cas_norb digits."""
            return format(int(s, 2), f"0{cas_norb}b")

        # Sort and only take the most important determinants for display
        dominant = sorted(dominant, key=lambda x: -abs(x[0]))[:max_det]

        det_str = " + ".join(
            f"{coeff:+.4f}|{_fmt_det(stra, cas_norb)},{_fmt_det(strb, cas_norb)}>"
            for coeff, stra, strb in dominant
        )

        print(f"  State {root}: {det_str}")
        print(f"    Occupancies: {np.round(occ, 4).tolist()}")

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
