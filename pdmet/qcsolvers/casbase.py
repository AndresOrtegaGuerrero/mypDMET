from pdmet.qcsolvers.base import BaseSolver
from pyscf import lib


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

    def _cas_rdm1_to_local(self, ci, mc, cas_norb):
        """
        Build full-space RDM1 in local basis from CAS density matrix.
        """
        core_norb = mc.ncore
        mo = mc.mo_coeff

        core_MO = mo[:, :core_norb]
        active_MO = mo[:, core_norb : core_norb + cas_norb]
        casdm1_mo = mc.fcisolver.make_rdm1(ci, cas_norb, mc.nelecas)
        # core contribution
        coredm1 = core_MO @ core_MO.T * 2
        # active contribution
        casdm1 = lib.einsum(
            "ap,pq,bq->ab", active_MO, casdm1_mo, active_MO, optimize=True
        )

        return coredm1 + casdm1

    def _impurity_energy_from_cas(self, ci, mc, cas_norb, RDM1):
        Nimp = self.Nimp
        ncore = mc.ncore
        mo = mc.mo_coeff
        core_MO = mo[:, :ncore]
        active_MO = mo[:, ncore : ncore + cas_norb]

        # only CAS 2-RDM needed
        casdm2 = mc.fcisolver.make_rdm2(ci, cas_norb, mc.nelecas)
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
                self.TEI[:Nimp, :, :, :],
                optimize=True,
            )
            + lib.einsum(
                "ijkl,ijkl->",
                casdm2_loc[:, :Nimp, :, :],
                self.TEI[:, :Nimp, :, :],
                optimize=True,
            )
            + lib.einsum(
                "ijkl,ijkl->",
                casdm2_loc[:, :, :Nimp, :],
                self.TEI[:, :, :Nimp, :],
                optimize=True,
            )
            + lib.einsum(
                "ijkl,ijkl->",
                casdm2_loc[:, :, :, :Nimp],
                self.TEI[:, :, :, :Nimp],
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
            ) - lib.einsum("ps,rq->pqrs", casdm1_loc, coredm1)
            dm2_corr = coredm2 + effdm2

            two_body += 0.125 * (
                lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:Nimp, :, :, :],
                    self.TEI[:Nimp, :, :, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :Nimp, :, :],
                    self.TEI[:, :Nimp, :, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :, :Nimp, :],
                    self.TEI[:, :, :Nimp, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :, :, :Nimp],
                    self.TEI[:, :, :, :Nimp],
                    optimize=True,
                )
            )

        return one_body + two_body

    def _impurity_energy_from_cas_test(self, ci, mc, cas_norb, RDM1):
        """
        Impurity energy for CAS solvers.

        Efficiency: avoids building full (Norb^4) RDM2 in local basis.
        Instead, each permutation transforms only one index to Nimp,
        reducing cost from O(Norb^4 * Ncas^4) to O(Nimp * Norb^3 * Ncas).

        Valid for both Gamma-point and k-point DMET.
        """
        Nimp = self.Nimp
        ncore = mc.ncore
        mo = mc.mo_coeff

        core_MO = mo[:, :ncore]
        active_MO = mo[:, ncore : ncore + cas_norb]

        casdm2 = mc.fcisolver.make_rdm2(ci, cas_norb, mc.nelecas)

        # --- One-body ---
        one_body = 0.5 * lib.einsum(
            "ij,ij->",
            RDM1[:Nimp, :],
            self.FOCK[:Nimp, :] + self.OEI[:Nimp, :],
            optimize=True,
        )

        # --- Two-body: partial transformation per permutation ---
        # For each permutation we only transform the impurity-projected index
        # This avoids ever allocating the full (Norb,Norb,Norb,Norb) array

        def _contract_imp_idx(imp_mo, full_mo, dm2, TEI_slice):
            """
            Transform dm2 with imp_mo on one index and full_mo on the rest,
            then contract with TEI_slice.

            imp_mo  : active_MO[:Nimp, :]  — (Nimp, Ncas)
            full_mo : active_MO            — (Norb, Ncas)
            Scaling : O(Nimp * Norb^3 * Ncas) vs O(Norb^4 * Ncas^4)
            """
            # Step through indices sequentially — numpy can optimize each step
            tmp = lib.einsum(
                "ip,pqrs->iqrs", imp_mo, dm2, optimize=True
            )  # (Nimp, Ncas, Ncas, Ncas)
            tmp = lib.einsum(
                "jq,iqrs->ijrs", full_mo, tmp, optimize=True
            )  # (Nimp, Norb, Ncas, Ncas)
            tmp = lib.einsum(
                "kr,ijrs->ijks", full_mo, tmp, optimize=True
            )  # (Nimp, Norb, Norb, Ncas)
            tmp = lib.einsum(
                "ls,ijks->ijkl", full_mo, tmp, optimize=True
            )  # (Nimp, Norb, Norb, Norb)
            return lib.einsum("ijkl,ijkl->", tmp, TEI_slice, optimize=True)

        imp_MO = active_MO[:Nimp, :]  # only impurity rows

        # Permutation 1: first index impurity  [:Nimp,:,:,:]
        t2 = _contract_imp_idx(imp_MO, active_MO, casdm2, self.TEI[:Nimp, :, :, :])
        # Permutation 2: second index impurity  [:,:Nimp,:,:]
        # casdm2[p,q,r,s] = casdm2[q,p,s,r] by symmetry → reuse _contract_imp_idx
        t2 += _contract_imp_idx(
            imp_MO, active_MO, casdm2.transpose(1, 0, 3, 2), self.TEI[:, :Nimp, :, :]
        )
        # Permutation 3: third index impurity  [:,:,:Nimp,:]
        t2 += _contract_imp_idx(
            imp_MO, active_MO, casdm2.transpose(2, 3, 0, 1), self.TEI[:, :, :Nimp, :]
        )
        # Permutation 4: fourth index impurity  [:,:,:,:Nimp]
        t2 += _contract_imp_idx(
            imp_MO, active_MO, casdm2.transpose(3, 2, 1, 0), self.TEI[:, :, :, :Nimp]
        )

        two_body = 0.125 * t2

        # --- Core + core-active correction ---
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

            # Core correction: already in local basis — slice directly
            two_body += 0.125 * (
                lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:Nimp, :, :, :],
                    self.TEI[:Nimp, :, :, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :Nimp, :, :],
                    self.TEI[:, :Nimp, :, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :, :Nimp, :],
                    self.TEI[:, :, :Nimp, :],
                    optimize=True,
                )
                + lib.einsum(
                    "ijkl,ijkl->",
                    dm2_corr[:, :, :, :Nimp],
                    self.TEI[:, :, :, :Nimp],
                    optimize=True,
                )
            )

        return one_body + two_body
