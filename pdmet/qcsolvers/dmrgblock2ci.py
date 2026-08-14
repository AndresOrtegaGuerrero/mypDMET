"""DMRG-CASCI (block2): one fixed-orbital DMRG solve, no macro loop, + NEVPT2.

CASCI optimizes no orbitals, so state selection is nroots-driven (nroots=1 ->
single state; nroots>1 -> densities averaged with settings.state_percent), not
SS/SA. All block2 plumbing / NEVPT2 / RDM helpers come from the base.

NEVPT2: unlike CASSCF, the main CASCI solve already sits at the final orbitals,
so single-state NEVPT2 reuses that MPS directly -- canonicalizing core/virtual
(cas_natorb=False) leaves the active MPS valid, so no second DMRG solve. Multi-
root / different-sector requests fall back to the base fresh-solve path.
"""

import numpy as np
from pyscf import mcscf
from pdmet.qcsolvers.basedmrg import BaseDMRGBlock2Solver


class DMRGBlock2CISolver(BaseDMRGBlock2Solver):
    def _build_mc(self, cas_norb, cas_nelec):
        return mcscf.CASCI(self.mf, cas_norb, cas_nelec)

    def _fuse_nevpt2(self):
        """True if NEVPT2 can reuse the single main solve (no second DMRG)."""
        if self.settings.nevpt2_roots is None:
            return False
        spin = (
            self.settings.nevpt2_spin
            if self.settings.nevpt2_spin is not None
            else self.settings.twoS
        )
        return (
            self.settings.nroots == 1
            and spin == self.settings.twoS
            and list(self.settings.nevpt2_roots) == [0]
            and self.settings.nevpt2_nroots in (None, 1)
        )

    def kernel(self):
        self._setup_mf()
        self._reset_ntos()
        cas_nelec, cas_norb = self._cas_sizes()

        nroots = self.settings.nroots
        # When NEVPT2 will reuse this solve, make it NEVPT2-ready (restart_dir,
        # noreorder, singlet_embedding) so its MPS matches what NEVPT2 expects.
        path = "casci" if self._fuse_nevpt2() else None
        self.mc.fcisolver = self._get_dmrg_solver(self.settings.twoS, nroots, path=path)
        weights = self.settings.state_percent  # validate() fills uniform if unset
        if nroots > 1:
            self.mc = self.mc.state_average_(weights)

        self._setup_cas_object(self.mc, cas_norb, cas_nelec)
        mo = self._mo_guess(self.mc)
        e_tot, _, fcivec = self.mc.kernel(mo)[:3]
        if nroots > 1:
            e_tot = np.asarray(self.mc.e_states)
        self.mo = self.mo_nat = self.mc.mo_coeff
        if not self.mc.converged:
            print("WARNING: CASCI not converged")

        if nroots == 1:
            e_cell, RDM1 = self._single_root_casscf(fcivec, cas_norb, e_tot)
        else:
            e_cell, RDM1 = self._state_average(fcivec, cas_norb, e_tot, weights)

        if self.settings.nevpt2_roots is not None:
            e_tot = self._run_nevpt2_standard(cas_norb, cas_nelec, e_tot)
        elif self.settings.nto and nroots > 1:
            self._compute_dmrg_ntos(cas_norb, cas_nelec)

        return e_cell, e_tot, RDM1

    def _run_nevpt2_standard(self, cas_norb, cas_nelec, e_tot):
        # Single-state: reuse the main solve's MPS (canonicalize core/virtual,
        # active untouched). Otherwise defer to the base fresh-solve path.
        if not self._fuse_nevpt2():
            return super()._run_nevpt2_standard(cas_norb, cas_nelec, e_tot)

        print("=" * 45)
        mc_ci = self.mc
        n = mc_ci.fcisolver.nroots
        ms, cs, es = [None] * n, [None] * n, [None] * n
        ms[0], cs[0], es[0] = mc_ci.canonicalize(
            self.mc.mo_coeff, ci=0, cas_natorb=False
        )
        e_casci_nevpt2 = self._nevpt2_from_mc_ci(
            mc_ci, ms, cs, es, self.settings.nevpt2_roots, cas_norb
        )
        print("=" * 45)
        return (e_tot, np.asarray(e_casci_nevpt2))
