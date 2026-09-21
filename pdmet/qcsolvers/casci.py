from pdmet.qcsolvers.casbase import BaseCASSolver
from pyscf import mcscf, lib
import numpy as np


class CASCISolver(BaseCASSolver):
    """CASCI - FCI within a active space"""

    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.mo = None
        self.mo_nat = None
        cas_nelec, cas_norb = self._cas_sizes()
        self.mc = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
        self.mc.verbose = settings.verbose
        self.mc.max_memory = settings.max_memory
        self.mc.natorb = True

    def kernel(self, fci_solver="FCI"):
        """
        Run CASCI in the embedding basis
        fci_solver : str - 'FCI' or 'CheMPS2' (For CHEMPS2-CI)

        Returns
        -------
        e_cell   : float        — energy per unit cell
        e_solver : float        — CASCI total energy (tuple if nevpt2_roots set)
        RDM1     : (Norb, Norb) — 1-RDM in local basis
        """

        self._setup_mf()
        self._reset_analysis()
        cas_nelec, cas_norb = self._cas_sizes()
        self._setup_cas_object(self.mc, cas_norb, cas_nelec)
        self._set_fci_solver(fci_solver)
        self.mc.fcisolver.nroots = self.settings.nroots
        mo = self._mo_guess(self.mc)
        e_tot, _, fcivec = self.mc.kernel(mo)[:3]

        self.ci = fcivec
        self.mo_nat = self.mc.mo_coeff

        if not self.mc.converged:
            print("WARNING: CASCI did not converge")
        self.mo = self.mc.mo_coeff

        if self.settings.nroots == 1:
            e_cell, RDM1 = self._single_root(fcivec, cas_norb)
        else:
            e_cell, RDM1 = self._multi_root(fcivec, cas_norb, e_tot)

        if self.settings.nevpt2_roots is not None:
            e_tot = self._run_nevpt2(cas_norb, cas_nelec, e_tot, fci_solver)
        return e_cell, e_tot, RDM1

    def _single_root(self, fcivec, cas_norb):
        self.SS, _ = mcscf.spin_square(self.mc)
        casdm1_mo = self.mc.fcisolver.make_rdm1(fcivec, cas_norb, self.mc.nelecas)
        casdm2_mo = self.mc.fcisolver.make_rdm2(fcivec, cas_norb, self.mc.nelecas)
        RDM1 = self._cas_rdm1_to_local_from_dm(casdm1_mo, self.mc, cas_norb)
        e_cell = self.kmf_ecore + self._impurity_energy_from_cas_df(
            self.mc, cas_norb, RDM1, casdm2_mo
        )

        self._print_ci_analysis(
            fcivec, cas_norb, self.mc.nelecas[0], self.mc.nelecas[1], 0
        )

        return e_cell, RDM1

    def _multi_root(self, fcivec, cas_norb, e_tot):
        RDM1s, e_cells, ss_list = [], [], []
        for i, civec in enumerate(fcivec):
            casdm1_mo = self.mc.fcisolver.make_rdm1(civec, cas_norb, self.mc.nelecas)
            rdm1 = self._cas_rdm1_to_local_from_dm(casdm1_mo, self.mc, cas_norb)
            rdm2 = self.mc.fcisolver.make_rdm2(civec, cas_norb, self.mc.nelecas)
            e_imp = self.kmf_ecore + self._impurity_energy_from_cas_df(
                self.mc, cas_norb, rdm1, rdm2
            )
            ss = self.mc.fcisolver.spin_square(civec, cas_norb, self.mc.nelecas)[0]
            print(
                f"  Root {i}: E(CASCI)={e_tot[i]:12.8f}  E(imp)={e_imp:12.8f}  <S^2>={ss:8.6f}"
            )
            self._print_ci_analysis(
                civec, cas_norb, self.mc.nelecas[0], self.mc.nelecas[1], i
            )

            RDM1s.append(rdm1)
            e_cells.append(e_imp)
            ss_list.append(ss)

        w = np.asarray(self.settings.state_percent)
        RDM1 = lib.einsum("i,ijk->jk", w, RDM1s)
        e_cell = lib.einsum("i,i->", w, e_cells)
        self.SS = np.mean(ss_list)

        # NTOs and NDOs analysis
        if self.settings.nevpt2_nroots is None:
            roots = list(range(len(fcivec)))
            self._analyze_states(self.mc, fcivec, roots)

        return e_cell, RDM1

    def _run_nevpt2(self, cas_norb, cas_nelec, e_tot, solver_name):
        mc_ci = mcscf.CASCI(self.mf, cas_norb, cas_nelec)

        if solver_name == "FCI" and self.settings.e_shift is not None:
            target_SS = 0.5 * self.settings.twoS * (0.5 * self.settings.twoS + 1)
            mc_ci.fix_spin_(shift=self.settings.e_shift, ss=target_SS)

        # reuse the same CI solver (Check if is compatible - this might require refactor)
        mc_ci.fcisolver = self.mc.fcisolver
        mc_ci.fcisolver.nroots = self.settings.nevpt2_nroots
        fcivec = mc_ci.kernel(self.mc.mo_coeff)[2]

        self._analyze_states(mc_ci, fcivec, self.settings.nevpt2_roots)

        e_casci_nevpt2 = np.asarray(
            self._nevpt2_fci_roots(mc_ci, fcivec, self.settings.nevpt2_roots, cas_norb)
        )

        return (e_tot, e_casci_nevpt2)
