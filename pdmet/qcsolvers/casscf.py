from pdmet.qcsolvers.casbase import BaseCASSolver
from pyscf import mcscf, lib, fci
import numpy as np


class CASSCFSolver(BaseCASSolver):
    """CASSCF - FCI within a active space"""

    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.mo = None
        self.mo_nat = None
        cas_nelec, cas_norb = self._cas_sizes()
        self.mc = mcscf.CASSCF(self.mf, cas_norb, cas_nelec)
        self.mc.verbose = settings.verbose
        self.mc.max_memory = settings.max_memory
        self.mc.natorb = True

    def kernel(
        self,
        fci_solver="FCI",
        state_specific_=None,
        state_average_=None,
        state_average_mix_=None,
    ):
        """
        Run CASSCF in the embedding basis
        fci_solver : str - 'FCI' or 'ChemMPS2' (For DMRG-CI)

        Returns
        -------
        e_cell   : float        — energy per unit cell
        e_solver : float        — CASSCF total energy (tuple if nevpt2_roots set and tdm1)
        RDM1     : (Norb, Norb) — 1-RDM in local basis
        """
        self._setup_mf()
        self._reset_ntos()
        cas_nelec, cas_norb = self._cas_sizes()
        self._apply_state_averaging(state_specific_, state_average_, state_average_mix_)
        self._setup_cas_object(self.mc, cas_norb, cas_nelec)
        self._set_fci_solver(fci_solver)
        mo = self._mo_guess(self.mc)

        e_tot, _, fcivec = self.mc.kernel(mo)[:3]
        if state_specific_ is None:
            if state_average_ is not None or state_average_mix_ is not None:
                e_tot = np.asarray(self.mc.e_states)
        if not self.mc.converged:
            print("WARNING: CASSCF not converged")

        self.mo_nat = self.mc.mo_coeff
        self.mo = self.mc.mo_coeff

        if self.settings.nroots == 1 or state_specific_ is not None:
            e_cell, RDM1 = self._single_root_casscf(fcivec, cas_norb, e_tot)
        elif state_average_ is not None:
            e_cell, RDM1 = self._state_average(fcivec, cas_norb, e_tot, state_average_)
        elif state_average_mix_ is not None:
            weights = []
            for solver in state_average_mix_:
                weights += solver.weights
            print(f"State-average mixing weights: {weights}")
            e_cell, RDM1 = self._state_average(fcivec, cas_norb, e_tot, weights)

        if self.settings.nevpt2_roots is not None:
            if state_average_mix_ is not None:
                e_tot = self._run_nevpt2_mix(cas_norb, cas_nelec, e_tot)
            else:
                e_tot = self._run_nevpt2_standard(cas_norb, cas_nelec, e_tot)

        return e_cell, e_tot, RDM1

    def _apply_state_averaging(
        self, state_specific_, state_average_, state_average_mix_
    ):
        if state_specific_ is not None:
            if "FakeCISolver" not in str(self.mc.fcisolver):
                self.mc = self.mc.state_specific_(state_specific_)
        elif state_average_ is not None:
            if "FakeCISolver" not in str(self.mc.fcisolver):
                print(f"Applying state-average with weights: {state_average_}")
                self.mc = self.mc.state_average_(state_average_)
        elif state_average_mix_ is not None:
            solvers = []
            weight_list = []
            for solver in state_average_mix_:
                fci_solver = fci.addons.fix_spin(fci.direct_spin1.FCI(), ss=solver.spin)
                fci_solver.spin = solver.spin
                fci_solver.nroots = solver.roots
                fci_solver.max_cycle = (
                    120  # Hardcoded for now, can be made a user input if needed
                )
                solvers.append(fci_solver)
                weight_list += solver.weights

            mcscf.state_average_mix_(self.mc, solvers, weight_list)
        else:
            self.settings.nroots = 1
            self.mc.fcisolver.nroots = 1

    def _single_root_casscf(self, fcivec, cas_norb, e_tot):
        self.SS = self.mc.fcisolver.spin_square(fcivec, cas_norb, self.mc.nelecas)[0]
        casdm1_mo = self.mc.fcisolver.make_rdm1(fcivec, cas_norb, self.mc.nelecas)
        casdm2_mo = self.mc.fcisolver.make_rdm2(fcivec, cas_norb, self.mc.nelecas)
        RDM1 = self._cas_rdm1_to_local_from_dm(casdm1_mo, self.mc, cas_norb)
        e_cell = self.kmf_ecore + self._impurity_energy_from_cas_df(
            self.mc, cas_norb, RDM1, casdm2_mo
        )
        state_id = self.settings.state_specific_ or 0
        print(
            f"  State {state_id}: E(CASSCF)={e_tot:12.8f}  E_cell={e_cell:12.8f} <S^2>={self.SS:8.6f}"
        )
        self._print_ci_analysis(
            fcivec, cas_norb, self.mc.nelecas[0], self.mc.nelecas[1], state_id
        )
        return e_cell, RDM1

    def _state_average(self, fcivec, cas_norb, e_tot, weights):
        RDM1s, e_cells = [], []
        ss = self.mc.fcisolver.states_spin_square(fcivec, cas_norb, self.mc.nelecas)[0]
        rdm1s_cas, rdm2s_cas = self.mc.fcisolver.states_make_rdm12(
            fcivec, cas_norb, self.mc.nelecas
        )
        nelecas_list = self._nelecas_per_state(len(fcivec))
        for i, civec in enumerate(fcivec):
            rdm1 = self._cas_rdm1_to_local_from_dm(rdm1s_cas[i], self.mc, cas_norb)
            e_imp = self.kmf_ecore + self._impurity_energy_from_cas_df(
                self.mc, cas_norb, rdm1, rdm2s_cas[i]
            )

            print(
                f"  State {i} ({weights[i]:.3f}): E(CASSCF)={e_tot[i]:12.8f}  "
                f"E(imp)={e_imp:12.8f}  <S^2>={ss[i]:8.6f}"
            )
            neleca_i, nelecb_i = nelecas_list[i]
            self._print_ci_analysis(civec, cas_norb, neleca_i, nelecb_i, i)
            # self._print_ci_analysis(
            #    civec, cas_norb, self.mc.nelecas[0], self.mc.nelecas[1], i
            # )
            RDM1s.append(rdm1)
            e_cells.append(e_imp)

        w = np.asarray(weights)
        RDM1 = lib.einsum("i,ijk->jk", w, RDM1s)
        e_cell = lib.einsum("i,i->", w, e_cells)
        self.SS = np.mean(ss)

        # NTOs straight from the state-averaged CASSCF. Skip when nevpt2_roots is
        # set -- the NEVPT2 path computes them, and doing both would double them.
        # Only valid within one (na, nb) sector; state-average-mix mixes sectors.
        if self.settings.nto and self.settings.nevpt2_roots is None:
            if all(n == nelecas_list[0] for n in nelecas_list):
                self._compute_ntos(self.mc, fcivec, list(range(len(fcivec))), cas_norb)
            else:
                print(
                    "[nto] skipping NTOs for state-average-mix (mixed spin "
                    "sectors); spin-resolved NTOs are not implemented yet."
                )
        return e_cell, RDM1

    def _run_nevpt2_mix(self, cas_norb, cas_nelec, e_tot):
        from copy import copy

        print("=" * 45)
        e_casci_nevpt2 = []

        solvers = self.mc.fcisolver.fcisolvers
        nevpt2_roots = self.settings.nevpt2_roots
        nevpt2_nroots = self.settings.nevpt2_nroots

        # Iterate for each solver
        for i, solver in enumerate(solvers):
            spin = solver.spin
            nelecb = (cas_nelec - spin) // 2
            neleca = cas_nelec - nelecb
            mc_ci = mcscf.CASCI(self.mf, cas_norb, (neleca, nelecb))
            mc_ci.fcisolver = copy(solver)
            mc_ci.fcisolver.nroots = nevpt2_nroots[i]
            fcivec = mc_ci.kernel(self.mc.mo_coeff)[2]

            # NTOs from the pristine wavefunction, BEFORE NEVPT2 canonicalizes.
            self._compute_ntos(mc_ci, fcivec, nevpt2_roots[i], cas_norb)
            e_casci_nevpt2.extend(
                self._nevpt2_fci_roots(mc_ci, fcivec, nevpt2_roots[i], cas_norb)
            )

        print("=" * 45)
        return (e_tot, np.asarray(e_casci_nevpt2))

    def _run_nevpt2_standard(self, cas_norb, cas_nelec, e_tot):
        print("=" * 45)

        spin = (
            self.settings.nevpt2_spin
            if self.settings.nevpt2_spin is not None
            else self.settings.twoS
        )
        nevpt2_nroots = self.settings.nevpt2_nroots
        nevpt2_roots = self.settings.nevpt2_roots

        nelecb = (cas_nelec - spin) // 2
        neleca = cas_nelec - nelecb
        mc_ci = mcscf.CASCI(self.mf, cas_norb, (neleca, nelecb))
        mc_ci.fcisolver.nroots = nevpt2_nroots

        if self.settings.e_shift is not None:
            ss = 0.5 * spin * (0.5 * spin + 1)
            mc_ci.fix_spin_(shift=self.settings.e_shift, ss=ss)

        fcivec = mc_ci.kernel(self.mc.mo_coeff)[2]

        # NTOs from the pristine wavefunction, BEFORE NEVPT2 canonicalizes it.
        self._compute_ntos(mc_ci, fcivec, nevpt2_roots, cas_norb)
        e_casci_nevpt2 = self._nevpt2_fci_roots(mc_ci, fcivec, nevpt2_roots, cas_norb)

        print("=" * 45)
        return (e_tot, np.asarray(e_casci_nevpt2))
