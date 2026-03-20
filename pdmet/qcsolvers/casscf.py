from pdmet.qcsolvers.casbase import BaseCASSolver
from pyscf import mcscf, mrpt, lib
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
        cas_nelec, cas_norb = self._cas_sizes()
        self._apply_state_averaging(state_specific_, state_average_, state_average_mix_)
        self._setup_cas_object(self.mc, cas_norb, cas_nelec)
        self._set_fci_solver(fci_solver)
        mo = self._mo_guess(self.mc)

        e_tot, _, fcivec = self.mc.kernel(mo)[:3]
        if state_specific_ is None and state_average_ is not None:
            e_tot = np.asarray(self.mc.e_states)
        if not self.mc.converged:
            print("WARNING: CASSCF not converged")

        self.mo_nat = self.mc.mo_coeff
        self.mo = self.mc.mo_coeff

        if self.settings.nroots == 1 or state_specific_ is not None:
            e_cell, RDM1 = self._single_root_casscf(fcivec, cas_norb, e_tot)
        elif state_average_ is not None:
            e_cell, RDM1 = self._state_average(fcivec, cas_norb, e_tot, state_average_)

        if self.settings.nevpt2_roots is not None:
            e_tot = self._run_nevpt2(cas_norb, cas_nelec, e_tot)

        return e_cell, e_tot, RDM1

    def _apply_state_averaging(
        self, state_specific_, state_average_, state_average_mix_
    ):
        if state_specific_ is not None:
            if "FakeCISolver" not in str(self.mc.fcisolver):
                self.mc = self.mc.state_specific_(state_specific_)
        elif state_average_ is not None and state_average_mix_ is None:
            if "FakeCISolver" not in str(self.mc.fcisolver):
                self.mc = self.mc.state_average_(state_average_)
        elif state_average_mix_ is not None:
            s1, s2, w = state_average_mix_
            mcscf.state_average_mix_(self.mc, [s1, s2], w)
        else:
            self.settings.nroots = 1
            self.mc.fcisolver.nroots = 1

    def _single_root_casscf(self, fcivec, cas_norb, e_tot):
        self.SS = self.mc.fcisolver.spin_square(fcivec, cas_norb, self.mc.nelecas)[0]
        RDM1 = self._cas_rdm1_to_local(fcivec, self.mc, cas_norb)
        e_cell = self.kmf_ecore + self._impurity_energy_from_cas(
            fcivec, self.mc, cas_norb, RDM1
        )
        state_id = self.settings.state_specific_ or 0
        print(
            f"  State {state_id}: E(CASSCF)={e_tot:12.8f}  E_cell={e_cell:12.8f} <S^2>={self.SS:8.6f}"
        )
        return e_cell, RDM1

    def _state_average(self, fcivec, cas_norb, e_tot, weights):
        RDM1s, e_cells = [], []
        ss = self.mc.fcisolver.states_spin_square(fcivec, cas_norb, self.mc.nelecas)[0]
        for i, civec in enumerate(fcivec):
            # Fast Implementation
            rdm1 = self._cas_rdm1_to_local(civec, self.mc, cas_norb)
            e_imp = self.kmf_ecore + self._impurity_energy_from_cas(
                civec, self.mc, cas_norb, rdm1
            )

            print(
                f"  State {i} ({weights[i]:.3f}): E(CASSCF)={e_tot[i]:12.8f}  "
                f"E(imp)={e_imp:12.8f}  <S^2>={ss:8.6f}"
            )
            RDM1s.append(rdm1)
            e_cells.append(e_imp)

        w = np.asarray(weights)
        RDM1 = lib.einsum("i,ijk->jk", w, RDM1s)
        e_cell = lib.einsum("i,i->", w, e_cells)
        self.SS = np.mean(ss)
        return e_cell, RDM1

    def _run_nevpt2(self, cas_norb, cas_nelec, e_tot):
        spin = self.settings.nevpt2_spin or self.settings.twoS
        nelecb = (cas_nelec - spin) // 2
        neleca = cas_nelec - nelecb
        mc_ci = mcscf.CASCI(self.mf, cas_norb, (neleca, nelecb))
        mc_ci.fcisolver.nroots = self.settings.nevpt2_nroots
        fcivec = mc_ci.kernel(self.mc.mo_coeff)[2]

        print("=" * 45)
        e_casci_nevpt2, t_dm1s = [], []
        for root in self.settings.nevpt2_roots:
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
            self._print_ci_analysis(ci, cas_norb, neleca, nelecb, root)
        e_casci_nevpt2 = np.asarray(e_casci_nevpt2)

        print("=" * 45)
        return (e_tot, e_casci_nevpt2, t_dm1s)

    def _transition_dm1(self, mc_ci, fcivec, root, cas_norb):
        """Transition 1-RDM between ground state and excited root."""
        t_dm1 = mc_ci.fcisolver.trans_rdm1(
            fcivec[0], fcivec[root], mc_ci.ncas, mc_ci.nelecas
        )
        orbcas = mc_ci.mo_coeff[:, mc_ci.ncore : mc_ci.ncore + mc_ci.ncas]
        return orbcas @ t_dm1 @ orbcas.T

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

        # Sort and only take the most important determinants for display
        dominant = sorted(dominant, key=lambda x: -abs(x[0]))[:max_det]

        det_str = " + ".join(
            f"{coeff:+.4f}|{stra},{strb}>" for coeff, stra, strb in dominant
        )

        print(f"  State {root}: {det_str}")
        print(f"    Occupancies: {np.round(occ, 4).tolist()}")
