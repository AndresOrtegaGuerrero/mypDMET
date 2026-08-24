"""Shared block2 machinery for DMRG-CASSCF and DMRG-CASCI solvers.

Subclasses override only _build_mc() to pick the mc driver (CASSCF vs CASCI).
SS / SA / SA-mix are runtime kwargs to kernel(), not subclasses.
"""

from pdmet.qcsolvers.casbase import BaseCASSolver
from pyscf import mcscf, lib, mrpt, dmrgscf
import numpy as np
import os


class BaseDMRGBlock2Solver(BaseCASSolver):
    """DMRG solver using Block2 library (base: driver-agnostic)."""

    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self.mo = None
        self.mo_nat = None
        cas_nelec, cas_norb = self._cas_sizes()
        self.mc = self._build_mc(cas_norb, cas_nelec)
        self.mc.verbose = settings.verbose
        self.mc.max_memory = settings.max_memory
        self.mc.natorb = True
        self._init_dmrg_settings()

    def _build_mc(self, cas_norb, cas_nelec):
        raise NotImplementedError("subclass returns mcscf.CASSCF or mcscf.CASCI")

    def _init_dmrg_settings(self):
        dmrgscf.settings.BLOCKVERSION = "2.0.0"  # Specify the Block version being used
        dmrgscf.settings.BLOCKEXE = self.settings.dmrg.BLOCKEXE
        dmrgscf.settings.MPIPREFIX = self.settings.dmrg.MPIPREFIX
        dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = self.settings.dmrg.BLOCKEXE

    def kernel(
        self,
        state_specific_=None,
        state_average_=None,
        state_average_mix_=None,
    ):
        self._setup_mf()
        self._reset_analysis()
        cas_nelec, cas_norb = self._cas_sizes()
        self._apply_state_averaging(state_specific_, state_average_, state_average_mix_)
        self._setup_cas_object(self.mc, cas_norb, cas_nelec)
        mo = self._mo_guess(self.mc)

        self.mo_nat = self.mc.mo_coeff
        self.mo = self.mc.mo_coeff

        e_tot, _, fcivec = self.mc.kernel(mo)[:3]
        if state_specific_ is None:
            if state_average_ is not None or state_average_mix_ is not None:
                e_tot = np.asarray(self.mc.e_states)
        if not self.mc.converged:
            print("WARNING: CASSCF not converged")

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
            self.mc.fcisolver = self._get_dmrg_solver(
                self.settings.twoS, self.settings.nroots
            )
            self.mc = self.mc.state_specific_(state_specific_)
        elif state_average_ is not None:
            self.mc.fcisolver = self._get_dmrg_solver(
                self.settings.twoS,
                self.settings.nroots,
                nto_ndo=self.settings.nto or self.settings.ndo,
            )
            self.mc = self.mc.state_average_(state_average_)
        elif state_average_mix_ is not None:
            solvers = []
            weight_list = []
            for i, solver in enumerate(state_average_mix_):
                # Tne nto/ndo for mix solver might need to be reconsidered
                dmrg_solver = self._get_dmrg_solver(
                    solver.spin, solver.roots, index=i, nto_ndo=solver.nto or solver.ndo
                )
                solvers.append(dmrg_solver)
                weight_list += solver.weights
            mcscf.state_average_mix_(self.mc, solvers, weight_list)

    def _get_dmrg_solver(self, spin, roots, path=None, index=None, nto_ndo=False):
        dmrg_settings = self.settings.dmrg
        base_dir = dmrg_settings.scratch_dir
        if index is not None:
            solver_dir = os.path.join(base_dir, f"solver_{index}")
        else:
            solver_dir = base_dir

        if path is not None:
            solver_dir = os.path.join(solver_dir, path)

        dmrg_solver = dmrgscf.DMRGCI(
            self.mol, maxM=dmrg_settings.maxM, tol=dmrg_settings.tol
        )
        dmrg_solver.spin = spin
        dmrg_solver.nroots = roots
        dmrg_solver.memory = dmrg_settings.memory
        dmrg_solver.conv_tol = dmrg_settings.conv_tol
        dmrg_solver.threads = dmrg_settings.threads
        dmrg_solver.runtimeDir = solver_dir
        dmrg_solver.scratchDirectory = solver_dir

        block_extra_keyword = []
        if path == "casci":
            block_extra_keyword = [
                "restart_dir %s" % (solver_dir + "/restart"),
                "noreorder",
                "singlet_embedding",
            ]

        if nto_ndo and roots > 1:
            block_extra_keyword.append("tran_onepdm")

        dmrg_solver.block_extra_keyword = block_extra_keyword

        return dmrg_solver

    def _single_root_casscf(self, fcivec, cas_norb, e_tot):
        self.SS = self.mc.fcisolver.spin_square(fcivec, cas_norb, self.mc.nelecas)[0]
        casdm1_mo, casdm2_mo = self.mc.fcisolver.make_rdm12(
            fcivec, cas_norb, self.mc.nelecas
        )
        RDM1 = self._cas_rdm1_to_local_from_dm(casdm1_mo, self.mc, cas_norb)
        e_cell = self.kmf_ecore + self._impurity_energy_from_cas_df(
            self.mc, cas_norb, RDM1, casdm2_mo
        )
        state_id = self.settings.state_specific_ or 0
        print(
            f"  State {state_id}: E(DMRG)={e_tot:12.8f}  E_cell={e_cell:12.8f} <S^2>={self.SS:8.6f}"
        )
        return e_cell, RDM1

    def _state_average(self, fcivec, cas_norb, e_tot, weights):
        RDM1s, e_cells = [], []
        ss = self.mc.fcisolver.states_spin_square(fcivec, cas_norb, self.mc.nelecas)[0]
        rdm1s_cas, rdm2s_cas = self.mc.fcisolver.states_make_rdm12(
            fcivec, cas_norb, self.mc.nelecas
        )
        for i, civec in enumerate(fcivec):
            rdm1 = self._cas_rdm1_to_local_from_dm(rdm1s_cas[i], self.mc, cas_norb)
            e_imp = self.kmf_ecore + self._impurity_energy_from_cas_df(
                self.mc, cas_norb, rdm1, rdm2s_cas[i]
            )

            print(
                f"  State {i} ({weights[i]:.3f}): E(DMRG)={e_tot[i]:12.8f}  "
                f"E(imp)={e_imp:12.8f}  <S^2>={ss[i]:8.6f}"
            )
            RDM1s.append(rdm1)
            e_cells.append(e_imp)

        w = np.asarray(weights)
        RDM1 = lib.einsum("i,ijk->jk", w, RDM1s)
        e_cell = lib.einsum("i,i->", w, e_cells)
        self.SS = np.mean(ss)

        if self.settings.nevpt2_nroots is None:
            roots = list(range(len(fcivec)))
            self._analyze_states(self.mc, fcivec, roots, allow_nto=self.settings.nto)

        return e_cell, RDM1

    def _run_nevpt2_mix(self, cas_norb, cas_nelec, e_tot):
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
            mc_ci.fcisolver = self._get_dmrg_solver(
                spin, nevpt2_nroots[i], path="casci", index=i
            )

            mc_ci.fcisolver.nroots = nevpt2_nroots[i]
            if self.settings.e_shift is not None:
                ss = 0.5 * spin * (0.5 * spin + 1)
                mc_ci.fix_spin_(shift=self.settings.e_shift, ss=ss)

            mc_ci.kernel(mo_coeff=self.mc.mo_coeff)

            # Canonicalization for each state
            ms = [None] * mc_ci.fcisolver.nroots
            cs = [None] * mc_ci.fcisolver.nroots
            es = [None] * mc_ci.fcisolver.nroots

            for ir in nevpt2_roots[i]:
                ms[ir], cs[ir], es[ir] = mc_ci.canonicalize(
                    self.mc.mo_coeff, ci=ir, cas_natorb=False
                )

            self._analyze_states(mc_ci, cs, nevpt2_roots[i])

            e_casci_nevpt2.extend(
                self._nevpt2_from_mc_ci(mc_ci, ms, cs, es, nevpt2_roots[i], cas_norb)
            )

        print("=" * 45)
        return (e_tot, np.asarray(e_casci_nevpt2))

    def _run_nevpt2_standard(self, cas_norb, cas_nelec, e_tot):
        print("=" * 45)
        e_casci_nevpt2 = []

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
        mc_ci.fcisolver = self._get_dmrg_solver(spin, nevpt2_nroots, path="casci")

        if self.settings.e_shift is not None:
            ss = 0.5 * spin * (0.5 * spin + 1)
            mc_ci.fix_spin_(shift=self.settings.e_shift, ss=ss)
        mc_ci.kernel(self.mc.mo_coeff)

        # Canonicalization for each state
        ms = [None] * mc_ci.fcisolver.nroots
        cs = [None] * mc_ci.fcisolver.nroots
        es = [None] * mc_ci.fcisolver.nroots

        # for ir in range(mc_ci.fcisolver.nroots):
        for ir in nevpt2_roots:
            ms[ir], cs[ir], es[ir] = mc_ci.canonicalize(
                self.mc.mo_coeff, ci=ir, cas_natorb=False
            )

        self._analyze_states(mc_ci, cs, nevpt2_roots)

        e_casci_nevpt2 = self._nevpt2_from_mc_ci(
            mc_ci, ms, cs, es, nevpt2_roots, cas_norb
        )

        print("=" * 45)
        return (e_tot, np.asarray(e_casci_nevpt2))

    def _nevpt2_from_mc_ci(self, mc_ci, ms, cs, es, roots, cas_norb):
        self._ensure_eri(mc_ci._scf)

        e_casci_nevpt2 = []

        # Apply NEVPT2 correction
        for root in roots:
            ci = cs[root]
            ss = mc_ci.fcisolver.spin_square(ci, cas_norb, mc_ci.nelecas)[0]
            mc_ci.mo_coeff = ms[root]  # root-specific canonicalized MOs
            mc_ci.ci = cs  # full list of CI vectors (all roots)
            mc_ci.mo_energy = es[root]  # root-specific MO energies
            pt = mrpt.NEVPT(mc_ci, root).set(canonicalized=True)
            pt.load_ci = lambda r=root: r
            if self.settings.dmrg.use_compress_nevpt2:
                e_corr = pt.compress_approx(
                    maxM=self.settings.dmrg.nevpt2_maxM
                ).kernel()
            else:
                e_corr = pt.kernel()
            e_cas_root = (
                mc_ci.e_tot
                if not isinstance(mc_ci.e_tot, np.ndarray)
                else mc_ci.e_tot[root]
            )
            e_casci_nevpt2.append([ss, e_cas_root, e_cas_root + e_corr])
        self._print_ci_dmrg(mc_ci)

        return e_casci_nevpt2

    def _transition_dm1(self, mc_ci, fcivec, root):
        """DMRG transition density and NTOs for one root."""
        t_dm1_cas = self._load_dmrg_rdm1(mc_ci, 0, root)

        if t_dm1_cas is None:
            return None, {
                "lambdas": np.array([]),
                "V_hole": None,
                "U_part": None,
            }

        return self._build_nto(mc_ci, t_dm1_cas)

    def _difference_dm1(self, mc_ci, fcivec, root):
        """DMRG difference density and NDOs for one root."""
        dm1_0 = self._load_dmrg_rdm1(mc_ci, 0, 0)
        dm1_I = self._load_dmrg_rdm1(mc_ci, root, root)

        if dm1_0 is None or dm1_I is None:
            empty = {
                "kappa": np.array([]),
                "W": None,
                "D_det": None,
                "D_att": None,
            }
            return None, empty

        return self._build_ndo(mc_ci, dm1_I - dm1_0)

    def _print_ci_dmrg(self, mc_ci):
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes

        path = mc_ci.fcisolver.scratchDirectory + "/restart"
        det_cutoff = self.settings.dmrg.det_cutoff
        driver = DMRGDriver(scratch=path, symm_type=SymmetryTypes.SU2, n_threads=1)
        kets = driver.load_mps(tag="KET", nroots=mc_ci.fcisolver.nroots)

        if mc_ci.fcisolver.nroots == 1:
            print("\nDMRG CI coefficients:")
            csfs, coeffs = driver.get_csf_coefficients(
                kets, cutoff=det_cutoff, iprint=1
            )
        else:
            for i in range(mc_ci.fcisolver.nroots):
                ket = driver.split_mps(kets, iroot=i, tag="KET%d" % i)
                print(f"\nRoot {i} DMRG CI coefficients:")
                csfs, coeffs = driver.get_csf_coefficients(
                    ket, cutoff=det_cutoff, iprint=1
                )

    def _load_dmrg_rdm1(self, mc_ci, bra, ket):
        """Load a Block2 1-RDM and spin-trace it if needed."""
        path = mc_ci.fcisolver.scratchDirectory
        filepath = os.path.join(path, f"node0/1pdm-{bra}-{ket}.npy")

        if not os.path.isfile(filepath):
            print(f"  Warning: 1-RDM file not found: {filepath}")
            return None

        dm = np.load(filepath)

        # Block2: (spin, ncas, ncas) -> spatial 1-RDM
        if dm.ndim == 3:
            dm = dm.sum(axis=0)

        return dm
