#!/usr/bin/env python -u
"""
pDMET: Density Matrix Embedding theory for Periodic Systems
Copyright (C) 2022 Abhishek Mitra and Hung Q. Pham. All Rights Reserved.
A few functions in pDMET are modifed from QC-DMET Copyright (C) 2015 Sebastian Wouters

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import numpy as np
from pyscf import lib
from scipy import optimize
from pdmet import localbasis, diis, df_hamiltonian
from pdmet.schmidtbasis import get_bath_using_RHF_1RDM
from pdmet.tools import tchkfile, tplot, tprint, tunix, misc
from pdmet.lib import libdmet
from pdmet.settings import EmbeddingSettings, SolverSettings, SCFSettings

import pywannier90


class pDMET:
    def __init__(
        self,
        cell,
        kmf,
        w90,
        solver="HF",
        state_average_mix_=None,
        nevpt2_spin=None,
        verbose=0,
        max_memory=4000,
    ):
        """
        Args:
            cell                           : a pyscf Cell object
            kmf                            : a rhf wave function from pyscf/pbc
            w90                            : a converged pywannier90 object
            solver                         : name of the quantum chemistry solver for the impurity problem.
            umat                            : correlation potential
            chempot                         : global chemical potential
            emb_corr_1RDM                   : correlated 1RDM from high-level calculations
            emb_orbs                        : a list of the fragment and bath orbitals for each fragment
        Return:

        """

        tprint.print_header()
        # Chkfiles:
        self.cell = cell
        self.kmf = kmf
        self.w90 = w90
        self.kmf_chkfile = None
        self.w90_chkfile = None

        # Self consistency options
        self.scf = SCFSettings()
        # Embedding options
        self.emb = EmbeddingSettings()
        # Solver options
        self.solver = SolverSettings(
            name=solver,
            nevpt2_spin=nevpt2_spin,
            state_average_mix_=state_average_mix_,
            verbose=verbose,
            max_memory=max_memory,
        )

        # DMET Output
        self.verbose = verbose
        self.max_memory = max_memory
        self.loc_OEH_kpts = None
        self.loc_1RDM_kpts = None
        self.loc_1RDM_R0 = None
        self.loc_corr_1RDM_R0 = None
        self.baths = None
        self.emb_corr_1RDM = None
        self.emb_orbs = None
        self.emb_core_orbs = None
        self.emb_mf_1RDM = None
        self.e_tot = 0.0  # energy per unit cell
        self.e_corr = 0.0
        self.nelec_per_cell = None

        # Others
        self.chkfile = "pdmet.chk"  # Save integrals in the WFs basis as well as chem potential and uvec
        self.restart = False  # Run a calculation using saved chem potential and uvec
        self._cycle = 1

    def initialize(self):
        """
        Prepare the local integrals, correlation/chemical potential
        """
        tprint.print_msg("Initializing ...")
        self._load_checkfiles()
        self._detect_rohf()
        self._setup_xc()
        self._initialize_local_basis()
        self._initialize_impurity()
        self._initialize_embedding_settings()
        self._initialize_correlation_potential()
        self._initialize_scf_settings()
        self._initialize_qcsolver()
        tprint.print_msg("Initializing ... DONE")

    def _load_checkfiles(self):
        """Load saved kmf/w90 object if checkfiles are set"""
        assert (self.chkfile is None) or isinstance(self.chkfile, str)
        self.kmesh = self.w90.mp_grid_loc
        if self.kmf_chkfile is not None and hasattr(self.kmf.with_df, "_cderi"):
            self.kmf = tchkfile.load_kmf(
                self.cell,
                self.kmf,
                self.kmesh,
                self.kmf_chkfile,
                max_memory=self.max_memory,
            )
            if self.kmf.with_df._cderi is None:
                if tunix.check_exist("gdf.h5"):
                    self.kmf.with_df._cderi = "gdf.h5"
                else:
                    print(
                        "WARNING: Provide density fitting file in initiating kmf object or make sure the saved kmf object is using the same density fitting"
                    )
        if self.w90_chkfile is not None:
            self.w90 = tchkfile.load_w90(self.w90, self.w90_chkfile)

        if self.kmf.exxdiv is not None:
            raise Exception(
                "The pDMET has not been developed for the RHF calculation with exxdiv is not None"
            )
            # TODO: if self.kmf.exxdiv != None, consider to run two SCF (one with and one without exx treatment
            # if self.kmf.exxdiv == 'ewald': actOEI_kpts += self.exxdiv_ewald(cell)
            # to get the finite correction, see https://github.com/pyscf/pyscf/issues/250

    def _detect_rohf(self):
        """Determine if the mean-field is ROHF and validate spin settings."""
        from pyscf.pbc import scf

        self._is_ROHF = isinstance(self.kmf, scf.krohf.KROHF) or getattr(
            self.kmf, "_is_ROHF", False
        )

        if self.solver.twoS is None:
            self.solver.twoS = self.cell.spin
        elif self.solver.twoS != self.cell.spin:
            tprint.print_msg(
                " WARNING: the 2S in DMET is different from that of the mean-field wave function. \
                               Hope you know what you're doing"
            )

    def _setup_xc(self):
        """Setup the exchange-correlation functional for DFT embedding."""
        xc = self.emb.xc
        self.emb.dft_CF = xc is not None

        if xc is not None:
            self.emb.OEH_type = xc

            if xc == "RSH-PBE0" and self.emb.xc_omega is None:
                self.emb.xc_omega = 0.2

    def _initialize_local_basis(self):
        """Create the local basis from Pywannier90 object"""
        self.kpts = self.kmf.kpts
        self.Nkpts = self.kpts.shape[0]

        self.local = localbasis.Local(
            self.cell, self.kmf, self.w90, self._is_ROHF, self.emb.xc_omega
        )
        self.e_core = self.local.e_core

        self.Norbs = self.local.nlo * self.Nkpts
        self.Nelec_total = self.local.nelec_total
        self.Nelec_per_cell = self.local.nelec_per_cell
        self.numPairs = self.Nelec_per_cell // 2

    def _initialize_impurity(self):
        """Initialize the impurity system."""
        if self.emb.impCluster is not None:
            assert np.prod(self.kmesh) == 1, (
                "impCluster is used only for a Gamma-point sampling calculation"
            )

            self._impOrbs, self._impAtms = misc.make_imp_orbs(
                self.cell,
                self.w90,
                self.emb.impCluster,
                threshold=self.emb.impOrbs_threshold,
                rm_list=self.emb.impOrbs_rmlist,
                add_list=self.emb.impOrbs_addlist,
            )

            self.Nimp = np.sum(self._impOrbs)
            self._is_gamma = True

            self._print_impurity_cluster()

        else:
            self.Nimp = self.local.nlo
            self._impOrbs = None
            self._impAtms = None
            self._is_gamma = False

            assert self.solver.twoS == 0, (
                "ROHF bath is only available for Gamma-sampling calculation"
            )

    def _print_impurity_cluster(self):
        tprint.print_msg("==== Impurity cluster ====")
        tprint.print_msg(f" No. of Impurity atoms   : {len(self.emb.impCluster)}")
        tprint.print_msg(f" No. of Impurity orbitals: {self.Nimp}")

        atom_coords = self.cell.atom_coords() * lib.param.BOHR

        for i, atm in enumerate(self.emb.impCluster):
            symbol = self.cell.atom_symbol(atm - 1)
            x, y, z = atom_coords[atm - 1]

            tprint.print_msg(f"  {atm:3d}  {symbol:3s}  {x:3.5f} {y:3.5f} {z:3.5f}")

            impAtms = self._impAtms[i].tolist()
            nimpOrbs = len(impAtms)

            impAtms = [nimpOrbs] + impAtms

            tprint.print_msg(
                ("       {:d} Orbitals: " + "{:d} " * nimpOrbs).format(*impAtms)
            )

        tprint.print_msg("==========================")

    def _initialize_embedding_settings(self):
        """Initialize the embedding settings."""
        # The number of bath orbitals depends on whether one does Schmidt decomposition on RHF or ROHF wave function
        self.bathtype = "ROHF" if self._is_ROHF else "RHF"
        if self.scf.CF_type in ["diagF", "diagFB"]:
            self.Nterms = self.Nimp
        else:
            self.Nterms = self.Nimp * (self.Nimp + 1) // 2

        self.mask = self.make_mask(self._is_gamma)

        self.mask4Gamma = self.mask if self._is_gamma else None

        self.H1start, self.H1row, self.H1col = self.make_H1(
            self._is_gamma, self._impOrbs
        )[1:4]  # Use in the calculation of 1RDM derivative

    def _initialize_correlation_potential(self):
        self.chempot = 0.0

        # If DFT CF is used, the correlation potential is initialized as the DFT XC potential
        if self.emb.dft_CF:
            self.uvec = df_hamiltonian.get_init_uvec(self.emb.xc, self.emb.dft_HF)
            self.bounds = df_hamiltonian.get_bounds(
                self.emb.xc,
                self.emb.dft_CF_constraint,
                self.emb.dft_HF,
            )

        else:
            self.uvec = np.zeros(self.Nterms, dtype=np.float64)

        self.umat = self.uvec2umat(self.uvec)

    def _initialize_restart(self):
        self.restart_success = False

        if self.chkfile is not None and self.restart:
            if tunix.check_exist(self.chkfile):
                self.save_pdmet = tchkfile.load_pdmet(self.chkfile)

                self.chempot = self.save_pdmet.chempot
                self.uvec = self.save_pdmet.uvec
                self.umat = self.save_pdmet.umat

                self.emb_corr_1RDM = self.save_pdmet.actv1RDMloc
                self.emb_orbs = self.save_pdmet.emb_orbs
                self.emb_core_orbs = self.save_pdmet.emb_core_orbs

                tprint.print_msg("-> Load the pDMET chkfile")

                self.restart_success = True

            else:
                tprint.print_msg("-> Cannot load the pDMET chkfile")

    def _initialize_scf_settings(self):
        if self.scf.alt_CF:
            pass
        else:
            self.CF = self.cost_func
            self.CF_grad = self.cost_func_grad

        if self.scf.use_DIIS:
            self._diis = diis.DIIS(self.scf.DIIS_start, self.scf.DIIS_nvector)

        self.scf.validate()

    def _initialize_qcsolver(self):
        self._SS = 0.5 * self.solver.twoS * (0.5 * self.solver.twoS + 1)
        self.solver.validate()
        self.qcsolver = self.solver.build_solver(is_KROHF=self._is_ROHF)

    def _build_pdft_context(self):
        from pdmet.qcsolvers.caspdft import PDFTContext

        """Build PDFTContext from local + pDMET state."""
        return PDFTContext(
            cell=self.cell,
            kmf=self.kmf,
            local=self.local,
            emb_orbs=self.emb_orbs,
            core_orbs=self.core_orbs,
            emb_core_orbs=self.emb_core_orbs,
            OEH_type=self.emb.OEH_type,
            mask4Gamma=self.mask4Gamma,
            otxc=self.solver.otxc,
        )

    def kernel(self, chempot=0.0):
        """
        This is the main kernel for DMET calculation.
        It is solving the embedding problem, then returning the total number of electrons per unit cell
        and updating the schmidt orbitals and 1RDM.
        Args:
            chempot                    : global chemical potential to adjust the number of electrons in the unit cell
        Return:
            nelecs                     : the total number of electrons
        Update the class attributes:
            energy                          : the energy for the unit cell
            nelec                           : the number of electrons for the unit cell
            emb_corr_1RDM                   : correlated 1RDM for the unit cell
        """
        from pdmet.qcsolvers.registry import dispatch

        if self._is_new_bath:
            ao2eo = self.local.get_ao2eo(self.emb_orbs)
            self.emb_OEI = self.local.get_emb_OEI(ao2eo)
            # Original way of getting TEI, but allocates the full TEI in memory
            # self.emb_TEI = (
            #     self.local.get_emb_TEI(ao2eo)
            #     if self.emb.use_GDF
            #     else self.local.get_TEI(ao2eo)
            # )
            self.emb_TEI = (
                self.local.get_emb_B(ao2eo)
                if self.emb.use_GDF
                else self.local.get_TEI(ao2eo)
            )
            self.emb_mf_1RDM = self.local.loc_kpts_to_emb(
                self.loc_1RDM_kpts, self.emb_orbs
            )
            self.emb_JK = self.local.get_emb_JK(self.loc_1RDM_kpts, ao2eo)
            # self.emb_coreJK = self.local.get_emb_coreJK(
            #     self.emb_JK, self.emb_TEI, self.emb_mf_1RDM
            # )
            self.emb_coreJK = self.local.get_emb_coreJK_df(
                self.emb_JK, self.emb_TEI, self.emb_mf_1RDM
            )
            self.ao2eo = ao2eo

        # Emb 1-RDM guess
        # Gamma-point: chempot=0 always - use precomputed mf_1RDM
        # k-points   : chempot varies
        # ------------------------------------------------------------------
        if self._is_gamma:
            emb_guess_1RDM = self.emb_mf_1RDM
        else:
            emb_FOCK = self.emb_OEI + self.emb_coreJK
            emb_guess_1RDM = self.local.get_emb_guess_1RDM(
                emb_FOCK, self.Nelec_in_emb, self.Nimp, chempot
            )
        if self._cycle == 1:
            tprint.print_msg(
                "   Embedding size: %2d electrons in (%2d impurities + %2d baths )"
                % (self.Nelec_in_emb, self.Nimp, self.Nbath)
            )

        # ------------------------------------
        # Initialize and run QC solver
        self.qcsolver.initialize(
            self.local.e_core,
            self.emb_OEI,
            self.emb_TEI,
            self.emb_coreJK,
            emb_guess_1RDM,
            self.Nimp + self.Nbath,
            self.Nelec_in_emb,
            self.Nimp,
            chempot,
        )

        # Build PDFTContext only when needed
        pdft_context = None
        if "CASPDFT" in self.solver.name:
            pdft_context = self._build_pdft_context()

        # Launch the kernel of the solver with settings
        e_cell, e_solver, RDM1 = dispatch(self.qcsolver, self.solver, pdft_context)
        # ------------------------------------
        # ------------------------------------
        # Update correlated 1-RDM
        self.emb_corr_1RDM = RDM1
        self.loc_corr_1RDM_R0 = lib.einsum(
            "Rim,mn,jn->Rij", self.emb_orbs, RDM1, self.emb_orbs[0].conj()
        )

        if not np.isclose(self._SS, self.qcsolver.SS):
            tprint.print_msg(
                "           WARNING: Spin contamination. Computed <S^2>: %10.8f, Target: %10.8f"
                % (self.qcsolver.SS, self._SS)
            )
        # ------------------------------------
        # ------------------------------------
        # Compute cell energy
        if self._is_gamma:
            self._compute_gamma_energy(e_cell, e_solver, RDM1)
        else:
            self.nelec_per_cell = np.trace(RDM1[: self.Nimp, : self.Nimp])
            self.e_tot = e_cell

        return self.nelec_per_cell

    def _compute_gamma_energy(self, e_cell, e_solver, RDM1):
        """
        Compute total energy for Gamma-point DMET
        E_total = E_solver + E_core(DMET) + E_core(Wannier)
            E_solver = From QC solver in embedding space
            E_core(DMET) = From frozen bath orbitals
            E_core(Wannier) = From frozen Bloch bands (local)
        """
        # --- DMET core energy (unentangled bath orbitals) ---
        if self._is_new_bath:
            if self.core_orbs is not None and self.core_orbs.shape[-1] != 0:
                self.core_energy, self.loc_core_1RDM = self._compute_dmet_core_energy(
                    RDM1
                )
            else:
                self.core_energy = 0.0
                self.loc_core_1RDM = np.zeros(
                    (1, self.Norbs, self.Norbs)
                )  # or should be

        self.loc_corr_1RDM_R0 += self.loc_core_1RDM
        self.nelec_per_cell = self.Nelec_total

        # Unpack e_solver
        if "CASPDFT" in self.solver.name:
            e_cas, e_pdft = e_solver
            self.e_tot = np.asarray(e_pdft) + self.core_energy + self.local.e_core
            self.e_cas = np.asarray(e_cas) + self.core_energy + self.local.e_core
            self.e_emb = np.asarray(e_pdft)
            self.e_imp = e_cell - self.local.e_core

        elif self.solver.nevpt2_roots is not None:
            e_CAS, e_CASCI_NEVPT2, t_dm1s = e_solver
            self.e_tot = e_CAS + self.core_energy + self.local.e_core
            self.e_emb = e_CAS
            self.e_imp = e_cell - self.local.e_core
            self.e_casci_tot = (
                np.asarray(e_CASCI_NEVPT2[:, 1]) + self.core_energy + self.local.e_core
            )
            self.e_nevpt2_tot = (
                np.asarray(e_CASCI_NEVPT2[:, 2]) + self.core_energy + self.local.e_core
            )
            self.ss_CASCI = e_CASCI_NEVPT2[:, 0]
            self.t_dm1s = t_dm1s
        else:
            self.e_tot = e_solver + self.core_energy + self.local.e_core
            self.e_emb = e_solver
            self.e_imp = e_cell - self.local.e_core

    def _compute_dmet_core_energy(self, RDM1):
        """Compute energy contribution from DMET-frozen (unentangled) bath orbitals
        E_core = Tr[h_core * D_core] + 0.5 * Tr[JK_core * D_core]
        D_core is the MF 1-RDM projected onto core orbitals.
        """
        ao2core = self.local.get_ao2core(self.core_orbs)
        lo2core = self.local.get_lo2core(self.core_orbs)
        core_OEI = self.local.get_core_OEI(ao2core)

        Nelec_in_core = self.Nelec_total - self.Nelec_in_emb
        core_1RDM = self.local.get_core_mf_1RDM(
            lo2core, Nelec_in_core, self.loc_OEH_kpts
        )
        loc_core_1RDM = lib.einsum(
            "kim,mn,kjn->kij", lo2core, core_1RDM, lo2core.conj()
        )
        core_JK = self.local.get_core_JK(ao2core, loc_core_1RDM)
        core_energy = np.sum((core_OEI + 0.5 * core_JK) * core_1RDM).real

        # Update modified 1-RDM for post-processing
        self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = (
            self.local.make_loc_1RDM(
                0.0,
                self.mask4Gamma,
                OEH_type=self.emb.OEH_type,
                dft_HF=None,
            )
        )
        Norb = self.Nimp + self.Nbath
        self.loc_1RDM_R0_modified = self.loc_1RDM_R0.copy()
        self.loc_1RDM_R0_modified[0][:Norb, :Norb] = RDM1
        self.loc_1RDM_R0_modified_ao_basis = lib.einsum(
            "Rim,mn,jn->Rij",
            self.local.ao2lo,
            self.loc_1RDM_R0_modified[0],
            self.local.ao2lo.conj()[0],
        ).real
        # ------------------------------------------
        loc_core_1RDM_reshaped = loc_core_1RDM.real.reshape(1, self.Norbs, self.Norbs)
        return core_energy, loc_core_1RDM_reshaped

    def bath_construction(self, loc_1RDM_R0, impCluster):
        """Get the bath orbitals"""
        emb_orbs, core_orbs, Nelec, Nbath = get_bath_using_RHF_1RDM(
            loc_1RDM_R0,
            impCluster,
            is_ROHF=self._is_ROHF,
            num_bath=self.emb.num_bath,
            bath_truncation=self.emb.bath_truncation,
        )
        self.emb_core_orbs = np.hstack([emb_orbs, core_orbs])

        # Fix num_bath after first cycle
        if self.emb.num_bath is None:
            self.emb.num_bath = Nbath

        Nemb = self.Nimp + Nbath
        Nenv = self.Norbs - Nemb

        # Reshape to k-space
        emb_orbs = emb_orbs.reshape(self.Nkpts, self.local.nlo, Nemb)  # NR = Nkpts
        core_orbs = core_orbs.reshape(self.Nkpts, self.local.nlo, self.local.nlo - Nemb)

        Nelec_in_emb = min(Nelec, self.Nelec_total)
        if self.Nelec_total - Nelec_in_emb > 2 * Nenv:
            Nelec_in_emb = self.Nelec_total - 2 * Nenv

        # Set the flag to indicate bath orbitals updated
        self._is_new_bath = True

        return emb_orbs, core_orbs, Nbath, Nelec_in_emb

    def check_exact(self, error=1.0e-6):
        """
        Do one-shot DMET, only the chemical potential is optimized
        """

        tprint.print_msg("-" * 60)
        if self.emb.dft_CF:
            umat = df_hamiltonian.get_init_uvec(self.emb.xc)
        else:
            umat = 0.0

        self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = (
            self.local.make_loc_1RDM(
                umat, self.mask4Gamma, OEH_type=self.emb.OEH_type, dft_HF=None
            )
        )

        self.emb_orbs, self.core_orbs, self.Nbath, self.Nelec_in_emb = (
            self.bath_construction(self.loc_1RDM_R0, self._impOrbs)
        )

        self.solver.name = "HF"
        nelec_cell = self.kernel(chempot=0.0)

        diff = abs(self.e_tot - self.kmf.e_tot)
        tprint.print_msg("   E(RHF)        : %12.8f" % (self.kmf.e_tot))
        tprint.print_msg("   E(RHF-DMET)   : %12.8f" % (self.e_tot))
        tprint.print_msg("   |RHF - RHF(DMET)|          : %12.8f" % (diff))
        tprint.print_msg(" No. of electrons per cell : %12.8f" % (nelec_cell))
        if diff < error:
            tprint.print_msg("   HF-in-HF embedding is exact: True")
        else:
            raise Exception("WARNING: HF-in-HF embedding is not exact")

    def one_shot(self, umat=0.0, proj_DMET=False):
        """
        Do one-shot DMET, only the chemical potential is optimized
        this function takes umat or loc_1RDM_R0 (p-DMET algorthm)
        """

        self._print_solver_header()

        self._cycle = 1
        if not proj_DMET:
            self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = (
                self.local.make_loc_1RDM(
                    umat,
                    self.mask4Gamma,
                    OEH_type=self.emb.OEH_type,
                    dft_HF=self.emb.dft_HF,
                )
            )

        self.emb_orbs, self.core_orbs, self.Nbath, self.Nelec_in_emb = (
            self.bath_construction(self.loc_1RDM_R0, self._impOrbs)
        )

        # Solve embedding
        if self._is_gamma:
            _ = self.kernel(chempot=0.0)
        else:
            self.chempot = optimize.newton(self.nelec_cost_func, self.chempot)
            tprint.print_msg(
                "   No. of electrons per cell : %12.8f" % (self.nelec_per_cell)
            )

        self._print_energies()

    def _print_solver_header(self):
        """Solver header"""
        tprint.print_msg("-- One-shot DMET ... starting at %s" % (tunix.current_time()))
        extra = f" | 2S = {self.solver.twoS}"
        if self.solver.name == "HF":
            qc_label = "RHF" if self.solver.twoS == 0 and not self._is_ROHF else "ROHF"
        else:
            qc_label = self.solver.name

        if self.solver.name not in ["HF", "RCCSD"]:
            extra += f" | Nroots: {self.solver.nroots}"

        # Print Bath Solver line
        tprint.print_msg(
            f"   Bath type: {self.bathtype} | QC Solver: {qc_label}{extra}"
        )

        required = {"CAS", "DMRG"}
        if any(req in self.solver.name for req in required):
            if self.solver.cas is not None:
                tprint.print_msg("   Active space     :", self.solver.cas)
                tprint.print_msg("   Active space MOs :", self.solver.molist)

            if self.solver.name.startswith("SS-"):
                tprint.print_msg(
                    "   State-specific CASSCF using state id :",
                    self.solver.state_specific_,
                )

            if self.solver.name.startswith("SA-"):
                if self.solver.state_average_ is not None:
                    tprint.print_msg(
                        "   State-average CASSCF with weight :",
                        self.solver.state_average_,
                    )
                elif self.solver.state_average_mix_ is not None:
                    tprint.print_msg("   State-average CASSCF with mixed Solvers :")
                    for i, mix in enumerate(self.solver.state_average_mix_):
                        tprint.print_msg(
                            " Solver %d: spin %d, roots %d, weight %s"
                            % (i, mix.spin, mix.roots, str(mix.weights))
                        )
            if "CASPDFT" in self.solver.name:
                tprint.print_msg("   On-top functional:", self.solver.otxc or "tPBE")

    def _print_energies(self):
        """Generic printer function"""

        name = self.solver.name
        weights = getattr(self.solver, "state_average_", None)
        is_pdft = "CASPDFT" in name

        tprint.print_msg("-" * 60)
        tprint.print_msg("Results")
        tprint.print_msg("-" * 60)

        if is_pdft:
            # Always report both CASSCF and PDFT
            if isinstance(self.e_cas, (list, np.ndarray)):
                # SA-CASSCF: per-state CASSCF energies
                for i, e in enumerate(self.e_cas):
                    w = self.solver.state_average_[i]
                    tprint.print_msg(
                        "      State %d (w=%5.3f): E(pDMET CASSCF) = %12.8f Eh"
                        % (i, w, e)
                    )
                for i, e in enumerate(self.e_tot):
                    w = self.solver.state_average_[i]
                    tprint.print_msg(
                        "      State %d (w=%5.3f): E(pDMET-PDFT) = %12.8f Eh"
                        % (i, w, e)
                    )
            else:
                tprint.print_msg("   E(pDMET CASSCF)    : %12.8f Eh" % self.e_cas)
                tprint.print_msg("   E(pDMET-PDFT) : %12.8f Eh" % self.e_tot)

        if isinstance(self.e_tot, (list, np.ndarray)) and not is_pdft:
            # SA-CASSCF or multi-root
            tprint.print_msg("   Energy per cell  : %12.8f Eh" % self.e_tot[0])
            if self.solver.state_average_ is not None:
                weights = self.solver.state_average_
            elif self.solver.state_average_mix_ is not None:
                weights = []
                for solver in self.solver.state_average_mix_:
                    weights += solver.weights
            else:
                weights = None
            for i, e in enumerate(self.e_tot):
                if weights is not None:
                    tprint.print_msg(
                        "      State %d (w=%5.3f): E(pDMET %s) = %12.8f Eh"
                        % (i, weights[i], name, e)
                    )
                else:
                    tprint.print_msg(
                        "      State %d: E(pDMET %s) = %12.8f Eh" % (i, name, e)
                    )

        else:
            # Standard scalar energy
            tprint.print_msg(
                "   Energy per cell E(pDMET %s) : %12.8f Eh" % (name, self.e_tot)
            )

        # NEVPT2

        def ensure_flat(x):
            """Helper function to flatten states for state_average_mix_"""
            if x and isinstance(x[0], list):
                return [item for sublist in x for item in sublist]
            return x

        if self.solver.nevpt2_roots is not None:
            nevpt2_states = ensure_flat(self.solver.nevpt2_roots)
            tprint.print_msg("   NEVPT2 energies for the selected states:")
            for i, e_nevpt2 in enumerate(self.e_nevpt2_tot):
                tprint.print_msg(
                    "      State %d: E(pDMET CASCI) = %12.8f Eh  E(pDMET NEVPT2) = %12.8f Eh   <S^2> = %8.6f"
                    % (
                        nevpt2_states[i],
                        self.e_casci_tot[i],
                        e_nevpt2,
                        self.ss_CASCI[i],
                    )
                )

        tprint.print_msg("-- One-shot DMET ... finished at %s" % (tunix.current_time()))

    def self_consistent(self, get_band=False, interpolate_band=None):
        """
        Do self-consistent pDMET
        """
        tprint.print_msg("-" * 60)
        tprint.print_msg("- SELF-CONSISTENT DMET CALCULATION ... STARTING -")
        tprint.print_msg("  Convergence criteria")
        tprint.print_msg("    Threshold :", self.scf.threshold)
        tprint.print_msg("  Fitting 1-RDM of :", self.scf.CF_type)

        if self.emb.dft_CF:
            tprint.print_msg("  DF-like cost function:", self.emb.xc)
        if self.scf.damping != 1.0:
            tprint.print_msg("  Damping factor   :", self.scf.damping)
        if self.scf.use_DIIS:
            tprint.print_msg(
                "  DIIS start at %dth cycle and using %d previous umats"
                % (self.scf.DIIS_start, self.scf.DIIS_nvector)
            )

        # ---- SELF-CONSISTENT PROCEDURE ----
        OEH_kpts, rdm1_kpts, rdm1_R0 = self.local.make_loc_1RDM(
            self.umat,
            self.mask4Gamma,
            OEH_type=self.emb.OEH_type,
            dft_HF=self.emb.dft_HF,
        )
        for cycle in range(self.scf.maxcycle):
            tprint.print_msg("- CYCLE %d:" % (cycle + 1))
            umat_old = self.umat
            rdm1_R0_old = rdm1_R0
            energy_old = self.e_tot

            # Do one-shot with each uvec
            self.one_shot(umat=self.umat)
            tprint.print_msg("   + Chemical potential        : %12.8f" % (self.chempot))

            # Optimize uvec to minimize the cost function
            if self.emb.dft_CF:
                result = optimize.minimize(
                    self.CF,
                    self.uvec,
                    method="L-BFGS-B",
                    jac=None,
                    options={"disp": False, "gtol": 1e-4, "eps": 1e-8},
                    bounds=self.bounds,
                )
            else:
                # result = optimize.minimize(self.CF, self.uvec, method=self.SC_method, jac=self.CF_grad, options={'disp': False, 'gtol': 1e-12})
                result = optimize.minimize(
                    self.CF,
                    self.uvec,
                    method=self.scf.method,
                    options={"disp": False, "gtol": 1e-6},
                    tol=1e-4,
                )

            if not result.success:
                tprint.print_msg("     WARNING: Correlation potential is not converged")

            uvec = result.x
            self.umat = self.uvec2umat(uvec)

            # Construct new global 1RDM in k-space
            global_corr_1RDM = self.local.get_1RDM_Rs(self.loc_corr_1RDM_R0)
            global_corr_1RDM = 0.5 * (global_corr_1RDM.T.conj() + global_corr_1RDM)
            if not self._is_gamma:
                loc_1RDM_R0 = global_corr_1RDM[:, : self.Nimp].reshape(
                    self.Nkpts, self.Nimp, self.Nimp
                )
            else:
                loc_1RDM_R0 = global_corr_1RDM
            rdm1_R0 = loc_1RDM_R0

            # Remove arbitrary chemical potential shifts
            if not self.emb.dft_CF:
                self.umat = self.umat - np.eye(self.umat.shape[0]) * np.average(
                    np.diag(self.umat)
                )

            if self.verbose > 0:
                tprint.print_msg("   + Correlation potential vector    : ", uvec)

            umat_diff = umat_old - self.umat
            rdm_diff = rdm1_R0_old - rdm1_R0
            energy_diff = self.e_tot - energy_old
            if self.solver.state_average_ is not None:
                energy_diff = self.e_tot - energy_old
                energy_diff = np.sum(
                    energy_diff * np.asarray(self.solver.state_average_)
                )
            norm_u = np.linalg.norm(umat_diff)
            norm_rdm = np.linalg.norm(rdm_diff)

            tprint.print_msg("   + Cost function             : %20.15f" % (result.fun))
            tprint.print_msg("   + 2-norm of umat difference : %20.15f" % (norm_u))
            tprint.print_msg("   + 2-norm of rdm1 difference : %20.15f" % (norm_rdm))
            tprint.print_msg("   + Energy difference         : %20.15f" % (energy_diff))

            # Export band structure at every cycle:
            if get_band:
                band = self.get_bands()
                pywannier90.save_kmf(
                    band, str(self.solver.name) + "_band_cyc_" + str(cycle + 1)
                )

            # DEBUG
            if interpolate_band is not None:
                frac_kpts = interpolate_band
                bands = self.interpolate_band(frac_kpts)  # noqa: F841

            # Check convergence of 1-RDM
            if self.emb.dft_CF:
                if norm_rdm <= self.scf.threshold:
                    break
            elif norm_u <= self.scf.threshold:
                break

            if self.scf.use_DIIS:
                self.umat = self._diis.update(cycle, self.umat, umat_diff)

            if self.scf.damping != 1.0:
                self.umat = (
                    1.0 - self.scf.damping
                ) * umat_old + self.scf.damping * self.umat

            self.uvec = self.umat2uvec(self.umat)
            tprint.print_msg()

        tprint.print_msg("- SELF-CONSISTENT DMET CALCULATION ... DONE -")
        tprint.print_msg("-" * 60)

    def projected_DMET(self, get_band=False):
        """
        Do projected DMET
        """

        tprint.print_msg("-" * 60)
        tprint.print_msg("- p-DMET CALCULATION ... STARTING -")
        tprint.print_msg("  Convergence criteria")
        tprint.print_msg("    Threshold :", self.scf.threshold)
        tprint.print_msg("  Fitting 1-RDM of :", self.scf.CF_type)

        if self.scf.damping != 1.0:
            tprint.print_msg("  Damping factor   :", self.scf.damping)
        if self.scf.use_DIIS:
            tprint.print_msg(
                "  DIIS start at %dth cycle and using %d previous umats"
                % (self.scf.DIIS_start, self.scf.DIIS_nvector)
            )

        # ------------------------------------#
        # ---- SELF-CONSISTENT PROCEDURE ----#
        # ------------------------------------#
        self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = (
            self.local.make_loc_1RDM(
                0.0, self.mask4Gamma, OEH_type=self.emb.OEH_type, dft_HF=self.emb.dft_HF
            )
        )
        global_corr_1RDM = self.local.k_to_R(self.loc_1RDM_kpts)
        for cycle in range(self.scf.maxcycle):
            tprint.print_msg("- CYCLE %d:" % (cycle + 1))
            global_corr_1RDM_old = global_corr_1RDM
            self.one_shot(proj_DMET=True)

            if not self._is_gamma:
                tprint.print_msg(
                    "   + Chemical potential        : %12.8f" % (self.chempot)
                )

            # Construct new global 1RDM in k-space
            global_corr_1RDM = self.local.get_1RDM_Rs(self.loc_corr_1RDM_R0)
            global_corr_1RDM = 0.5 * (global_corr_1RDM.T.conj() + global_corr_1RDM)
            global_corr_1RDM_residual = global_corr_1RDM_old - global_corr_1RDM
            norm_1RDM = np.linalg.norm(global_corr_1RDM_residual) / self.kpts.shape[0]
            tprint.print_msg("   + 2-norm of rdm1 difference : %20.15f" % (norm_1RDM))

            if get_band is True:
                band = self.get_bands()
                pywannier90.save_kmf(
                    band, str(self.solver.name) + "_band_cyc_" + str(cycle + 1)
                )

            # Check convergence of 1-RDM
            if norm_1RDM <= self.scf.threshold:
                break

            if self.scf.use_DIIS is True:
                global_corr_1RDM = self._diis.update(
                    cycle, global_corr_1RDM, global_corr_1RDM_residual
                )

            if self.scf.damping != 1.0:
                global_corr_1RDM = (
                    self.scf.damping * global_corr_1RDM
                    + (1 - self.scf.damping) * global_corr_1RDM_old
                )

            # Construct new mean-field 1-RDM from the correlated one
            eigenvals, eigenvecs = np.linalg.eigh(global_corr_1RDM)
            idx = (-eigenvals).argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            num_pairs = self.Nelec_total // 2
            global_mf_1RDM = 2 * eigenvecs[:, :num_pairs].dot(
                eigenvecs[:, :num_pairs].T
            )
            if self._is_gamma:
                self.loc_1RDM_R0 = (
                    global_mf_1RDM.reshape(1, self.Norbs, self.Norbs)
                    + self.loc_core_1RDM
                )
            else:
                self.loc_1RDM_R0 = global_mf_1RDM[:, : self.Nimp].reshape(
                    self.Nkpts, self.Nimp, self.Nimp
                )

            self.loc_1RDM_kpts = self.local.R0_to_k(self.loc_1RDM_R0)
            tprint.print_msg()

        tprint.print_msg("- p-DMET CALCULATION ... DONE -")
        tprint.print_msg("-" * 60)

    def nelec_cost_func(self, chempot):
        """
        The different in the correct number of electrons (provided) and the calculated one
        """

        nelec_per_cell_from_embedding = self.kernel(chempot)
        self._is_new_bath = False
        tprint.print_msg(
            "     Cycle %2d. Chem potential: %12.8f | Elec/cell = %12.8f | <S^2> = %12.8f"
            % (self._cycle, chempot, nelec_per_cell_from_embedding, self.qcsolver.SS)
        )
        self._cycle += 1
        return nelec_per_cell_from_embedding - self.Nelec_per_cell

    def cost_func(self, uvec):
        """
        Cost function: \mathbf{CF}(u) = \mathbf{\Sigma}_{rs} (D^{mf}_{rs}(u) - D^{corr}_{rs})^2
        where D^{mf} and D^{corr} are the mean-field and correlated 1-RDM, respectively.
        and D^{mf} = \mathbf{FT}(D^{mf}(k))
        """
        rdm_diff = self.get_rdm_diff(uvec)
        cost = np.power(rdm_diff, 2).sum()
        return cost

    def cost_func_grad(self, uvec):
        """
        Analytical derivative of the cost function,
        deriv(CF(u)) = Sum^x [Sum_{rs} (2 * rdm_diff^x_{rs}(u) * deriv(rdm_diff^x_{rs}(u))]
        ref: J. Chem. Theory Comput. 2016, 12, 2706−2719
        """
        rdm_diff = self.get_rdm_diff(uvec)
        rdm_diff_grad = self.rdm_diff_grad(uvec)
        CF_grad = np.zeros(self.Nterms)

        for u in range(self.Nterms):
            CF_grad[u] = np.sum(2 * rdm_diff * rdm_diff_grad[u])
        return CF_grad

    def get_rdm_diff(self, uvec):
        """
        Calculating the different between mf-1RDM (transformed in schmidt basis) and correlated-1RDM for the unit cell
        Args:
            uvec            : the correlation potential vector
        Return:
            error            : an array of errors for the unit cell.
        """

        loc_OEH_kpts, loc_1RDM_kpts, loc_1RDM_R0 = self.local.make_loc_1RDM(
            self.uvec2umat(uvec),
            self.mask4Gamma,
            OEH_type=self.emb.OEH_type,
            dft_HF=self.emb.dft_HF,
        )
        if self.scf.CF_type in ["F", "diagF"]:
            mf_1RDM = self.local.loc_kpts_to_emb(
                loc_1RDM_kpts, self.emb_orbs[:, :, : self.Nimp]
            )
            corr_1RDM = self.emb_corr_1RDM[: self.Nimp, : self.Nimp]
        elif self.scf.CF_type in ["FB", "diagFB"]:
            mf_1RDM = self.local.loc_kpts_to_emb(loc_1RDM_kpts, self.emb_orbs)
            corr_1RDM = self.emb_corr_1RDM

        error = mf_1RDM - corr_1RDM
        if self.scf.CF_type in ["diagF", "diagFB"]:
            error = np.diag(error)

        return error

    def rdm_diff_grad(self, uvec):
        """
        Compute the rdm_diff gradient
        Args:
            uvec            : the correlation potential vector
        Return:
            the_gradient    : a list with the size of the number of u values in uvec
                              Each element of this list is an array of derivative corresponding to each rs.

        """

        RDM_deriv_kpts = self.construct_1RDM_response_kpts(uvec)
        the_gradient = []
        for u in range(self.Nterms):
            RDM_deriv_R0 = self.local.k_to_R0(  # noqa: F841
                RDM_deriv_kpts[:, u, :, :]
            )  # Transform RDM_deriv from k-space to the reference cell
            if self.scf.CF_type in ["F", "diagF"]:
                emb_error_deriv = self.local.loc_kpts_to_emb(
                    RDM_deriv_kpts[:, u, :, :], self.emb_orbs[:, :, : self.Nimp]
                )
            elif self.scf.CF_type in ["FB", "diagFB"]:
                emb_error_deriv = self.local.loc_kpts_to_emb(
                    RDM_deriv_kpts[:, u, :, :], self.emb_orbs
                )
            if self.scf.CF_type in ["diagF", "diagFB"]:
                emb_error_deriv = np.diag(emb_error_deriv)
            the_gradient.append(emb_error_deriv)

        return np.asarray(the_gradient)

    def glob_cost_func(self, uvec):
        """TODO write it"""
        rdm_diff = self.get_glob_rdm_diff(uvec)
        cost = np.power(rdm_diff, 2).sum()
        return cost

    def glob_cost_func_grad(self, uvec):
        """TODO"""
        rdm_diff = self.get_glob_rdm_diff(uvec)
        rdm_diff_grad = self.glob_rdm_diff_grad(uvec)
        CF_grad = np.zeros(self.Nterms)

        for u in range(self.Nterms):
            CF_grad[u] = np.sum(2 * rdm_diff * rdm_diff_grad[u])
        return CF_grad

    def get_glob_rdm_diff(self, uvec):
        """
        Calculating the different between mf-1RDM (transformed in schmidt basis) and correlated-1RDM for the unit cell
        Args:
            uvec            : the correlation potential vector
        Return:
            error            : an array of errors for the unit cell.
        """
        loc_OEH_kpts, loc_1RDM_kpts, loc_1RDM_R0 = self.local.make_loc_1RDM(
            self.uvec2umat(uvec),
            self.mask4Gamma,
            OEH_type=self.emb.OEH_type,
            dft_HF=self.emb.dft_HF,
        )
        error = loc_1RDM_R0 - self.loc_corr_1RDM_R0
        return error

    def glob_rdm_diff_grad(self, uvec):
        """
        Compute the rdm_diff gradient
        Args:
            uvec            : the correlation potential vector
        Return:
            the_gradient    : a list with the size of the number of u values in uvec
                              Each element of this list is an array of derivative corresponding to each rs.

        """

        RDM_deriv_kpts = self.construct_1RDM_response_kpts(uvec)
        the_gradient = []

        for u in range(self.Nterms):
            RDM_deriv_R0 = self.local.k_to_R0(
                RDM_deriv_kpts[:, u, :, :]
            )  # Transform RDM_deriv from k-space to the reference cell
            the_gradient.append(RDM_deriv_R0)

        return np.asarray(the_gradient)

    def alt_cost_func(self, uvec):
        """
        TODO: DEBUGGING
        """

        umat = self.uvec2umat(uvec)

        loc_OEH_kpts, loc_1RDM_kpts, loc_1RDM_R0 = self.local.make_loc_1RDM(
            umat, self.mask4Gamma, OEH_type=self.emb.OEH_type, dft_HF=self.emb.dft_HF
        )
        if self.emb.OEH_type == "FOCK":
            OEH = self.local.loc_actFOCK_kpts  # +umat
            OEH = self.local.k_to_R(OEH)
            e_fun = np.trace(OEH.dot(loc_1RDM_R0))
        else:
            tprint.print_msg("Other type of 1e electron is not supported")

        rdm_diff = self.glob_rdm_diff(uvec)[0]
        e_cstr = np.sum(umat * rdm_diff)

        return -e_fun - e_cstr

    def alt_cost_func_grad(self, uvec):
        """
        TODO: DEBUGGING
        """
        rdm_diff = self.glob_rdm_diff(uvec)[0]
        return -rdm_diff

    ######################################## USEFUL FUNCTION for pDMET class ########################################

    def make_irred_kpts(self, kpts=None):
        """
        Make k-dependent uvec considering kmesh symmetry
        Attributes:
            kpts_irred      : a list of irreducible k-point
            sym_id          : a list of symmetry label. k and -k should have the same label
            sym_map         : used to map the uvec (irreducible k-points) to umat (full k-points)
        """
        if kpts is None:
            kpts = self.kpts
        kpts = np.asarray(kpts)

        sym_id = np.asarray(range(self.Nkpts))
        kpts_irred, sym_counts = np.unique(sym_id, return_counts=True)
        sym_map = [
            np.where(kpts_irred == sym_id[kpt])[0][0] for kpt in range(self.Nkpts)
        ]
        nkpts_irred = kpts_irred.size
        num_u = nkpts_irred * self.Nterms
        uvec = np.zeros(num_u, dtype=np.float64)

        return kpts_irred, sym_counts, sym_map, uvec

    def make_mask(self, is_gamma=False):
        """
        Make a mask used to convert uvec to umat and vice versa
        """
        if is_gamma:
            impCluster = np.asarray(self._impOrbs)
            if self.scf.CF_type in ["F", "FB"]:
                mask = np.matrix(impCluster).T.dot(np.matrix(impCluster)) == 1
                mask[np.tril_indices(self.Norbs, -1)] = False
            else:
                mask = np.zeros([self.Norbs, self.Norbs], dtype=bool)
                mask[impCluster == 1, impCluster == 1] = True
        else:
            mask = np.zeros([self.Nimp, self.Nimp], dtype=bool)
            if self.scf.CF_type in ["F", "FB"]:
                mask[np.triu_indices(self.Nimp)] = True
            else:
                np.fill_diagonal(mask, True)
        return mask

    def uvec2umat(self, uvec):
        """
        Convert uvec to the umat which is will be added up to the local one-electron Hamiltonian at each k-point
        """
        if self.emb.dft_CF:
            the_umat = uvec
        elif self._is_gamma:
            the_umat = np.zeros([self.Norbs, self.Norbs], dtype=np.float64)
            the_umat[self.mask] = uvec
            the_umat = the_umat.T
            the_umat[self.mask] = uvec
        else:
            the_umat = np.zeros([self.Nimp, self.Nimp], dtype=np.float64)
            the_umat[self.mask] = uvec
            the_umat = the_umat.T
            the_umat[self.mask] = uvec

        return np.asarray(the_umat)

    def umat2uvec(self, umat):
        """
        Convert umat to the uvec
        """
        if self.emb.dft_CF is True:
            return umat
        else:
            return umat[self.mask]

    def make_H1(self, is_gamma=False, impCluster=None):
        """
        The H1 is the correlation potential operator, this function taking advantage of sparsity of the u matrix in calculating gradient of 1-RDM at each k-point
        Return:
            H1start:
            H1row:
            H1col:
        """
        if is_gamma is True:
            assert impCluster is not None, (
                "In Gamma-point sampling, you need a list to define impurity orbitals"
            )

        theH1 = []
        if is_gamma is True:
            imp_indices = np.where(np.asarray(impCluster) == 1)[0]
            if self.scf.CF_type in ["diagF", "diagFB"]:
                for idx in imp_indices:
                    H1 = np.zeros([self.Norbs, self.Norbs])
                    H1[idx, idx] = 1
                    theH1.append(H1)
            else:
                for i, row in enumerate(imp_indices):
                    for col in imp_indices[i:]:
                        H1 = np.zeros([self.Norbs, self.Norbs])
                        H1[row, col] = 1
                        H1[col, row] = 1
                        theH1.append(H1)
        else:
            if self.scf.CF_type in ["diagF", "diagFB"]:
                for row in range(self.Nimp):
                    H1 = np.zeros([self.Nimp, self.Nimp])
                    H1[row, row] = 1
                    theH1.append(H1)
            else:
                for row in range(self.Nimp):  # Fitting the whole umat
                    for col in range(row, self.Nimp):
                        H1 = np.zeros([self.Nimp, self.Nimp])
                        H1[row, col] = 1
                        H1[col, row] = 1
                        theH1.append(H1)

        # Convert the sparse H1 to one dimension H1start, H1row, H1col arrays used in libdmet.rhf_response()
        H1start = []
        H1row = []
        H1col = []
        H1start.append(0)
        totalsize = 0
        for count in range(len(theH1)):
            rowco, colco = np.where(theH1[count] == 1)
            totalsize += len(rowco)
            H1start.append(totalsize)
            for count2 in range(len(rowco)):
                H1row.append(rowco[count2])
                H1col.append(colco[count2])
        H1start = np.array(H1start)
        H1row = np.array(H1row)
        H1col = np.array(H1col)

        return theH1, H1start, H1row, H1col

    def construct_1RDM_response_kpts(self, uvec):
        """
        Calculate the derivative of 1RDM
        TODO: Currently the number of electron is the same at every k-point. This is not the case for
        metallic sytem. So need to consider this later
        """

        rdm_deriv_kpts = []
        loc_actFOCK_kpts = self.local.loc_actFOCK_kpts + self.uvec2umat(uvec)
        Norb = loc_actFOCK_kpts.shape[-1]
        for kpt in range(self.Nkpts):
            # rdm_deriv = libdmet.rhf_response_c(Norb, self.Nterms, self.numPairs, self.H1start, self.H1row, self.H1col, loc_actFOCK_kpts[kpt])
            rdm_deriv = libdmet.rhf_response(
                Norb,
                self.Nterms,
                self.numPairs,
                self.H1start,
                self.H1row,
                self.H1col,
                loc_actFOCK_kpts[kpt].real,
            )
            rdm_deriv = np.complex128(rdm_deriv)
            rdm_deriv_kpts.append(rdm_deriv)

        return np.asarray(rdm_deriv_kpts)

    def construct_global_1RDM(self):
        """Construct the global 1RDM in the R-space"""

        imp_1RDM = lib.einsum(
            "Rim,mn,jn->Rij", self.emb_orbs, self.emb_corr_1RDM, self.emb_orbs[0]
        )
        RDM1_Rs = self.local.get_1RDM_Rs(imp_1RDM)
        RDM1_Rs = 0.5 * (RDM1_Rs.T + RDM1_Rs)  # make sure the global DM is hermitian

        return RDM1_Rs

    ######################################## POST pDMET ANALYSIS ########################################
    def get_bands(
        self, cell=None, dm_kpts=None, kpts=None, cost_func="glob", method="BFGS"
    ):
        """Embedding 1RDM is used to construct the global 1RDM.
        The 'closest' mean-field 1RDM to the global 1RDM is found by minizing the norm(D_global - D_mf)
        """
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kmf.kpts

        # Compute the total DM in the local basis
        if cost_func == "FB":
            CF = self.cost_func
            CF_grad = self.cost_func_grad
            self.scf.CF_type = "FB"
        elif cost_func == "F":
            CF = self.cost_func
            CF_grad = self.cost_func_grad
            self.scf.CF_type = "F"
        else:
            CF = self.glob_cost_func
            CF_grad = self.glob_cost_func_grad  # noqa: F841

        if self.emb.dft_CF and self.emb.xc == "PBE0":
            result = optimize.minimize(
                self.CF,
                self.uvec,
                method="L-BFGS-B",
                jac=None,
                options={"disp": False, "gtol": 1e-6},
                bounds=self.bounds,
            )
        else:
            result = optimize.minimize(
                CF,
                self.uvec,
                method=method,
                jac=None,
                options={"disp": False, "gtol": 1e-12},
            )

        uvec = result.x
        error = np.linalg.norm(self.get_glob_rdm_diff(uvec))
        if result.success is False:
            tprint.print_msg("Band structure error: %12.8f" % (error))
            tprint.print_msg(" WARNING: Correlation potential is not converged")
        else:
            tprint.print_msg("Band structure error: %12.8f" % (error))

        if self.emb.dft_CF:
            eigvals, eigvecs = self.local.make_loc_1RDM_kpts(
                self.uvec2umat(uvec),
                self.mask4Gamma,
                OEH_type=self.emb.xc,
                get_band=True,
                dft_HF=self.emb.dft_HF,
            )
        else:
            eigvals, eigvecs = self.local.make_loc_1RDM_kpts(
                self.uvec2umat(uvec),
                self.mask4Gamma,
                OEH_type="FOCK",
                get_band=True,
                dft_HF=self.emb.dft_HF,
            )

        dmet_orbs = lib.einsum(
            "kpq,kqr->kpr", self.local.ao2lo, eigvecs
        )  # embedding orbs are spaned by AO instead of MLWFs here
        mo_coeff_kpts = []
        mo_energy_kpts = []
        for kpt in range(self.Nkpts):
            mo_coeff = self.kmf.mo_coeff_kpts[kpt].copy()
            mo_coeff[:, self.w90.band_included_list] = dmet_orbs[kpt]
            mo_energy = self.kmf.mo_energy_kpts[kpt].copy()
            mo_energy[self.w90.band_included_list] = eigvals[kpt]
            mo_coeff_kpts.append(mo_coeff)
            mo_energy_kpts.append(mo_energy)

        ovlp = self.kmf.get_ovlp()

        class fake_kmf:
            def __init__(self):
                self.kpts = kpts
                self.mo_energy_kpts = mo_energy_kpts
                self.mo_coeff_kpts = mo_coeff_kpts
                self.get_ovlp = lambda *arg: ovlp

        kmf = fake_kmf()

        return kmf

    def interpolate_band(
        self,
        frac_kpts,
        use_ws_distance=True,
        ws_search_size=[2, 2, 2],
        ws_distance_tol=1e-6,
    ):
        """Interpolate the band structure using the Slater-Koster scheme
        Return:
            eigenvalues and eigenvectors at the desired kpts
        """
        OEH_kpts, eigvals, eigvecs = self.local.make_loc_1RDM_kpts(
            self.uvec2umat(self.uvec),
            self.mask4Gamma,
            OEH_type=self.emb.xc,
            get_ham=True,
            dft_HF=self.emb.dft_HF,
        )
        eigvals, eigvecs = self.w90.interpolate_band(
            frac_kpts, OEH_kpts, use_ws_distance, ws_search_size, ws_distance_tol
        )
        return (eigvals, eigvecs)

    def plot(self, orb="emb", grid=[50, 50, 50], path="./", fmt="xsf"):
        """Plot orbitals for CAS solvers
        orb = 'emb', 'mf', 'mc', 'mc_nat'
        """

        emb_orbs = self.emb_orbs[0]
        if orb == "wfs":
            rotate_mat = None
        if orb == "emb":
            rotate_mat = emb_orbs
        elif orb == "mf":
            mo = self.qcsolver.mf.mo_coeff
            rotate_mat = emb_orbs.dot(mo)
        elif orb == "mc":
            mo = self.qcsolver.mo
            rotate_mat = emb_orbs.dot(mo)
        elif orb == "nat":
            mo = self.qcsolver.mo_nat
            rotate_mat = emb_orbs.dot(mo)
        elif orb == "nto":
            assert self.solver.nevpt2_roots is not None, (
                "NEVPT2 must be called to calculate the NTOs"
            )
            pass

        tplot.plot_wf(self.w90, rotate_mat, path + "/" + orb, self.kmesh, grid, fmt=fmt)

    def get_trans_dipole(self):
        """Calculate transition dipole"""
        assert self.solver.nevpt2_roots is not None, (
            "NEVPT2 must be called to calculate the NTOs"
        )
        charges = self.cell.atom_charges()
        coords = self.cell.atom_coords()
        nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
        self.cell.set_common_orig_(nuc_charge_center)
        dip_ints = self.cell.intor("cint1e_r_sph", comp=3)
        ao2eo = self.local.get_ao2eo(self.emb_orbs)[0]

        def makedip(ci_id):
            t_dm1_emb = self.t_dm1s[ci_id]
            # transform density matrix from MO to AO representation
            t_dm1_ao = ao2eo @ t_dm1_emb @ ao2eo.T.conj()
            return np.einsum("xij,ji->x", dip_ints, t_dm1_ao).real

        for i in range(len(self.e_nept2_tot)):
            dipole = makedip(i)
            norm = np.linalg.norm(dipole)
            print(
                "Transition dipole between |0> and |{0:d}>: {1:3.5f} {2:3.5f} {3:3.5f} | Norm: {4:3.5f}".format(
                    i, dipole[0], dipole[1], dipole[2], norm
                )
            )
