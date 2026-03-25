from dataclasses import dataclass
from pdmet.qcsolvers.casbase import BaseCASSolver
from typing import Optional
from pyscf import mcscf, lib
import numpy as np


@dataclass
class PDFTContext:
    """
    Everything pDMET.kernel() pre-computes via local
    that the solver needs for pDME-PDFT energy evaluation.

    local stays in pDMET — but its products are passed here.
    """

    cell: object  # dft.RKS(cell) + V_NN
    kmf: object  # T, E_J, E_x from periodic integrals
    local: object  #
    emb_orbs: np.ndarray  # embedding orbitals
    core_orbs: np.ndarray  # core orbitals
    emb_core_orbs: np.ndarray  # combined emb+core orbitals
    OEH_type: str  # RHF or ROHF
    mask4Gamma: np.ndarray  # Check
    otxc: str = "tPBE"  # We should do an enum to we define the ones considered


class CASPDFTSolver(BaseCASSolver):
    """CASPDFT - FCI within a active space"""

    def __init__(self, settings, is_KROHF=False):
        super().__init__(settings, is_KROHF)
        self._ctx: Optional[PDFTContext] = None
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
        pdft_context: Optional[PDFTContext] = None,
    ):
        self._ctx = pdft_context
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

        # Run DFT grid RKS impossed
        self.otfnal = self.build_otfnal(self._ctx.cell, self._ctx.otxc)

        if self.settings.nroots == 1 or state_specific_ is not None:
            e_cell, e_pdft, RDM1 = self._single_root_casscf(fcivec, cas_norb, e_tot)
        elif state_average_ is not None:
            e_cell, e_pdft, RDM1 = self._state_average(
                fcivec, cas_norb, e_tot, state_average_
            )

        return e_cell, (e_tot, e_pdft), RDM1

    def build_otfnal(self, cell, otxc):
        """Helper to build the on-top functional object for CASPDFT."""
        from pyscf import dft
        from pyscf.mcpdft.otfnal import transfnal, ftransfnal

        ks = dft.RKS(cell).density_fit()

        if otxc.upper().startswith("T"):
            ks.xc = otxc[1:]
            otfnal = transfnal(ks)
        elif otxc.upper().startswith("FT"):
            ks.xc = otxc[2:]
            otfnal = ftransfnal(ks)
        else:
            raise ValueError(f"Unknown otxc: {otxc}")

        grids = dft.gen_grid.Grids(cell)
        grids.level = 6  # Default is 3, (Refactor to control this in the Settings)

        otfnal.grids = grids
        otfnal.verbose = 3

        return otfnal

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
        casdm1_mo = self.mc.fcisolver.make_rdm1(fcivec, cas_norb, self.mc.nelecas)
        casdm2_mo = self.mc.fcisolver.make_rdm2(fcivec, cas_norb, self.mc.nelecas)
        dm1sa_mo, dm1sb_mo = self.mc.fcisolver.make_rdm1s(
            fcivec, cas_norb, self.mc.nelecas
        )
        RDM1 = self._cas_rdm1_to_local_from_dm(casdm1_mo, self.mc, cas_norb)
        e_cell = self.kmf_ecore + self._impurity_energy_from_cas_df(
            self.mc, cas_norb, RDM1, casdm2_mo
        )
        # Run DMET-PDFT correction
        e_pdft, _ = self._get_dmet_pdft(
            cas_norb, casdm1_mo, casdm2_mo, [dm1sa_mo, dm1sb_mo]
        )
        state_id = self.settings.state_specific_ or 0
        print(f"  State {state_id}: E(CASSCF)={e_tot:12.8f}  <S^2>={self.SS:8.6f}")

        return e_cell, e_pdft, RDM1

    def _state_average(self, fcivec, cas_norb, e_tot, weights):
        RDM1s, e_cells, e_pdfts = [], [], []
        ss = self.mc.fcisolver.states_spin_square(fcivec, cas_norb, self.mc.nelecas)[0]
        rdm1s_cas, rdm2s_cas = self.mc.fcisolver.states_make_rdm12(
            fcivec, cas_norb, self.mc.nelecas
        )
        dm1sa_mos, dm1sb_mos = self.mc.fcisolver.states_make_rdm1s(
            fcivec, cas_norb, self.mc.nelecas
        )
        for i, civec in enumerate(fcivec):
            rdm1 = self._cas_rdm1_to_local_from_dm(rdm1s_cas[i], self.mc, cas_norb)
            e_imp = self.kmf_ecore + self._impurity_energy_from_cas_df(
                self.mc, cas_norb, rdm1, rdm2s_cas[i]
            )
            e_pdft, _ = self._get_dmet_pdft(
                cas_norb, rdm1s_cas[i], rdm2s_cas[i], [dm1sa_mos[i], dm1sb_mos[i]]
            )
            print(
                f"  State {i} ({weights[i]:.3f}): E(CASSCF)={e_tot[i]:12.8f}  "
                f"E(imp)={e_imp:12.8f}  <S^2>={ss[i]:8.6f}"
            )
            RDM1s.append(rdm1)
            e_cells.append(e_imp)
            e_pdfts.append(e_pdft)

        w = np.asarray(weights)
        RDM1 = lib.einsum("i,ijk->jk", w, RDM1s)
        e_cell = lib.einsum("i,i->", w, e_cells)
        self.SS = np.mean(ss)
        return e_cell, e_pdfts, RDM1

    def _get_dmet_pdft(
        self,
        cas_norb,
        casdm1_mo,
        casdm2_mo,
        casdm1s,
    ):
        ao2eo = self._ctx.local.get_ao2eo(self._ctx.emb_orbs)
        # Spin separated
        RDM1Sa, RDM1Sb = self._cas_rdm1s_to_local(
            self.mc, cas_norb, casdm1s[0], casdm1s[1]
        )
        ao_basis_rdm1s = self._build_ao_basis_rdm1s(RDM1Sa, RDM1Sb)
        return get_dmet_pdft(
            self.mc,
            self._ctx,
            casdm1s,
            casdm1_mo,
            casdm2_mo,
            ao_basis_rdm1s,
            self.otfnal,
            ao2eo,
        )

    def _build_ao_basis_rdm1s(self, RDM1Sa, RDM1Sb):
        """
        Build full periodic AO-basis spin seprated 1RDMs , Gamma only
        """
        ctx = self._ctx
        Norb = RDM1Sa.shape[0]  # Nimp + Nbath

        # MF 1-RDM in local basis at Gamma
        _, loc_1RDM_kpts, loc_1RDM_R0 = ctx.local.make_loc_1RDM(
            0.0, ctx.mask4Gamma, OEH_type=ctx.OEH_type, dft_HF=None
        )

        # MF 1-RDM projected to emb+core orbital
        emb_core_1RDM = ctx.local.make_emb_space_RDM(
            loc_1RDM_R0, ctx.emb_orbs, ctx.core_orbs, ctx.emb_core_orbs
        )

        # Build spin emb+core 1RDMs
        emb_core_1sa = emb_core_1RDM / 2
        emb_core_1sb = emb_core_1RDM / 2
        emb_core_1sa[:Norb, :Norb] = RDM1Sa  # CAS alpha replaces MF
        emb_core_1sb[:Norb, :Norb] = RDM1Sb  # CAS beta  replaces MF

        # transform to periodic AO basis
        # loc_kpts_to_emb_trial_2 does:
        ao_rdm1sa = ctx.local.loc_kpts_to_emb_trial_2(
            loc_1RDM_R0,
            ctx.emb_orbs,
            ctx.core_orbs,
            ctx.emb_core_orbs,
            emb_core_1sa,
        )
        ao_rdm1sb = ctx.local.loc_kpts_to_emb_trial_2(
            loc_1RDM_R0,
            ctx.emb_orbs,
            ctx.core_orbs,
            ctx.emb_core_orbs,
            emb_core_1sb,
        )

        # Return as (1, NAO, NAO) Gamma only
        return [np.asarray([ao_rdm1sa]), np.asarray([ao_rdm1sb])]

    def _cas_rdm1s_to_local(self, mc, cas_norb, dm1sa_mo, dm1sb_mo):
        """
        Spin-separated full-space 1-RDMs in local basis.
        Only needed for periodic AO density construction.
        """
        core_norb = mc.ncore
        core_MO = mc.mo_coeff[:, :core_norb]
        active_MO = mc.mo_coeff[:, core_norb : core_norb + cas_norb]

        # dm1sa_mo, dm1sb_mo = mc.fcisolver.make_rdm1s(ci, cas_norb, mc.nelecas)

        coredm1 = core_MO @ core_MO.T * 2  # total core — split equally
        casdm1a = lib.einsum(
            "ap,pq,bq->ab", active_MO, dm1sa_mo, active_MO, optimize=True
        )
        casdm1b = lib.einsum(
            "ap,pq,bq->ab", active_MO, dm1sb_mo, active_MO, optimize=True
        )

        return coredm1 / 2 + casdm1a, coredm1 / 2 + casdm1b


def get_dmet_pdft(
    mc,
    pdft_context,
    casdm1s,
    casdm1,
    casdm2,
    ao_basis_rdm1s,
    otfnal,
    ao2eo,
):
    """
    DMET-PDFT (single-state + state-averaged)

    Parameters
    ----------
    mc : CASSCF object (embedding Hamiltonian already used)
    pdft_context : PDFTContext
    rdm1 : total 1-RDM (embedding space)
    casdm1s : spin-separated CAS 1-RDMs (MO basis)
    casdm1 : CAS 1-RDM
    casdm2 : CAS 2-RDM
    ao_basis_rdm1s : AO-basis spin RDMs (embedding corrected)
    otfnal : DFT ot Grid
    """

    print(f"[PDFT] Functional: {pdft_context.otxc}")

    E_mc_pdft, E_ot = _compute_pdft_correction(
        mc,
        otfnal,
        pdft_context,
        casdm1s,
        casdm1,
        casdm2,
        ao_basis_rdm1s,
        ao2eo,
    )

    print("[PDFT] E_mc_pdft:", E_mc_pdft)
    print("[PDFT] E_ot   :", E_ot)

    return E_mc_pdft, E_ot


def _compute_pdft_correction(
    mc,
    ot,
    ctx,
    casdm1s,
    casdm1,
    casdm2,
    ao_basis_rdm1s,
    ao2eo,
):
    from pyscf import ao2mo

    kmf = ctx.kmf
    cell = ctx.cell
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    hyb_x, hyb_c = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)[2]

    #  One-electron + Coulomb
    h = kmf.get_hcore()
    ao_rdm1 = ao_basis_rdm1s[0] + ao_basis_rdm1s[1]

    if abs(hyb_x) > 1e-10 or abs(hyb_c) > 1e-10:
        vj, vk = kmf.get_jk(dm_kpts=np.asarray(ao_basis_rdm1s[0]), hermi=1)
        vj = vj[0] + vj[1]
    else:
        vj = kmf.get_j(dm_kpts=np.asarray(ao_rdm1[0]), hermi=1)
        vk = None

    Te_Vne = np.tensordot(h, np.asarray(ao_rdm1[0]))[0]  # [0] to get scalar
    E_j = np.tensordot(vj, np.asarray(ao_rdm1[0])) / 2

    #  Exchange
    if vk is not None:  # abs(hyb_x) > 1e-10 or abs(hyb_c) > 1e-10
        dm1s = mc.make_rdm1s()
        E_x = -(np.tensordot(vk[0], dm1s[0]) + np.tensordot(vk[1], dm1s[1])) / 2
    else:
        E_x = 0.0

    #  Nuclear
    Vnn = cell.energy_nuc()

    # CAS correlation contribution (hybrid only)
    E_c = 0.0
    if abs(hyb_c) > 1e-10:
        aeri = ao2mo.restore(1, mc.get_h2eff(mc.mo_coeff), mc.ncas)
        E_c = np.tensordot(aeri, casdm2, axes=4) / 2

    #  On-top energy
    E_ot = get_E_ot(
        mc,
        ot,
        casdm1s,
        casdm1,
        casdm2,
        ao_basis_rdm1s,
        ao2eo,
    )

    # MC-PDFT energy (eq 1)
    E_mc_pdft = Vnn + Te_Vne + E_j + hyb_x * E_x + hyb_c * E_c + E_ot

    print("-" * 45)
    print("Breakdown")
    print("Vnn", Vnn)
    print("Te_Vne", Te_Vne)
    print("E_j", E_j)
    print("E_x", E_x)
    print("E_c", E_c)
    print("E_ot", E_ot)
    print("-" * 45)
    # Hybrid scaling (eq 2)
    # lambda_hyb = 0.25 if "0" in ot.otxc else 0.0
    # e_tot = lambda_hyb * e_cell + (1 - lambda_hyb) * E_mc_pdft

    return E_mc_pdft, E_ot


def get_E_ot(
    mc,
    ot,
    casdm1s,
    casdm1,
    casdm2,
    ao_basis_rdm1s,
    ao2eo,
    max_memory=2000,  # Refactor , settings shuold be controlled
    hermi=1,
):
    from pyscf.mcpdft.otpd import get_ontop_pair_density
    from pyscf.mcpdft import _dms

    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    mo_cas = mc.mo_coeff[:, ncore:nocc]
    mo_cas_ao2eo = lib.einsum("ap,pq->aq", ao2eo[0].real, mo_cas)

    oneCDMs = np.asarray(
        [
            ao_basis_rdm1s[0][0],  # alpha (NAO, NAO)
            ao_basis_rdm1s[1][0],
        ]
    )  # beta  (NAO, NAO)
    ni, dens_deriv = ot._numint, ot.dens_deriv
    norbs_ao = ao2eo.shape[1]
    E_ot = 0.0

    make_rho = tuple(
        ni._gen_rho_evaluator(ot.mol, oneCDMs[i, :, :], hermi) for i in range(2)
    )

    for ao, mask, weight, _ in ni.block_loop(
        ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory
    ):
        rho = np.asarray([m[0](0, ao, mask, ot.xctype) for m in make_rho])
        cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)
        Pi = get_ontop_pair_density(ot, rho, ao, cascm2, mo_cas_ao2eo, dens_deriv, mask)
        eot = ot.eval_ot(rho=rho, Pi=Pi)[0]
        E_ot += np.dot(eot, weight)

    return E_ot
