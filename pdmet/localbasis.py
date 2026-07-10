#!/usr/bin/env python -u
"""
pDMET: Density Matrix Embedding theory for Periodic Systems
Copyright (C) 2018 Hung Q. Pham. All Rights Reserved.
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

Email: Hung Q. Pham <pqh3.14@gmail.com>
"""

import warnings
import numpy as np
import scipy.linalg as la
from functools import reduce
from pyscf.pbc.tools import pbc as pbctools
from pyscf import lib, ao2mo
from pyscf.pbc import scf
from pyscf.lo import iao as _pyscf_iao
from pdmet import helper, df, df_hamiltonian
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from pdmet.tools import tchkfile
from pdmet.settings import LOMethod
from pdmet.tools.tchkfile import to_numpy


def _klowdin(C, S, tol=1.0e-12):
    """Lowdin orthogonalize C[k] against S[k] at every k-point.
    Symmetric (Lowdin) orthogonalization
    """
    out = np.zeros_like(C)
    for k in range(C.shape[0]):
        M = reduce(np.dot, (C[k].conj().T, S[k], C[k]))
        e, v = la.eigh(M)
        keep = e > tol
        X = (v[:, keep] / np.sqrt(e[keep])) @ v[:, keep].conj().T
        out[k] = C[k] @ X
    return out


def _check_orthonormal_kbasis(C_ao_lo, S, tol=1.0e-10):
    """
    Check if C_ao_lo[k]^H S[k] C_ao_lo[k] = I at every k-point.
    Returns (is_orth: bool, max_err: float). Used as a final sanity check
    """
    nkpts, _, nlo = C_ao_lo.shape
    Id = np.eye(nlo)
    max_err = 0.0
    for k in range(nkpts):
        M = C_ao_lo[k].conj().T @ S[k] @ C_ao_lo[k]
        max_err = max(max_err, float(np.abs(M - Id).max()))
    return max_err < tol, max_err


def _ao_to_atom(cell):
    """List, length nao, mapping each AO index to its atom index (0-based)."""
    return [int(lbl.split()[0]) for lbl in cell.ao_labels()]


def _atom_block_lowdin(cell, C_virt, S, virt_idx, tol=1.0e-12):
    """
    Per-atom Lowdin orthogonalization of the PAO virtuals
    """
    nkpts, nao, nvirt = C_virt.shape
    natm = cell.natm
    ao_atom = _ao_to_atom(cell)

    # Group virtual *column indices* by host atom.
    blocks = [[] for _ in range(natm)]
    for vpos, ao_i in enumerate(virt_idx):
        blocks[ao_atom[ao_i]].append(vpos)

    out = C_virt.copy()
    for k in range(nkpts):
        for cols in blocks:
            if len(cols) <= 1:
                continue  # nothing to mix
            block = C_virt[k][:, cols]  # (nao, n_block)
            M = block.conj().T @ S[k] @ block  # (n_block, n_block)
            e, v = la.eigh(M)
            keep = e > tol
            X = (v[:, keep] / np.sqrt(e[keep])) @ v[:, keep].conj().T
            out[k][:, cols] = block @ X
    return out


def _get_iao_virt_kbasis(cell, C_val, S, virt_idx, full_virt=False, max_ovlp=False):
    """
    Construct PAO virtuals as the orthogonal complement of the IAO valence.

    Algorithm
    ---------
    A(k) = I − C_val(k) * C_val(k)^H * S(k)
    full_virt=False  →  C_virt(k) = A(k)[:, virt_idx]   (size nao − nval)
    full_virt=True   →  C_virt(k) = A(k)[:, :]          (size nao, rank
                                                          nao − nval — the
                                                          extra columns are
                                                          dropped by Lowdin)
    max_ovlp=True      apply per-atom Lowdin so each PAO stays atom-centered

    """
    nkpts, nao, _ = C_val.shape
    Id = np.eye(nao)

    if full_virt:
        cols = np.arange(nao)
    else:
        cols = np.asarray(virt_idx, dtype=int)

    nvirt = len(cols)
    C_virt = np.zeros((nkpts, nao, nvirt), dtype=np.complex128)
    for k in range(nkpts):
        A = Id - C_val[k] @ C_val[k].conj().T @ S[k]
        C_virt[k] = A[:, cols]

    if max_ovlp and not full_virt:
        # Per-atom Lowdin only meaningful when columns are atom-tagged
        C_virt = _atom_block_lowdin(cell, C_virt, S, cols)

    return C_virt


def _build_iao_pao_labels(cell, minao, full_virt=False):
    """Labels for [IAO valence | PAO virtual] in that order."""
    pmol = _pyscf_iao.reference_mol(cell, minao)
    B1_labels = cell.ao_labels()
    B2_labels = pmol.ao_labels()

    val_labels = list(B2_labels)
    if full_virt:
        virt_labels = list(B1_labels)
    else:
        virt_labels = [lbl for lbl in B1_labels if lbl not in B2_labels]
    return val_labels + virt_labels, B2_labels


def make_iao_pao_kbasis(
    cell,
    kmf=None,
    kpts=None,
    mo_coeff_kpts=None,
    mo_occ_kpts=None,
    minao="minao",
    orth_virt=True,
    full_virt=False,
    max_ovlp=True,
    tol=1.0e-10,
    return_labels=True,
):
    """
    Build the k-adapted IAO + PAO transformation C_ao_lo, mirroring
    libdmet.lo.iao.get_C_ao_lo_iao for the ncore=0 path (RHF/ROHF, no
    smearing — smearing is Phase 2).

    For pDMET use
    -------------
    pDMET requires orthonormal LOs and a square AO→LO transform
    (``nlo == nao``), so it always calls this with the defaults
    ``orth_virt=True, full_virt=False``. The recommended call from
    ``Local.__init__`` (pseudopotential ROHF, atom-centered PAOs):

        >>> C_ao_lo, _, _, lo_labels = make_iao_pao_kbasis(
        ...     cell, kmf=kmf,
        ...     minao={"Ni": "gth-szv-molopt-sr", "O": "gth-szv"},
        ...     max_ovlp=(lobasis.pao_localization
        ...               == PAOLocalization.MAX_OVLP),
        ... )

    pDMET only consumes ``C_ao_lo`` (→ ``Local.ao2lo``) and ``lo_labels``
    (→ ``Local.lo_labels`` for ``imp_orbital_filter`` matching). ``C_val``
    and ``C_virt`` are kept for diagnostics and plotting.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
    kmf  : pyscf.pbc.scf object (KRHF, KROHF, ...). Optional if you pass
           mo_coeff_kpts, mo_occ_kpts, kpts directly.
    kpts : (Nk, 3) ndarray. Defaults to ``kmf.kpts``.
    mo_coeff_kpts, mo_occ_kpts : Optional overrides; default to kmf's.
    minao : str | dict. B2 reference basis. For ECP/PP calculations use
            something like ``'gth-szv-molopt-sr'`` or a per-atom dict
            ``{"Ni": "gth-szv-molopt-sr", "O": "gth-szv"}``.
    orth_virt : Lowdin-orthogonalize PAO virtuals globally after projection.
                **Always True for pDMET use.** Disable only for analysis
                (e.g. inspecting non-orthogonal PAO character). Must be
                False if ``full_virt=True``.
    full_virt : keep ALL nao virtuals (overcomplete; rank is still
                nao − nval). Default False keeps only nao − nval virtuals
                whose AO labels are NOT in the B2 reference. **For pDMET
                this MUST be False** — embedding requires nlo == nao.
                ``full_virt=True`` is for diagnostic/comparison use only.
    max_ovlp  : per-atom Lowdin within the PAO subspace so each PAO stays
                atom-centered. **Default True** — recommended for pDMET
                because the impurity selector (``imp_orbital_filter``)
                relies on labels being chemically meaningful. Set False
                to reproduce libdmet's default behavior exactly.
    tol       : tolerance for the final orthonormality check.
    return_labels : whether to return the LO labels list. **Always True
                for pDMET** — labels drive ``imp_orbital_filter``.

    Returns
    -------
    C_ao_lo : (Nk, nao, nlo) complex   — full IAO+PAO transformation
    C_val   : (Nk, nao, nval) complex  — IAO valence (diagnostic)
    C_virt  : (Nk, nao, nvirt) complex — PAO virtual (diagnostic)
    labels  : list[str] of length nlo  — AO-style labels (val first)
              or None if ``return_labels=False``

    Notes
    -----
    ROHF: ``mo_occ_kpts`` has values in {0, 1, 2}. The ``mo_occ > 0``
    filter selects both singly and doubly occupied MOs — correct for both
    RHF and ROHF, no special handling needed.

    UHF: not yet supported (would need a spin-loop wrapper).
    """
    # 1. Resolve inputs
    if kmf is not None:
        if kpts is None:
            kpts = kmf.kpts
        if mo_coeff_kpts is None:
            mo_coeff_kpts = kmf.mo_coeff_kpts
        if mo_occ_kpts is None:
            mo_occ_kpts = kmf.mo_occ_kpts
    if kpts is None or mo_coeff_kpts is None or mo_occ_kpts is None:
        raise ValueError("Need kmf or (kpts, mo_coeff_kpts, mo_occ_kpts).")

    mo_coeff_kpts = [to_numpy(c) for c in mo_coeff_kpts]
    mo_occ_kpts = [to_numpy(o) for o in mo_occ_kpts]
    nkpts = len(kpts)
    nao = cell.nao_nr()

    if orth_virt and full_virt:
        raise ValueError(
            "orth_virt=True with full_virt=True is rank-deficient; "
            "either drop full_virt or set orth_virt=False."
        )

    # 2. Occupied MOs at each k. RHF: mo_occ ∈ {0, 2}; ROHF: ∈ {0, 1, 2}.
    orbocc = [mo_coeff_kpts[k][:, mo_occ_kpts[k] > 0] for k in range(nkpts)]
    has_frac = any(
        (
            (mo_occ_kpts[k] != 0.0) & (mo_occ_kpts[k] != 1.0) & (mo_occ_kpts[k] != 2.0)
        ).any()
        for k in range(nkpts)
    )
    if has_frac:
        warnings.warn(
            "make_iao_pao_kbasis: detected fractional mo_occ. "
            "Phase-1 IAO uses a hard `mo_occ > 0` filter, which over-"
            "includes near-Fermi bands. Smearing-aware IAO is Phase 2."
        )

    # 3. AO overlap S(k)
    S = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))

    # 4. Build IAO valence — pyscf supports str OR dict for minao
    C_val = np.asarray(
        _pyscf_iao.iao(cell, orbocc, minao=minao, kpts=kpts),
        dtype=np.complex128,
    )
    C_val = _klowdin(C_val, S)  # global Lowdin within valence

    # 5. Identify partial-PAO column indices (AOs in B1\B2)
    pmol = _pyscf_iao.reference_mol(cell, minao)
    B1_labels = cell.ao_labels()
    B2_labels = pmol.ao_labels()
    virt_idx = np.array(
        [i for i, lbl in enumerate(B1_labels) if lbl not in B2_labels],
        dtype=int,
    )
    nval = len(B2_labels)
    if not full_virt:
        assert nval + len(virt_idx) == nao, (
            f"IAO/PAO partition mismatch: nao={nao}, nval={nval}, "
            f"|B1\\B2|={len(virt_idx)}. Check your minao reference."
        )

    # 6. Build PAO virtuals
    if (not full_virt) and len(virt_idx) == 0:
        C_virt = np.zeros((nkpts, nao, 0), dtype=np.complex128)
    else:
        C_virt = _get_iao_virt_kbasis(
            cell,
            C_val,
            S,
            virt_idx,
            full_virt=full_virt,
            max_ovlp=max_ovlp,
        )
        if orth_virt:
            C_virt = _klowdin(C_virt, S)  # global Lowdin within virtuals

    # 7. Stack [val | virt]
    C_ao_lo = np.concatenate([C_val, C_virt], axis=-1).astype(np.complex128)

    # 8. Final orthonormality sanity check (libdmet pattern).
    # full_virt=True is overcomplete by construction — skip the check.
    if not full_virt:
        is_orth, err = _check_orthonormal_kbasis(C_ao_lo, S, tol=tol)
        if not is_orth:
            warnings.warn(
                f"IAO+PAO set is not orthonormal: max |C^H S C - I| = "
                f"{err:.3e} > tol={tol:.0e}. Check your minao reference "
                f"and the smoothness of mo_coeff_kpts (e.g. KRHF "
                f"convergence)."
            )

    # 9. Labels
    if return_labels:
        labels, _ = _build_iao_pao_labels(cell, minao, full_virt=full_virt)
    else:
        labels = None

    return C_ao_lo, C_val, C_virt, labels


class Local:
    def __init__(
        self,
        cell,
        kmf,
        lobasis,
        is_KROHF=False,
        xc_omega=0.2,
        OEH_type="FOCK",
    ):
        """
        Args:
            kmf        : a k-dependent mean-field wf
            w90        : a converged wannier90 object
            lobasis    : LocalBasisSettings dataclass instance
            is_KROHF   : whether the kmf is a KROHF object (vs KRHF)
            xc_omega   : if not None, range-separation parameter for pDFT (in inverse bohr). Default 0.2 is a common choice for solids.

        Indices
            u, v    : k-space ao indices
            i, j    : k-space mo/local indices
            m, n    : embedding indices
            capital : R-space indices
            R, S, T : R-spacce lattice vector

        """

        # Collect cell and kmf object information
        lobasis.validate()  # sanity check the settings dataclass
        self.lobasis = lobasis
        self.cell = cell
        self.spin = cell.spin
        self.e_tot = kmf.e_tot
        self.kmesh = tuple(kpts_to_kmesh(cell, kmf.kpts))
        self.kmf = kmf
        self._is_KROHF = is_KROHF
        self.kpts = kmf.kpts
        self.Nkpts = kmf.kpts.shape[0]
        self.nao = cell.nao_nr()

        self.mo_coeff_kpts = [to_numpy(c) for c in kmf.mo_coeff_kpts]
        self.mo_occ_kpts = [to_numpy(o) for o in kmf.mo_occ_kpts]
        self.mo_energy_kpts = [to_numpy(e) for e in kmf.mo_energy_kpts]

        _, self.phase = self.get_phase(self.cell, self.kpts, self.kmesh)

        if lobasis.method == LOMethod.WANNIER:
            self.w90 = lobasis.w90
            if lobasis.w90_chkfile is not None:
                self.w90 = tchkfile.load_w90(self.w90, lobasis.w90_chkfile)

            self.ao2lo = self.get_ao2lo_w90(self.w90)
            self.lo_labels = None
            self._active_band_mask = self.w90.band_included_list
        else:
            self.w90 = None
            self._active_band_mask = None
            if lobasis.lo_chkfile is not None:
                cached = tchkfile.load_lo_iao(lobasis.lo_chkfile)
                self.ao2lo = cached["ao2lo"]
                self.lo_labels = cached["lo_labels"]
                self.minao = cached["minao"]
                self._verify_loaded_lo()
            else:
                C_ao_lo, _, _, self.lo_labels = self.get_ao2lo_iao_pao(
                    minao=lobasis.minao, full_return=True
                )
                self.ao2lo = C_ao_lo
                self.minao = lobasis.minao

        self.nlo = self.ao2lo.shape[-1]

        # -------------------------------------------------------------
        # Construct the effective Hamiltonian due to the frozen core  |
        # -------------------------------------------------------------
        if self._is_KROHF:
            self.nelec = [self.cell.nelec[0], self.cell.nelec[1]]

        self.nelec_total = 0
        for kpt, mo_occ in enumerate(self.mo_occ_kpts):
            active = self._get_active_mo_indices(kpt, len(mo_occ))
            self.nelec_total += int(np.asarray(mo_occ)[active].sum())

        self.nelec_per_cell = self.nelec_total // self.Nkpts

        full_OEI_k = kmf.get_hcore()
        coreDM_kpts = []
        for kpt, mo_coeff in enumerate(self.mo_coeff_kpts):
            core_band = self._get_core_band_mask(mo_coeff, kpt)

            if not np.any(core_band):
                nao = mo_coeff.shape[0]
                coreDM_kpts.append(np.zeros((nao, nao), dtype=np.complex128))
            else:
                coreDMmo = self.mo_occ_kpts[kpt][core_band].copy()
                mo_k = mo_coeff[:, core_band]
                coreDMao = reduce(np.dot, (mo_k, np.diag(coreDMmo), mo_k.T.conj()))
                coreDM_kpts.append(coreDMao)

        self.coreDM_kpts = np.asarray(coreDM_kpts, dtype=np.complex128)

        has_core_bands = np.any(self.coreDM_kpts != 0.0)
        if has_core_bands:
            if self._is_KROHF:
                dma = dmb = self.coreDM_kpts * 0.5
                coreJK_kpts_ab = kmf.get_veff(
                    cell, dm_kpts=[dma, dmb], hermi=1, kpts=self.kpts, kpts_band=None
                )
                coreJK_kpts = 0.5 * (coreJK_kpts_ab[0] + coreJK_kpts_ab[1])
            else:
                coreJK_kpts = kmf.get_veff(
                    cell, self.coreDM_kpts, hermi=1, kpts=self.kpts, kpts_band=None
                )
        else:
            # No core bands — coreJK contribution is zero
            nkpts = len(self.kpts)
            nao = self.coreDM_kpts.shape[-1]
            coreJK_kpts = np.zeros((nkpts, nao, nao), dtype=np.complex128)

        # Core energy from the frozen orbitals
        self.e_core = (
            cell.energy_nuc()
            + 1.0
            / self.Nkpts
            * lib.einsum(
                "kij,kji->", full_OEI_k + 0.5 * coreJK_kpts, self.coreDM_kpts
            ).real
        )

        # 1e integral for the active part
        self.actOEI_kpts = full_OEI_k + coreJK_kpts
        self.fullfock_kpts = kmf.get_fock(s1e=kmf.get_ovlp(), dm=kmf.make_rdm1())
        self.loc_actFOCK_kpts = self.ao_2_loc(self.fullfock_kpts, self.ao2lo)
        # Spin-resolved LO densities (ROHF only), filled by make_loc_1RDM_kpts
        self.loc_1RDM_a_kpts = self.loc_1RDM_b_kpts = None

        # DF-like DMET: the effective-Hamiltonian (DF/DFT) cost function needs the
        # mean-field J/K, density and a KS object. These are consumed *only* by
        # df_hamiltonian.get_OEH_kpts, which runs solely when OEH_type != "FOCK".
        self.xc_omega = xc_omega
        if OEH_type != "FOCK":
            self.dm_kpts = self.kmf.make_rdm1()
            self.vj, self.vk = self.kmf.get_jk(dm_kpts=self.dm_kpts)
            self.h_core = self.kmf.get_hcore()
            if self._is_KROHF:
                self.kks = scf.KROKS(self.cell, self.kpts).density_fit()
            else:
                self.kks = scf.KKS(self.cell, self.kpts).density_fit()
            self.kks.with_df._cderi = self.kmf.with_df._cderi
            if self.xc_omega is not None:
                self.vklr = self.kmf.get_k(
                    self.cell, self.dm_kpts, 1, self.kpts, None, omega=self.xc_omega
                )
                self.vksr = self.vk - self.vklr

    def _verify_loaded_lo(self):
        """Sanity-check that an IAO+PAO chkfile matches the current cell."""
        ao2lo = np.asarray(self.ao2lo)
        expected = (self.Nkpts, self.cell.nao, self.cell.nao)
        assert ao2lo.shape == expected, (
            f"loaded ao2lo shape {ao2lo.shape} != expected {expected} "
            f"(Nkpts={self.Nkpts}, nao={self.cell.nao}). "
            f"Wrong chkfile, or cell/basis changed since save."
        )
        if self.lo_labels is not None:
            assert len(self.lo_labels) == self.cell.nao, (
                f"loaded lo_labels has {len(self.lo_labels)} entries, "
                f"expected {self.cell.nao}. Wrong chkfile?"
            )

    def _get_active_mo_indices(self, kpt, nmo):
        """
        Indices (or boolean mask) of MOs in the active embedding space at k=kpt.

        Wannier               : same set every k (w90.band_included_list).
        IAO+PAO               : all MOs active (returns slice(None)).

        """
        if self._active_band_mask is None:
            return slice(None)  # all active

        first = self._active_band_mask[0]
        if isinstance(first, (int, np.integer)):
            return self._active_band_mask  # Wannier shape
        return self._active_band_mask[kpt]  # per-k shape

    def _get_core_band_mask(self, mo_coeff_at_k, kpt):
        """Get a boolean mask for the core (frozen) bands based on mo_coeff."""
        nmo = mo_coeff_at_k.shape[1]
        active = self._get_active_mo_indices(kpt, nmo)
        if isinstance(active, slice):
            return np.zeros(nmo, dtype=bool)  # all active -> no core
        mask = np.ones(nmo, dtype=bool)
        mask[active] = False
        return mask

    def make_loc_1RDM_kpts(
        self,
        umat,
        mask4Gamma,
        OEH_type="FOCK",
        get_band=False,
        get_ham=False,
        dft_HF=None,
    ):
        """
        Construct 1-RDM at each k-point in the local basis given a u mat.
        mask4Gamma is used for the Gamma-sampling case.
        """
        # Modified mean-field Hamiltonian: h_tilde = h + u
        if OEH_type == "FOCK":
            OEH_kpts = self.loc_actFOCK_kpts + umat
        elif mask4Gamma is not None:
            # DF-like cost function, Gamma sampling
            OEH_kpts = self.loc_actFOCK_kpts[0].copy()
            OEH_kpts[mask4Gamma] = df_hamiltonian.get_OEH_kpts(
                self, umat, xc_type=OEH_type, dft_HF=dft_HF
            )[0][mask4Gamma]
            OEH_kpts = OEH_kpts.reshape(-1, self.nlo, self.nlo)
        else:
            OEH_kpts = df_hamiltonian.get_OEH_kpts(
                self, umat, xc_type=OEH_type, dft_HF=dft_HF
            )

        # eigh returns eigenvalues already in ascending order per k-point,
        # so no re-sort is needed.
        eigvals, eigvecs = np.linalg.eigh(OEH_kpts)

        if get_band:
            return eigvals, eigvecs
        if get_ham:
            return OEH_kpts, eigvals, eigvecs

        if self._is_KROHF:
            mo_occ = helper.get_occ_rohf(self.nelec, eigvals)
            # Keep the spin-resolved densities (alpha: occ>0, beta: occ==2).
            # They are the exact embedded-ROHF guess; total DM is their sum.
            occ = np.asarray(mo_occ)
            self.loc_1RDM_a_kpts = np.einsum(
                "kij,kj,klj->kil", eigvecs, (occ > 0).astype(float), eigvecs.conj()
            )
            self.loc_1RDM_b_kpts = np.einsum(
                "kij,kj,klj->kil", eigvecs, (occ == 2).astype(float), eigvecs.conj()
            )
        else:
            mo_occ = helper.get_occ_rhf(self.nelec_total, eigvals)

        # mo_occ is 0 on virtuals, so no occupied-orbital masking is needed:
        # loc_OED[k] = (C[k] * n[k]) @ C[k]^H
        loc_OED = np.einsum("kij,kj,klj->kil", eigvecs, mo_occ, eigvecs.conj())

        return OEH_kpts, loc_OED

    def make_loc_1RDM(self, umat, mask4Gamma, OEH_type="FOCK", dft_HF=None):
        """
        Construct the local 1-RDM at the reference unit cell
        """
        loc_OEH_kpts, loc_1RDM_kpts = self.make_loc_1RDM_kpts(
            umat, mask4Gamma, OEH_type=OEH_type, dft_HF=dft_HF
        )
        loc_1RDM_R0 = self.k_to_R0(loc_1RDM_kpts)
        return loc_OEH_kpts, loc_1RDM_kpts, loc_1RDM_R0

    def get_emb_OEI(self, ao2eo):
        """Get OEI projected into the embedding basis"""
        OEI = lib.einsum("kum,kuv,kvn->mn", ao2eo.conj(), self.actOEI_kpts, ao2eo)
        self.is_real(OEI)
        return OEI.real

    def get_real_space_OEI_for_MCPDFT(self, loc_1RDM_kpts, ao2eo):
        """Get OEI+JK in AO basis - under development"""
        ao_1RDM_kpts = self.loc_2_ao(loc_1RDM_kpts)
        if self._is_KROHF:
            dma = dmb = ao_1RDM_kpts * 0.5
            ao_JK_ab = self.kmf.get_veff(
                self.cell, dm_kpts=[dma, dmb], hermi=1, kpts=self.kpts, kpts_band=None
            )
            ao_JK = 0.5 * (ao_JK_ab[0] + ao_JK_ab[1])
        else:
            ao_JK = self.kmf.get_veff(
                self.cell, dm_kpts=ao_1RDM_kpts, hermi=1, kpts=self.kpts, kpts_band=None
            )
        fock = self.actOEI_kpts + ao_JK
        self.is_real(fock)
        return fock.real

    def get_core_OEI(self, ao2core):
        """Get OEI projected into the core (unentangled) basis"""
        OEI = lib.einsum("kum,kuv,kvn->mn", ao2core.conj(), self.actOEI_kpts, ao2core)
        self.is_real(OEI)
        return OEI.real

    def get_emb_FOCK(self, emb_orbs, loc_OEH_kpts):
        """Get modified FOCK in embedding basis"""
        lo2eo = lib.einsum("Rk, Rim -> kim", self.phase.conj(), emb_orbs)
        emb_fock_kpts = lib.einsum("kim,kij,kjn->mn", lo2eo.conj(), loc_OEH_kpts, lo2eo)
        self.is_real(emb_fock_kpts)
        return emb_fock_kpts.real

    def get_emb_JK(self, loc_1RDM_kpts, ao2eo):
        """Get embedding JK from a local 1-RDM"""
        ao_1RDM_kpts = self.loc_2_ao(loc_1RDM_kpts)
        if self._is_KROHF:
            dma = dmb = ao_1RDM_kpts * 0.5
            ao_JK_ab = self.kmf.get_veff(
                self.cell, dm_kpts=[dma, dmb], hermi=1, kpts=self.kpts, kpts_band=None
            )
            ao_JK = 0.5 * (ao_JK_ab[0] + ao_JK_ab[1])
        else:
            ao_JK = self.kmf.get_veff(
                self.cell, dm_kpts=ao_1RDM_kpts, hermi=1, kpts=self.kpts, kpts_band=None
            )

        emb_JK = lib.einsum("kum,kuv,kvn->mn", ao2eo.conj(), ao_JK, ao2eo)
        self.is_real(emb_JK)
        return emb_JK.real

    def get_core_JK(self, ao2core, loc_core_1RDM):
        """Get JK projected into the core (unentangled) basis"""
        ao_core_kpts = self.loc_2_ao(loc_core_1RDM)

        # For Debugging
        if ao_core_kpts.size == 0 or np.allclose(ao_core_kpts, 0):
            print("[DMET WARNING] Core density matrix is zero → skipping core JK")
            return np.zeros((ao2core.shape[-1], ao2core.shape[-1]))

        if self._is_KROHF:
            dma = dmb = ao_core_kpts * 0.5
            ao_core_JK_ab = self.kmf.get_veff(
                self.cell, dm_kpts=[dma, dmb], hermi=1, kpts=self.kpts, kpts_band=None
            )
            ao_core_JK = 0.5 * (ao_core_JK_ab[0] + ao_core_JK_ab[1])
        else:
            ao_core_JK = self.kmf.get_veff(
                self.cell, dm_kpts=ao_core_kpts, hermi=1, kpts=self.kpts, kpts_band=None
            )

        core_JK = lib.einsum(
            "kum,kuv,kvn->mn", ao2core.conj(), ao_core_JK, ao2core, optimize=True
        )
        self.is_real(core_JK)
        return core_JK.real

    def get_emb_coreJK(self, emb_JK, emb_TEI, emb_1RDM):
        """Get embedding core JK
        Attributes:
         emb_JK  : total JK projected into the embedding space
         emb_TEI : TEI projected into the embedding space
         emb_1RDM: 1RDM projected into the embedding space
        """
        J = lib.einsum("pqrs,rs->pq", emb_TEI, emb_1RDM)
        K = lib.einsum("prqs,rs->pq", emb_TEI, emb_1RDM)
        emb_actJK = J - 0.5 * K
        emb_coreJK = (
            emb_JK - emb_actJK
        )  # Subtract JK from the active space (frag + bath) from the totak JK
        return emb_coreJK

    def get_emb_TEI(self, ao2eo):
        """Get embedding TEI with density fitting"""
        mydf = self.kmf.with_df
        TEI = df.get_emb_eri_gdf(self.cell, mydf, ao2eo)[0]
        return TEI

    def get_emb_coreJK_df(self, emb_JK, B, emb_1RDM):
        """
        DF version: no 4-index TEI needed
        B: (naux, nemb, nemb)
        """
        # Coulomb
        X = lib.einsum("Lrs,rs->L", B, emb_1RDM, optimize=True)
        J = lib.einsum("Lpq,L->pq", B, X, optimize=True)
        # Exchange
        K = lib.einsum("Lpr,Lqs,rs->pq", B, B, emb_1RDM, optimize=True)

        emb_actJK = J - 0.5 * K
        emb_coreJK = emb_JK - emb_actJK

        return emb_coreJK

    def get_emb_B(self, ao2eo):
        """
        Get 3-index DF tensor in embedding basis.
        Shape: (naux, nemb, nemb)
        For use in _impurity_energy_from_cas_df to avoid O(nemb^4) memory.
        """
        mydf = self.kmf.with_df
        return df.get_emb_Lmn(self.cell, mydf, ao2eo)

    def get_TEI(self, ao2eo):
        """Get embedding TEI without density fitting"""
        kconserv = pbctools.get_kconserv(self.cell, self.kpts)

        Nkpts, nao, neo = ao2eo.shape
        TEI = 0.0
        for i in range(Nkpts):
            for j in range(Nkpts):
                for k in range(Nkpts):
                    l = kconserv[i, j, k]  # noqa: E741
                    ki, COi = self.kpts[i], ao2eo[i]
                    kj, COj = self.kpts[j], ao2eo[j]
                    kk, COk = self.kpts[k], ao2eo[k]
                    kl, COl = self.kpts[l], ao2eo[l]
                    TEI += self.kmf.with_df.ao2mo(
                        [COi, COj, COk, COl], [ki, kj, kk, kl], compact=False
                    )

        return TEI.reshape(neo, neo, neo, neo).real / Nkpts

    def get_loc_TEI(self, ao2lo=None):
        """Get local TEI in R-space without density fitting"""
        kconserv = pbctools.get_kconserv(self.cell, self.kpts)
        if ao2lo is None:
            ao2lo = self.ao2lo

        Nkpts, nao, nlo = ao2lo.shape
        size = Nkpts * nlo
        mo_phase = lib.einsum("kui,Rk->kuRi", ao2lo, self.phase.conj()).reshape(
            Nkpts, nao, size
        )
        TEI = 0.0
        for i in range(Nkpts):
            for j in range(Nkpts):
                for k in range(Nkpts):
                    l = kconserv[i, j, k]  # noqa: E741
                    ki, COi = self.kpts[i], mo_phase[i]
                    kj, COj = self.kpts[j], mo_phase[j]
                    kk, COk = self.kpts[k], mo_phase[k]
                    kl, COl = self.kpts[l], mo_phase[l]
                    TEI += self.kmf.with_df.ao2mo(
                        [COi, COj, COk, COl], [ki, kj, kk, kl], compact=False
                    )
        self.is_real(TEI)
        return TEI.reshape(size, size, size, size).real / Nkpts

    def loc_to_emb_TEI(self, loc_TEI, emb_orbs):
        """Transform local TEI in R-space to embedding space"""
        NRs, nlo, neo = emb_orbs.shape
        emb_orbs = emb_orbs.reshape([NRs * nlo, neo])
        TEI = ao2mo.incore.full(ao2mo.restore(8, loc_TEI, neo), emb_orbs, compact=False)
        TEI = TEI.reshape(neo, neo, neo, neo)
        return TEI

    def emb_to_loc_kpts(self, emb_matrix, emb_orbs):
        """Get k-space embedding 1e quantities in the k-space local basis
        TODO: DEBUGGING THIS

        """
        lo2eo = lib.einsum("Rk, Rim -> kim", self.phase.conj(), emb_orbs)
        loc_coreJK_kpts = lib.einsum("kim,mn,kjn->kij", lo2eo, emb_matrix, lo2eo.conj())
        return loc_coreJK_kpts

    def loc_kpts_to_emb(self, RDM_kpts, emb_orbs):
        """Transform k-space 1-RDM in local basis to embedding basis"""
        lo2eo = lib.einsum("Rk, Rim -> kim", self.phase.conj(), emb_orbs)
        emb_1RDM = lib.einsum("kim, kij, kjn -> mn", lo2eo.conj(), RDM_kpts, lo2eo)
        self.is_real(emb_1RDM)
        return emb_1RDM.real

    def make_emb_space_RDM(self, RDM_kpts, emb_orbs, core_orbs, emb_core_orbs):
        """Transform k-space 1-RDM in local basis to embedding basis"""
        lo2eo = lib.einsum("Rk, Rim -> kim", self.phase.conj(), emb_orbs)
        emb_1RDM = lib.einsum("kim, kij, kjn -> mn", lo2eo.conj(), RDM_kpts, lo2eo)
        lo2core = lib.einsum("Rk, Rim -> kim", self.phase.conj(), core_orbs)
        core_1RDM = lib.einsum("kim, kij, kjn -> mn", lo2core.conj(), RDM_kpts, lo2core)  # noqa: F841
        lo2_emb_core = lib.einsum("Rk, Rim -> kim", self.phase.conj(), [emb_core_orbs])
        emb_core_1RDM_for_mcpdft = lib.einsum(
            "kim, kij, kjn -> mn", lo2_emb_core.conj(), RDM_kpts, lo2_emb_core
        )
        emb_core_1RDM_for_mcpdft_lo_basis = lib.einsum(
            "mi, ij, nj -> mn",
            lo2_emb_core[0],
            emb_core_1RDM_for_mcpdft,
            lo2_emb_core[0],
        ).real
        emb_core_1RDM_for_mcpdft_ao_basis = lib.einsum(  # noqa: F841
            "mi, ij, nj -> mn",
            self.ao2lo[0],
            emb_core_1RDM_for_mcpdft_lo_basis,
            self.ao2lo[0],
        ).real
        dummy_ao2eo = lib.einsum("ui, im -> um", self.ao2lo[0], lo2_emb_core[0])
        emb_core_1RDM_for_mcpdft_ao_basis2 = lib.einsum(  # noqa: F841
            "mi, ij, nj -> mn",
            dummy_ao2eo,
            emb_core_1RDM_for_mcpdft_lo_basis,
            dummy_ao2eo.conj(),
        ).real  # noqa: F841
        ao2eo_core_emb = self.get_ao2eo([emb_core_orbs])
        emb_core_1RDM_for_mcpdft_ao_basis3 = lib.einsum(  # noqa: F841
            "mi, ij, nj -> mn",
            ao2eo_core_emb[0],
            emb_core_1RDM_for_mcpdft_lo_basis,
            ao2eo_core_emb.conj()[0],
        ).real
        self.is_real(emb_1RDM)
        return emb_core_1RDM_for_mcpdft

    def loc_kpts_to_emb_trial_2(
        self, RDM_kpts, emb_orbs, core_orbs, emb_core_orbs, emb_core_1RDM_for_mcpdft
    ):
        """Transform k-space 1-RDM in local basis to embedding basis"""
        lo2eo = lib.einsum("Rk, Rim -> kim", self.phase.conj(), emb_orbs)
        emb_1RDM = lib.einsum("kim, kij, kjn -> mn", lo2eo.conj(), RDM_kpts, lo2eo)
        lo2core = lib.einsum("Rk, Rim -> kim", self.phase.conj(), core_orbs)
        core_1RDM = lib.einsum("kim, kij, kjn -> mn", lo2core.conj(), RDM_kpts, lo2core)  # noqa: F841
        lo2_emb_core = lib.einsum("Rk, Rim -> kim", self.phase.conj(), [emb_core_orbs])
        emb_core_1RDM_for_mcpdft_lo_basis = lib.einsum(
            "mi, ij, nj -> mn",
            lo2_emb_core[0],
            emb_core_1RDM_for_mcpdft,
            lo2_emb_core[0],
        ).real
        emb_core_1RDM_for_mcpdft_ao_basis = lib.einsum(
            "mi, ij, nj -> mn",
            self.ao2lo[0],
            emb_core_1RDM_for_mcpdft_lo_basis,
            self.ao2lo[0],
        ).real
        dummy_ao2eo = lib.einsum("ui, im -> um", self.ao2lo[0], lo2_emb_core[0])
        emb_core_1RDM_for_mcpdft_ao_basis2 = lib.einsum(  # noqa: F841
            "mi, ij, nj -> mn",
            dummy_ao2eo,
            emb_core_1RDM_for_mcpdft,
            dummy_ao2eo.conj(),
        ).real
        ao2eo_core_emb = self.get_ao2eo([emb_core_orbs])
        emb_core_1RDM_for_mcpdft_ao_basis3 = lib.einsum(  # noqa: F841
            "mi, ij, nj -> mn",
            ao2eo_core_emb[0],
            emb_core_1RDM_for_mcpdft_lo_basis,
            ao2eo_core_emb.conj()[0],
        ).real
        self.is_real(emb_1RDM)
        return emb_core_1RDM_for_mcpdft_ao_basis

    def loc_kpts_to_emb_trial(self, RDM_kpts, emb_orbs, core_orbs, emb_core_orbs):
        """Transform k-space 1-RDM in local basis to embedding basis"""
        lo2eo = lib.einsum("Rk, Rim -> kim", self.phase.conj(), emb_orbs)
        emb_1RDM = lib.einsum("kim, kij, kjn -> mn", lo2eo.conj(), RDM_kpts, lo2eo)
        lo2core = lib.einsum("Rk, Rim -> kim", self.phase.conj(), core_orbs)
        core_1RDM = lib.einsum("kim, kij, kjn -> mn", lo2core.conj(), RDM_kpts, lo2core)  # noqa: F841
        lo2_emb_core = lib.einsum("Rk, Rim -> kim", self.phase.conj(), [emb_core_orbs])
        emb_core_1RDM_for_mcpdft = lib.einsum(
            "kim, kij, kjn -> mn", lo2_emb_core.conj(), RDM_kpts, lo2_emb_core
        )
        emb_core_1RDM_for_mcpdft_lo_basis = lib.einsum(
            "mi, ij, nj -> mn",
            lo2_emb_core[0],
            emb_core_1RDM_for_mcpdft,
            lo2_emb_core[0],
        ).real
        emb_core_1RDM_for_mcpdft_ao_basis = lib.einsum(  # noqa: F841
            "mi, ij, nj -> mn",
            self.ao2lo[0],
            emb_core_1RDM_for_mcpdft_lo_basis,
            self.ao2lo[0],
        ).real
        dummy_ao2eo = lib.einsum("ui, im -> um", self.ao2lo[0], lo2_emb_core[0])
        emb_core_1RDM_for_mcpdft_ao_basis2 = lib.einsum(  # noqa: F841
            "mi, ij, nj -> mn",
            dummy_ao2eo,
            emb_core_1RDM_for_mcpdft_lo_basis,
            dummy_ao2eo.conj(),
        ).real
        ao2eo_core_emb = self.get_ao2eo([emb_core_orbs])
        emb_core_1RDM_for_mcpdft_ao_basis3 = lib.einsum(
            "mi, ij, nj -> mn",
            ao2eo_core_emb[0],
            emb_core_1RDM_for_mcpdft_lo_basis,
            ao2eo_core_emb.conj()[0],
        ).real
        self.is_real(emb_1RDM)
        return emb_core_1RDM_for_mcpdft_ao_basis3

    def loc_kpts_to_core(self, RDM_kpts, core_orbs):
        """Transform k-space 1-RDM in local basis to embedding basis"""
        lo2eo = lib.einsum("Rk, Rim -> kim", self.phase.conj(), core_orbs)
        emb_1RDM = lib.einsum("kim, kij, kjn -> mn", lo2eo.conj(), RDM_kpts, lo2eo)
        self.is_real(emb_1RDM)
        return emb_1RDM.real

    def get_emb_mf_1RDM(self, emb_FOCK, Nelec_in_emb):
        """Get k-space 1-RDM  or derivative 1-RDM in the embedding basis"""
        npairs = Nelec_in_emb // 2
        # eigh returns eigenvalues ascending, so the lowest npairs columns are
        # already the occupied orbitals - no re-sort needed.
        sigma, C = np.linalg.eigh(emb_FOCK)
        Cocc = C[:, :npairs]
        emb_mf_1RDM = 2 * np.dot(Cocc, Cocc.T.conj())
        return emb_mf_1RDM

    def get_emb_guess_1RDM(self, emb_FOCK, Nelec_in_emb, Nimp, chempot):
        """Get guessing 1RDM for the embedding problem"""
        Nemb = emb_FOCK.shape[0]
        npairs = Nelec_in_emb // 2
        chempot_vector = np.zeros(Nemb)
        chempot_vector[:Nimp] = chempot
        emb_FOCK = emb_FOCK - np.diag(chempot_vector)
        #      v lowest npairs columns are occupied.
        sigma, C = np.linalg.eigh(emb_FOCK)
        Cocc = C[:, :npairs]
        DMguess = 2 * np.dot(Cocc, Cocc.T.conj())
        return DMguess

    def get_core_mf_1RDM(self, lo2core, Nelec_in_core, loc_OEH_kpts):
        """Get k-space 1-RDM  or derivative 1-RDM in the embedding basis"""
        npairs = Nelec_in_core // 2
        core_FOCK = lib.einsum("kim,kij,kjn->mn", lo2core.conj(), loc_OEH_kpts, lo2core)
        self.is_real(core_FOCK)
        # eigh returns eigenvalues ascending; lowest npairs columns are occupied.
        sigma, C = np.linalg.eigh(core_FOCK.real)
        Cocc = C[:, :npairs]
        core_mf_1RDM = 2 * np.dot(Cocc, Cocc.T.conj())
        return core_mf_1RDM

    def get_1RDM_Rs(self, loc_1RDM_R0):
        """Construct a R-space 1RDM from the reference cell 1RDM"""
        NRs, nlo = loc_1RDM_R0.shape[:2]  # noqa: F841
        loc_1RDM_kpts = (
            lib.einsum("Rk,Rij,k->kij", self.phase.conj(), loc_1RDM_R0, self.phase[0])
            * self.Nkpts
        )
        loc_1RDM_Rs = self.k_to_R(loc_1RDM_kpts)
        return loc_1RDM_Rs

    def get_phase(self, cell=None, kpts=None, kmesh=None):
        """
        Get a super cell and the phase matrix that transform from real to k-space
        """
        if kmesh is None:
            kmesh = self.kmesh
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts

        a = cell.lattice_vectors()
        Ts = lib.cartesian_prod(
            (np.arange(kmesh[0]), np.arange(kmesh[1]), np.arange(kmesh[2]))
        )
        Rs = np.dot(Ts, a)
        NRs = Rs.shape[0]
        phase = 1 / np.sqrt(NRs) * np.exp(1j * Rs.dot(kpts.T))
        scell = pbctools.super_cell(cell, kmesh)

        return scell, phase

    def get_ao2lo_w90(self, w90):
        """
        Compute the k-space Wannier orbitals
        """
        ao2lo = []
        u_matrix_opt = np.transpose(w90.U_matrix_opt, axes=(2, 1, 0))
        u_matrix = np.transpose(w90.U_matrix, axes=(2, 1, 0))
        for kpt in range(self.Nkpts):
            mo_included = w90.mo_coeff_kpts[kpt][:, w90.band_included_list]
            mo_in_window = w90.lwindow[kpt]
            C_opt = mo_included[:, mo_in_window].dot(
                u_matrix_opt[kpt][:, mo_in_window].T
            )
            ao2lo.append(C_opt.dot(u_matrix[kpt].T))

        ao2lo = np.asarray(ao2lo, dtype=np.complex128)
        return ao2lo

    def get_ao2lo_iao_pao(
        self,
        minao="minao",
        orth_virt=True,
        full_virt=False,
        max_ovlp=True,
        tol=1.0e-10,
        full_return=False,
        mo_coeff_kpts=None,
        mo_occ_kpts=None,
    ):
        """
        Build the k-adapted IAO + PAO transformation C_ao_lo.

        Pass-through wrapper for ``make_iao_pao_kbasis`` — see its docstring
        for the meaning of full_virt / max_ovlp / tol. Defaults match the
        pDMET-recommended call (``orth_virt=True, full_virt=False,
        max_ovlp=True``).
        """
        C_ao_lo, C_val, C_virt, labels = make_iao_pao_kbasis(
            self.cell,
            kmf=self.kmf,
            kpts=self.kpts,
            mo_coeff_kpts=mo_coeff_kpts,
            mo_occ_kpts=mo_occ_kpts,
            minao=minao,
            orth_virt=orth_virt,
            full_virt=full_virt,
            max_ovlp=max_ovlp,
            tol=tol,
        )
        if full_return:
            return C_ao_lo, C_val, C_virt, labels
        return C_ao_lo

    def get_ao2eo(self, emb_orbs):
        """
        Get the transformation matrix from AO to EO
        """
        lo2eo = lib.einsum("Rk, Rim -> kim", self.phase.conj(), emb_orbs)
        ao2eo = lib.einsum("kui, kim -> kum", self.ao2lo, lo2eo)
        return ao2eo

    def get_lo2core(self, core_orbs):
        """
        Get the transformation matrix from AO to the unentangled orbitals
        """
        lo2core = lib.einsum("Rk, Rim -> kim", self.phase.conj(), core_orbs)
        return lo2core

    def get_ao2core(self, core_orbs):
        """
        Get the transformation matrix from AO to the unentangled orbitals
        """
        lo2core = lib.einsum("Rk, Rim -> kim", self.phase.conj(), core_orbs)
        ao2core = lib.einsum("kui, kim -> kum", self.ao2lo, lo2core)
        return ao2core

    def ao_2_loc(self, M_kpts, ao2lo=None):
        """
        Transform an k-space AO integral to local orbitals
        """
        if ao2lo is None:
            ao2lo = self.ao2lo
        return lib.einsum("kui,kuv,kvj->kij", ao2lo.conj(), M_kpts, ao2lo)

    def loc_2_ao(self, M_kpts, ao2lo=None):
        """
        Transform an k-space local integral to ao orbitals
        """
        if ao2lo is None:
            ao2lo = self.ao2lo
        return lib.einsum("kui,kij,kvj->kuv", ao2lo, M_kpts, ao2lo.conj())

    def k_to_R(self, M_kpts):
        """Transform AO or LO integral/1-RDM in k-space to R-space"""
        NRs, Nkpts = self.phase.shape
        nao = M_kpts.shape[-1]
        M_Rs = lib.einsum("Rk,kuv,Sk->RuSv", self.phase, M_kpts, self.phase.conj())
        M_Rs = M_Rs.reshape(NRs * nao, NRs * nao)
        self.is_real(M_Rs)
        return M_Rs.real

    def k_to_R0(self, M_kpts):
        """Transform AO or LO integral/1-RDM in k-space to the reference unit cell
        M(k) -> M(0,R) with index Ruv
        """
        NRs, Nkpts = self.phase.shape  # noqa: F841
        nao = M_kpts.shape[-1]  # noqa: F841
        M_R0 = lib.einsum("Rk,kuv,k->Ruv", self.phase, M_kpts, self.phase[0].conj())
        self.is_real(M_R0)
        return M_R0.real

    def R_to_k(self, M_Rs):
        """Transform AO or LO integral/1-RDM in R-space to k-space"""
        NRs, Nkpts = self.phase.shape
        nao = M_Rs.shape[0] // NRs
        M_Rs = M_Rs.reshape(NRs, nao, NRs, nao)
        M_kpts = lib.einsum("Rk,RuSv,Sk->kuv", self.phase.conj(), M_Rs, self.phase)
        return M_kpts

    def R0_to_k(self, M_R0):
        """Transform AO or LO integral/1-RDM in R-space to k-space"""
        NRs, nao = M_R0.shape[:2]
        M_kpts = lib.einsum("Rk,Ruv,k->kuv", self.phase.conj(), M_R0, self.phase[0])
        return M_kpts * NRs

    def is_real(self, M, threshold=1.0e-6):
        """Check if a matrix is real with a threshold"""
        assert abs(M.imag).max() < threshold, "The imaginary part is larger than %s" % (
            str(threshold)
        )
