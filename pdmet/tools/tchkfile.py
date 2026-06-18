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

import numpy as np
from pyscf.lib.chkfile import save, load
from pdmet.tools.optional import to_numpy, require_cupy, is_gpu_mf, is_krohf, _bridge


def _fix_empty(obj):
    if obj is None:
        return np.array([], dtype=float)
    if isinstance(obj, (list, tuple)) and len(obj) == 0:
        return np.array([], dtype=float)
    if isinstance(obj, dict) and len(obj) == 0:
        return np.array([], dtype=float)
    return obj


def symmetrize_kmf(cell, kmf, kmesh):
    """
    This function creates an equivalent Brillouin zone of the kmesh using in kmf calculation
    and also makes sure: C(k) = C(-k)_{*} with C is the orbital coefficient
    why?
    Due to the fact that, the eigenvectors at k C(k) is not equal to the C(-k)_{*}.
    That means the kmesh symmetry is not strictly employed in PBC-PySCF.
    This is not a problem for a HF or KS-DFT calculation, howevere, it is a big trouble for
    calculation that requires this strict symmetry, for example, FFT of crystalline orbitals to get Wannier functions
    in Wannier90 calculations.

    Attributes: kmf can be either a real kmf wavefunction or a saved one from the save_kmf function.

    TODO: 1/ To enforce k-point symmetry in PBC-PySCF. 2/ what about an even number of k point
    """

    kmf.kpts = cell.make_kpts(kmesh, wrap_around=True)
    for i in range(kmf.kpts.shape[0] - 1):
        for j in range(i + 1):
            if abs(kmf.kpts[i + 1] + kmf.kpts[j]).sum() < 1.0e-10:
                kmf.mo_coeff_kpts[i + 1] = kmf.mo_coeff_kpts[j].conj()
                kmf.mo_energy_kpts[i + 1] = kmf.mo_energy_kpts[j]
                break

    return kmf


def save_kmf(kmf, chkfile, gpu=None):
    """Save a converged SCF object to chkfile.

    Args:
        kmf: SCF mean-field (pyscf or gpu4pyscf)
        chkfile: filename to save the kmf object
        gpu:    None (default) auto-detects from kmf's class module
                True forces cupy -> numpy conversion before write
                False skips the conversion
    Notes:
        * Computing get_fock triggers one JK with GPU is fast
            with CPU GDF on large cells it's expensive.
        * The chkfile only stores numpy arrays
    """
    if gpu is None:
        gpu = is_gpu_mf(kmf)
    if gpu:
        require_cupy("save_kmf(..., gpu=True)")
        _np = to_numpy
    else:

        def _np(x):
            return x

    exxdiv = "None" if kmf.exxdiv is None else kmf.exxdiv
    max_memory = kmf.max_memory
    e_tot = kmf.e_tot
    kpts = np.asarray(_np(kmf.kpts))

    mo_occ_kpts = _np(kmf.mo_occ_kpts)
    mo_energy_kpts = _np(kmf.mo_energy_kpts)
    mo_coeff_kpts = _np(kmf.mo_coeff_kpts)

    # Call JK build once
    dm = kmf.make_rdm1()
    s1e = kmf.get_ovlp()
    fock = kmf.get_fock(s1e=s1e, dm=dm)

    kmf_dict = {
        "exxdiv": exxdiv,
        "max_memory": max_memory,
        "e_tot": e_tot,
        "kpts": kpts,
        "mo_occ_kpts": mo_occ_kpts,
        "mo_energy_kpts": mo_energy_kpts,
        "mo_coeff_kpts": mo_coeff_kpts,
        "get_fock": _np(fock),
        "make_rdm1": _np(dm),
    }

    save(chkfile, "scf", kmf_dict)


def load_kmf(kmf, chkfile, max_memory=4000, gpu=None):
    """
    Load a kmf object
    """
    if gpu is None:
        gpu = is_gpu_mf(kmf)
    if gpu:
        require_cupy("load_kmf(..., gpu=True)")

    saved = load(chkfile, "scf")

    class fake_kmf:
        def __init__(self, saved):
            ex = saved["exxdiv"]
            if isinstance(ex, bytes):
                ex = ex.decode()
            self.exxdiv = None if ex == "None" else ex  # wrapper: original
            kmf.exxdiv = None  # real kmf: bare J/K
            self._is_ROHF = is_krohf(kmf)

            # Stored data
            self.e_tot = saved["e_tot"]
            self.kpts = saved["kpts"]
            self.mo_occ_kpts = saved["mo_occ_kpts"]
            self.mo_energy_kpts = saved["mo_energy_kpts"]
            self.mo_coeff_kpts = saved["mo_coeff_kpts"]

            _fock = saved["get_fock"]
            _dm = saved["make_rdm1"]
            self.get_fock = lambda *a, **kw: _fock
            self.make_rdm1 = lambda *a, **kw: _dm

            self.eig = _bridge(kmf.eig, gpu)
            self.get_ovlp = _bridge(kmf.get_ovlp, gpu)
            self.get_hcore = _bridge(kmf.get_hcore, gpu)
            self.get_j = _bridge(kmf.get_j, gpu)
            self.get_k = _bridge(kmf.get_k, gpu)
            self.get_jk = _bridge(kmf.get_jk, gpu)
            self.get_veff = _bridge(kmf.get_veff, gpu)

            _bands = _bridge(kmf.get_bands, gpu)

            def get_bands(*a, **kw):
                if "dm_kpts" not in kw:
                    kw["dm_kpts"] = self.make_rdm1()
                return _bands(*a, **kw)

            self.get_bands = get_bands

            self.max_memory = getattr(kmf, "max_memory", max_memory)
            self.with_df = kmf.with_df

    final_kmf = fake_kmf(saved)

    return final_kmf


def save_w90(w90, chkfile):
    mp_grid_loc = w90.mp_grid_loc
    exclude_bands = w90.exclude_bands
    mp_grid_loc = w90.mp_grid_loc
    mo_coeff_kpts = w90.mo_coeff_kpts
    band_included_list = w90.band_included_list
    lwindow = w90.lwindow
    M_matrix_loc = w90.M_matrix_loc
    A_matrix_loc = w90.A_matrix_loc
    eigenvalues_loc = w90.eigenvalues_loc
    U_matrix_opt = w90.U_matrix_opt
    U_matrix = w90.U_matrix
    wann_centres = w90.wann_centres
    wann_spreads = w90.wann_spreads
    spread = w90.spread

    w90_dic = {
        "mp_grid_loc": mp_grid_loc,
        "exclude_bands": exclude_bands,
        "mo_coeff_kpts": mo_coeff_kpts,
        "band_included_list": band_included_list,
        "lwindow": lwindow,
        "U_matrix_opt": U_matrix_opt,
        "U_matrix": U_matrix,
        "M_matrix_loc": M_matrix_loc,
        "A_matrix_loc": A_matrix_loc,
        "eigenvalues_loc": eigenvalues_loc,
        "wann_centres": wann_centres,
        "wann_spreads": wann_spreads,
        "spread": spread,
    }
    w90_dic = {k: _fix_empty(v) for k, v in w90_dic.items()}
    save(chkfile, "w90", w90_dic)


def load_w90(w90, chkfile):
    save_w90 = load(chkfile, "w90")
    w90.mp_grid_loc = save_w90["mp_grid_loc"]
    w90.exclude_bands = save_w90["exclude_bands"]
    w90.mp_grid_loc = save_w90["mp_grid_loc"]
    w90.mo_coeff_kpts = save_w90["mo_coeff_kpts"]
    w90.band_included_list = save_w90["band_included_list"]
    w90.lwindow = save_w90["lwindow"]
    w90.U_matrix_opt = save_w90["U_matrix_opt"]
    w90.U_matrix = save_w90["U_matrix"]
    w90.M_matrix_loc = save_w90["M_matrix_loc"]
    w90.A_matrix_loc = save_w90["A_matrix_loc"]
    w90.eigenvalues_loc = save_w90["eigenvalues_loc"]
    w90.wann_centres = save_w90["wann_centres"]
    w90.wann_spreads = save_w90["wann_spreads"]
    w90.spread = save_w90["spread"]

    return w90


def _to_str(x):
    """h5py >= 3.0 returns variable-length strings as bytes; canonicalize."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode()
    return x


def save_lo_iao(local, chkfile):
    ao2lo = local.ao2lo
    lo_labels = local.lo_labels

    local_dic = {
        "ao2lo": ao2lo,
        "lo_labels": lo_labels,
        "minao": local.minao,
    }
    local_dic = {k: _fix_empty(v) for k, v in local_dic.items()}
    save(chkfile, "local", local_dic)


def load_lo_iao(chkfile):
    save_local = load(chkfile, "local")
    raw_labels = save_local["lo_labels"]
    return {
        "ao2lo": save_local["ao2lo"],
        "lo_labels": (
            None if raw_labels is None else [_to_str(label) for label in raw_labels]
        ),
        "minao": _to_str(save_local["minao"]),
    }


def save_pdmet(pdmet, chkfile):
    solver = pdmet.solver
    chempot = pdmet.chempot
    uvec = pdmet.uvec
    umat = pdmet.umat
    emb_orbs = pdmet.emb_orbs
    emb_core_orbs = pdmet.emb_core_orbs
    mf_mo = pdmet.qcsolver.mf.mo_coeff
    actv1RDMloc = pdmet.emb_corr_1RDM

    if pdmet.solver in ["CASCI", "CASSCF", "DMRG-CI", "DMRG-SCF"]:
        mc_mo = pdmet.qcsolver.mo
        mc_mo_nat = pdmet.qcsolver.mo_nat

    pdmet_dic = {
        "solver": solver,
        "chempot": chempot,
        "uvec": uvec,
        "umat": umat,
        "emb_orbs": emb_orbs,
        "emb_core_orbs": emb_core_orbs,
        "mf_mo": mf_mo,
        "actv1RDMloc": actv1RDMloc,
    }

    if pdmet.solver in ["CASCI", "CASSCF", "DMRG-CI", "DMRG-SCF"]:
        pdmet_dic["mc_mo"] = mc_mo
        pdmet_dic["mc_mo_nat"] = mc_mo_nat

    save(chkfile, "pdmet", pdmet_dic)


def load_pdmet(chkfile):
    save_pdmet = load(chkfile, "pdmet")

    class fake_pdmet:
        def __init__(self, save_pdmet):
            self.solver = None
            self.chempot = 0
            self.uvec = False
            self.umat = False
            self.emb_orbs = None
            self.emb_core_orbs = None
            self.mf_mo = None
            self.mc_mo = None
            self.mc_mo_nat = None
            if save_pdmet is not None:
                self.solver = save_pdmet["solver"]
                self.chempot = save_pdmet["chempot"]
                self.uvec = save_pdmet["uvec"]
                self.umat = save_pdmet["umat"]
                self.emb_orbs = save_pdmet["emb_orbs"]
                self.emb_core_orbs = save_pdmet["emb_core_orbs"]
                self.mf_mo = save_pdmet["mf_mo"]
                self.actv1RDMloc = save_pdmet["actv1RDMloc"]
                if self.solver in ["CASCI", "CASSCF", "DMRG-CI", "DMRG-SCF"]:
                    self.mc_mo = save_pdmet["mc_mo"]
                    self.mc_mo_nat = save_pdmet["mc_mo_nat"]

    pdmet = fake_pdmet(save_pdmet)

    return pdmet
