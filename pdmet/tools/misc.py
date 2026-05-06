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

import pyscf.data.nist as param

BOHR = param.BOHR


def make_imp_orbs(cell, w90, impCluster, threshold=0.5, rm_list=None, add_list=None):
    """
        Identify the impurity orbitals based on the distance between the MLWF centers and the impurity atoms.

    Parameters
    ----------
    cell        : PySCF cell
    w90         : pyWannier90 object (w90.wann_centres in Angstrom)
    impCluster  : list[int]   atom indices (1-based, PySCF convention)
    threshold   : float       distance cutoff in Angstrom (default 0.5)
    rm_list     : list[int]   1-based WF indices to force-exclude
    add_list    : list[int]   1-based WF indices to force-include
    verbose     : bool        print WF→atom assignments

    Return:
        impOrbs : ndarray
                (n_wann) array with 1 for impurity WFs
        impAtms : list[list[int]]
                Wannier indices grouped by impurity atom

    """
    impCluster = np.asarray(impCluster)

    assert impCluster.max() <= cell.natm, (
        f"impCluster contains atom index > natm ({cell.natm}): {impCluster}"
    )

    # Convert Lattice to Angstrom
    lattice = cell.lattice_vectors() * BOHR
    inv_lattice = np.linalg.inv(lattice)
    atom_cart = cell.atom_coords() * BOHR

    # Wannier in fractional coordinates
    wf_frac = (w90.wann_centres @ inv_lattice) % 1.0  # Ensure within [0, 1)
    # Imputity atoms in fractional coordinates
    imp_frac = (atom_cart[impCluster - 1] @ inv_lattice) % 1.0

    # minium image distance
    delta = wf_frac[:, None, :] - imp_frac[None, :, :]
    delta -= np.round(delta)
    dist = np.linalg.norm(delta @ lattice, axis=2)

    min_dist = dist.min(axis=1)
    min_idx = np.argmin(dist, axis=1)

    # remove / add lists
    if rm_list is not None:
        min_dist[np.asarray(rm_list, dtype=int) - 1] = np.inf
    if add_list is not None:
        min_dist[np.asarray(add_list, dtype=int) - 1] = 0.0

    #  Impurity mask
    impOrbs = (min_dist < threshold).astype(np.int32)
    imp_indices = np.where(impOrbs == 1)[0]

    for wf in imp_indices:
        atm = impCluster[min_idx[wf]]
        print(f"WF {wf:4d}  -> atom {atm:3d}   dist = {min_dist[wf]:.3f} Å")
    impAtms = [imp_indices[min_idx[imp_indices] == i] for i in range(len(impCluster))]

    return impOrbs, impAtms


def make_imp_orbs_from_labels(
    cell, lo_labels, impCluster, orbital_filter=None, rm_list=None, add_list=None
):
    """
    Pick impurity LOs by matching IAO/PAO labels.

    lo_labels[i] is in PySCF convention: "0 C 2pz", "3 Ni 3dz^2", ...
    impCluster   : list[int]   1-based atom indices to take orbitals from.
    orbital_filter: dict       see EmbeddingSettings.imp_orbital_filter.
                               None means "all orbitals on these atoms".
    """
    cluster0 = np.asarray(impCluster) - 1  # to 0-based
    impOrbs = np.zeros(len(lo_labels), dtype=np.int32)
    impAtms = [[] for _ in cluster0]

    def _shell_ok(atm_idx, sym, shell_tag):
        if orbital_filter is None:
            return True
        # Try atom-index key first, then element-symbol key
        allowed = orbital_filter.get(atm_idx + 1, orbital_filter.get(sym, None))
        if allowed is None:
            return True  # unfiltered atoms keep all
        return any(shell_tag.startswith(s) for s in allowed)

    for i, lbl in enumerate(lo_labels):
        toks = lbl.split()  # e.g. ["3", "Ni", "3dz^2"]
        atm_idx = int(toks[0])
        sym = toks[1]
        shell = toks[2]  # "3dz^2", "2pz", "1s", ...
        cluster_pos = {atm: j for j, atm in enumerate(cluster0)}
        if atm_idx in cluster_pos and _shell_ok(atm_idx, sym, shell):
            j = cluster_pos[atm_idx]
            impOrbs[i] = 1
            impAtms[j].append(i)

    if rm_list is not None:
        impOrbs[np.asarray(rm_list) - 1] = 0
    if add_list is not None:
        impOrbs[np.asarray(add_list) - 1] = 1
    return impOrbs, [np.array(a) for a in impAtms]
