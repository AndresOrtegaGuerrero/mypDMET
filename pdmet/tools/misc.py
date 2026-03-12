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
