"""
NiO sanity test for the k-adapted IAO + PAO builder.
"""

import os
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.tools import cubegen

from pdmet.localbasis import make_iao_pao_kbasis


def build_nio_cell(spin=0):
    """Rocksalt NiO, primitive cell (Ni at 0,0,0; O at 1/2,1/2,1/2)."""
    a = 4.17
    cell = gto.Cell()
    cell.atom = [
        ["Ni", (0.0, 0.0, 0.0)],
        ["O", (0.5 * a, 0.5 * a, 0.5 * a)],
    ]
    cell.a = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * a
    cell.basis = {"Ni": "gth-dzvp-molopt-sr", "O": "gth-dzvp"}
    cell.pseudo = "gth-pade"
    cell.spin = spin
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()
    return cell


def run_checks(label, cell, kmf, C_ao_lo, C_val, C_virt):
    kpts = kmf.kpts
    nkpts = len(kpts)
    nao = cell.nao_nr()

    # KRHF / KROHF return DIFFERENT shapes from make_rdm1():
    #   KRHF  -> (Nk, nao, nao)              total density
    #   KROHF -> (2, Nk, nao, nao)           alpha and beta
    is_rohf = isinstance(kmf, scf.krohf.KROHF) or getattr(kmf, "_is_ROHF", False)

    S = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))

    # orthonormality
    err1 = max(
        np.abs(
            C_ao_lo[k].conj().T @ S[k] @ C_ao_lo[k] - np.eye(C_ao_lo.shape[-1])
        ).max()
        for k in range(nkpts)
    )

    # IAO occupied MO span:  P_iao psi == psi
    # KROHF stores ONE shared MO set per k with mo_occ ∈ {0,1,2}, so this
    # branch is spin-agnostic — `> 0` grabs singly+doubly occupied at once.
    err2 = 0.0
    for k in range(nkpts):
        occ = kmf.mo_coeff_kpts[k][:, np.asarray(kmf.mo_occ_kpts[k]) > 0]
        if occ.size == 0:
            continue
        P = C_val[k] @ C_val[k].conj().T @ S[k]
        err2 = max(err2, np.abs(P @ occ - occ).max())

    # electron count survives the AO -> LO transform.
    # For ROHF, collapse the spin axis into the total density up-front.
    dm_ao = np.asarray(kmf.make_rdm1())
    if is_rohf:
        dm_ao = dm_ao[0] + dm_ao[1]  # α + β = total density per k

    n_ao = sum(np.einsum("ij,ji->", dm_ao[k], S[k]) for k in range(nkpts)).real / nkpts
    n_lo = 0.0
    for k in range(nkpts):
        D_lo_k = C_ao_lo[k].conj().T @ S[k] @ dm_ao[k] @ S[k] @ C_ao_lo[k]
        n_lo += np.trace(D_lo_k).real
    n_lo /= nkpts

    nlo = C_ao_lo.shape[-1]
    nval = C_val.shape[-1]
    nvirt = C_virt.shape[-1]

    # (e) Completeness: C_ao_lo spans EVERY MO (occupied and virtual)
    # Define the MO-to-LO unitary at each k:
    #     U(k) = C_ao_lo(k)^H · S(k) · C_mo(k)        shape (nao, nmo)
    # Then |psi_m^k> = sum_i U_im(k) |w_i^k>. If C_ao_lo really is a complete
    # orthonormal basis of the AO space, U is unitary: U^H U = I_nmo
    err5 = 0.0
    for k in range(nkpts):
        U = C_ao_lo[k].conj().T @ S[k] @ kmf.mo_coeff_kpts[k]
        err5 = max(err5, np.abs(U.conj().T @ U - np.eye(U.shape[1])).max())

    print(f"  [{label}] orth err            = {err1:.2e}")
    print(f"  [{label}] |P_iao psi - psi|   = {err2:.2e}")
    print(f"  [{label}] N_e ao={n_ao:.4f}, lo={n_lo:.4f}")
    print(f"  [{label}] nao={nao}, nlo={nlo}, nval={nval}, nvirt={nvirt}")
    print(f"  [{label}] |U^H U - I| (MO->LO) = {err5:.2e}")

    assert err1 < 1e-9, f"{label}: orthonormality broke"
    assert err2 < 1e-9, f"{label}: IAO doesn't span occupied"
    assert abs(n_ao - n_lo) < 1e-8, f"{label}: electron count drifted"
    assert nval + nvirt == nao, f"{label}: dimension mismatch"
    assert err5 < 1e-9, f"{label}: C_ao_lo doesn't span all MOs"


def _real_phase(coeff):
    j = np.argmax(np.abs(coeff))
    phase = np.exp(-1j * np.angle(coeff[j]))
    return (coeff * phase).real


def dump_cubes(label, cell, C_ao_lo, labels, picks, outdir, nx=40, ny=40, nz=40):
    """Write one cube per index in `picks`. Pass picks=range(nao) for all."""
    os.makedirs(outdir, exist_ok=True)
    for i in picks:
        coeff = _real_phase(C_ao_lo[0, :, i])
        # AO label into a filename
        tag = labels[i].strip().replace(" ", "_")
        fname = os.path.join(outdir, f"{label}_lo{i:02d}_{tag}.cube")
        cubegen.orbital(cell, fname, coeff, nx=nx, ny=ny, nz=nz)
        print(f"  [{label}] wrote {fname}")


def run_one(spin, label, outdir):
    print(f"\n=== NiO  {label}  (spin={spin}) ===")
    cell = build_nio_cell(spin=spin)

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)

    if spin == 0:
        kmf = scf.KRHF(cell, kpts).density_fit()
    else:
        kmf = scf.KROHF(cell, kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 50
    kmf.kernel()
    print(f"  SCF energy = {kmf.e_tot:.6f}")

    C_ao_lo, C_val, C_virt, lo_labels = make_iao_pao_kbasis(
        cell, kmf=kmf, minao="gth-szv-molopt-sr"
    )

    run_checks(label, cell, kmf, C_ao_lo, C_val, C_virt)

    nao = C_ao_lo.shape[-1]
    nval = C_val.shape[-1]
    nvirt = C_virt.shape[-1]

    print(f"\n  [{label}] full LO partition  (nao = {nao})")
    print(f"  [{label}]   IAO valence ({nval}):")
    for i in range(nval):
        print(f"  [{label}]     {i:3d}  {lo_labels[i]}")
    print(f"  [{label}]   PAO virtual ({nvirt}):")
    for i in range(nval, nao):
        print(f"  [{label}]     {i:3d}  {lo_labels[i]}")

    dump_cubes(label, cell, C_ao_lo, lo_labels, range(nao), outdir)
    return kmf.e_tot


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "iao_pao_cubes")

    e_rhf = run_one(spin=0, label="rhf", outdir=outdir)
    # e_rohf = run_one(spin=2, label="rohf", outdir=outdir)

    print("\nSummary")
    print(f"  RHF  E = {e_rhf:.6f}")
    # print(f"  ROHF E = {e_rohf:.6f}")
    print(f"\nCubes written to: {outdir}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
