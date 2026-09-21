"""NTO tests: numpy-only SVD core + invariants, settings validation, and an
IAO+PAO H4 pDMET integration (no Wannier90).

    pytest tests/test_nto.py -v               # all
    pytest tests/test_nto.py -v -m "not slow" # skip the heavy DMET run
"""

import os
import numpy as np
import pytest

pytest.importorskip("pyscf")  # whole stack imports pyscf at load

from pdmet.qcsolvers.casbase import BaseCASSolver
from pdmet.settings import SolverSettings


# --------------------------------------------------------------------------- #
#  1. Pure-math core: SVD convention and invariants                           #
# --------------------------------------------------------------------------- #


def test_rank1_transition_recovers_single_pair():
    """rank-1 T = |acceptor><donor| -> one lambda==1, donor/acceptor recovered."""
    rng = np.random.default_rng(0)
    donor = rng.standard_normal(6)
    donor /= np.linalg.norm(donor)
    acceptor = rng.standard_normal(6)
    acceptor /= np.linalg.norm(acceptor)

    T = np.outer(acceptor, donor)  # T_pq = acceptor_p * donor_q

    lam, V_hole, U_part = BaseCASSolver._decompose_nto_cas(T)

    assert lam.shape == (1,), "rank-1 T must give exactly one pair"
    assert abs(lam[0] - 1.0) < 1e-12, "single normalized route must weigh 1"

    # parallel up to sign => |cos| == 1
    assert abs(abs(V_hole[:, 0] @ donor) - 1.0) < 1e-10
    assert abs(abs(U_part[:, 0] @ acceptor) - 1.0) < 1e-10


def test_lambdas_sorted_nonneg_and_frobenius_identity():
    """lambda sorted desc, >= 0, and sum(lambda) == ||T||_F**2 (holds for any T)."""
    rng = np.random.default_rng(42)
    T = rng.standard_normal((7, 7))

    lam, V_hole, U_part = BaseCASSolver._decompose_nto_cas(T)

    assert (np.diff(lam) <= 1e-12).all(), "lambdas not sorted descending"
    assert (lam >= 0).all(), "negative lambda (sigma**2 cannot be negative)"
    assert abs(lam.sum() - np.linalg.norm(T, "fro") ** 2) < 1e-10
    # shape contract: columns are orbitals, one per kept singular value
    assert V_hole.shape == (7, len(lam))
    assert U_part.shape == (7, len(lam))


def test_svd_reconstructs_T():
    """U diag(sqrt(lambda)) V_hole^dagger rebuilds T -- guards a transpose bug."""
    rng = np.random.default_rng(7)
    T = rng.standard_normal((5, 5))
    lam, V_hole, U_part = BaseCASSolver._decompose_nto_cas(T)
    T_rebuilt = U_part @ np.diag(np.sqrt(lam)) @ V_hole.conj().T
    assert np.allclose(T_rebuilt, T, atol=1e-10)


def test_fix_orbital_phases_is_deterministic():
    """Largest hole entry positive; a joint (V,U) -> (-V,-U) flip is undone."""
    rng = np.random.default_rng(1)
    V = rng.standard_normal((6, 3))
    U = rng.standard_normal((6, 3))

    Vf, Uf = BaseCASSolver._fix_orbital_phases(V, U)
    for j in range(V.shape[1]):
        assert Vf[np.argmax(np.abs(Vf[:, j])), j] > 0

    # idempotent; flipping a pair together leaves the result unchanged
    Vf2, Uf2 = BaseCASSolver._fix_orbital_phases(Vf, Uf)
    assert np.allclose(Vf, Vf2) and np.allclose(Uf, Uf2)
    Vf3, Uf3 = BaseCASSolver._fix_orbital_phases(-V, -U)
    assert np.allclose(Vf, Vf3) and np.allclose(Uf, Uf3)


def test_fix_orbital_phases_preserves_reconstruction():
    """After the phase fix, U diag(sqrt(lam)) V^H must still rebuild T."""
    rng = np.random.default_rng(3)
    T = rng.standard_normal((5, 5))
    lam, V, U = BaseCASSolver._decompose_nto_cas(T)
    Vf, Uf = BaseCASSolver._fix_orbital_phases(V, U)
    assert np.allclose(Uf @ np.diag(np.sqrt(lam)) @ Vf.conj().T, T, atol=1e-10)


def test_numerical_zero_pairs_are_dropped():
    """sigma**2 <= thresh pairs are pruned as floating-point noise."""
    T = np.diag([1.0, 0.5, 1e-15, 0.0])
    lam, V_hole, U_part = BaseCASSolver._decompose_nto_cas(T, thresh=1e-12)
    assert len(lam) == 2, "only the two real singular values survive"
    assert np.allclose(np.sort(lam)[::-1], [1.0, 0.25])


# --------------------------------------------------------------------------- #
#  2. Settings validation                                                     #
# --------------------------------------------------------------------------- #


def test_nto_requires_multiroot():
    # a single-root run has no transition to decompose
    with pytest.raises(ValueError, match="multi-root"):
        SolverSettings(nto=True).validate()
    with pytest.raises(ValueError, match="multi-root"):
        SolverSettings(nto_export=True).validate()


def test_nto_flag_ok_without_nevpt2():
    # NTOs decoupled from NEVPT2: a multi-root run + flag is enough
    SolverSettings(nto=True, nroots=2).validate()  # must not raise
    SolverSettings(nto=True, state_average_=[0.5, 0.5], nroots=2).validate()


def test_nto_npairs_must_be_positive():
    s = SolverSettings(nto=True, nroots=2, nto_npairs=0)
    with pytest.raises(ValueError, match="nto_npairs"):
        s.validate()


def test_nto_lambda_floor_nonnegative():
    s = SolverSettings(nto=True, nroots=2, nto_lambda_floor=-1.0)
    with pytest.raises(ValueError, match="nto_lambda_floor"):
        s.validate()


def test_valid_nto_settings_pass():
    # still valid via the NEVPT2 route
    SolverSettings(
        nto_export=True, nevpt2_roots=[0, 1], nto_npairs=2, nto_lambda_floor=1e-3
    ).validate()


# --------------------------------------------------------------------------- #
#  3. IAO+PAO integration (H4, Gamma-only)                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def h4_pdmet(tmp_path_factory):
    """H4 pDMET, CASSCF + NEVPT2, IAO+PAO. cas=(4,4) over the whole H4 hosts a
    clean rank-1 single excitation; e_shift keeps singlets; 4 roots so it's in
    range. The strong-transition root is discovered (see `excited_state`).
    """
    from pyscf.pbc import gto, scf, df
    from pdmet import dmet

    work = tmp_path_factory.mktemp("h4_nto")
    gdf_file = os.path.join(work, "gdf.h5")

    cell = gto.Cell()
    cell.atom = """
    H 1.0 1.0 1.0
    H 1.0 1.0 2.0
    H 2.0 1.0 1.0
    H 2.0 1.0 2.0
    """
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.a = np.eye(3) * 10
    cell.spin = 0
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    if not os.path.exists(gdf_file):
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_file
        gdf.build()

    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.with_df._cderi = gdf_file
    kmf.exxdiv = None
    kmf.run()

    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="CASSCF")
    pdmet_obj.lobasis.minao = {"H": "gth-szv"}
    pdmet_obj.emb.impCluster = [1, 2, 3, 4]  # whole H4 as impurity
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (4, 4)
    pdmet_obj.solver.e_shift = 0.2  # spin penalty -> keep singlets
    # ground (0) + three excited roots on the ground-optimized orbitals
    pdmet_obj.solver.nevpt2_roots = [0, 1, 2, 3]
    pdmet_obj.solver.nevpt2_nroots = 4
    pdmet_obj.solver.nto = True  # explicit-flag contract: NEVPT2 no longer implies NTOs
    pdmet_obj.solver.nroots = 1
    pdmet_obj.initialize()
    pdmet_obj.one_shot()
    return pdmet_obj


def _strongest_excited_state(pdmet_obj, floor=1e-3):
    """(state, top_lambda) of the excited state with the largest dominant pair
    (state 0 is the ground density); (None, ..) if nothing clears the floor.
    """
    best, best_lam = None, 0.0
    for state in range(1, len(pdmet_obj.ntos_per_root)):
        lam = pdmet_obj.ntos_per_root[state]["lambdas"]
        top = float(lam[0]) if len(lam) else 0.0
        if top > best_lam:
            best, best_lam = state, top
    return (best, best_lam) if best_lam >= floor else (None, best_lam)


@pytest.fixture(scope="module")
def excited_state(h4_pdmet):
    """Index of the state carrying a real (visualizable) single excitation."""
    state, top = _strongest_excited_state(h4_pdmet)
    if state is None:
        pytest.skip(
            f"no excited state with a dominant NTO pair above floor "
            f"(strongest top-lambda={top:.2e}); tune cas / e_shift / nroots."
        )
    return state


@pytest.mark.slow
def test_ntos_per_root_populated(h4_pdmet):
    assert h4_pdmet.ntos_per_root is not None, "solver did not emit NTOs"
    assert len(h4_pdmet.ntos_per_root) == 4, "expected one entry per NEVPT2 root"
    for info in h4_pdmet.ntos_per_root:
        assert {"lambdas", "V_hole", "U_part"} <= set(info)


@pytest.mark.slow
def test_integration_invariants_eo_basis(h4_pdmet):
    """In the embedding basis: lambdas sorted/nonneg, shapes consistent, and
    sum(lambda) == ||t_dm1_emb||_F**2 (isometry invariance of singular values).
    """
    Norb = h4_pdmet.emb_orbs.shape[-1]
    for state, info in enumerate(h4_pdmet.ntos_per_root):
        lam = info["lambdas"]
        k = len(lam)
        assert (np.diff(lam) <= 1e-10).all()
        assert (lam >= 0).all()
        assert info["V_hole"].shape == (Norb, k)
        assert info["U_part"].shape == (Norb, k)

        t_emb = h4_pdmet.t_dm1s[state]
        assert abs(lam.sum() - np.linalg.norm(t_emb, "fro") ** 2) < 1e-8


@pytest.mark.slow
def test_strong_single_excitation_exists(h4_pdmet):
    """A clean single excitation shows up as a small (near-degenerate) block of
    pairs carrying ~all of sum(lambda) -- not necessarily one pair, since the
    total (a+b) density can give 2 equal lambdas for one spatial excitation.
    """
    state, top = _strongest_excited_state(h4_pdmet)
    assert state is not None, "no excited state cleared the lambda floor"
    lam = h4_pdmet.ntos_per_root[state]["lambdas"]

    # the dominant block = pairs within ~10% of the top weight (alpha/beta twins)
    block = lam[lam > 0.9 * lam[0]]
    assert block.sum() / lam.sum() > 0.9, (
        f"expected a small degenerate block to dominate; got lambdas={lam}"
    )
    # closed-shell total density: that block is (near-)2-fold degenerate
    assert len(block) in (1, 2), f"unexpected block size {len(block)}: {lam}"


@pytest.mark.slow
def test_get_ntos_returns_pairs_and_prints_table(h4_pdmet, excited_state, capsys):
    results = h4_pdmet.get_ntos(state=excited_state, n_pairs=2, outdir=None)
    out = capsys.readouterr().out
    assert f"NTOs for state {excited_state}" in out
    assert len(results) >= 1
    for entry in results:
        assert {"pair", "lambda", "donor", "acceptor"} <= set(entry)
        assert entry["donor"].shape == (h4_pdmet.emb_orbs.shape[-1],)


@pytest.mark.slow
def test_gamma_cube_export_writes_files(h4_pdmet, excited_state, tmp_path):
    outdir = os.path.join(tmp_path, "ntos")
    # n_pairs=2 captures both spin "lanes" of the degenerate single excitation
    results = h4_pdmet.get_ntos(
        state=excited_state,
        n_pairs=2,
        lambda_floor=1e-6,
        outdir=outdir,
        grid=(20, 20, 20),
    )
    assert len(results) >= 1, "discovered state should yield >=1 pair"
    files = os.listdir(outdir)
    assert any("donor" in f for f in files), files
    assert any("acceptor" in f for f in files), files
    # no leftover temp files from the rename step
    assert not any(f.startswith("_nto_state") for f in files), files


@pytest.mark.slow
def test_kpts_guard_blocks_cube_export(h4_pdmet, excited_state, tmp_path, monkeypatch):
    """k>1: cubes refuse (on outdir alone), table still works. Nkpts is patched
    so the guard fires without running a real k-mesh.
    """
    # table-only path works regardless of k-mesh
    assert h4_pdmet.get_ntos(state=excited_state, outdir=None) is not None

    monkeypatch.setattr(h4_pdmet.local, "Nkpts", 2, raising=False)
    with pytest.raises(NotImplementedError, match="Gamma-only|Nkpts"):
        h4_pdmet.get_ntos(
            state=excited_state, outdir=os.path.join(tmp_path, "should_not_write")
        )


# --------------------------------------------------------------------------- #
#  4. Periodic H2 chain (Gamma): the textbook sigma_g -> sigma_u NTO           #
# --------------------------------------------------------------------------- #
# Rank-1 excitation: hole = bonding sigma_g (occupied), particle = sigma_u
# (virtual). lambda_0 ~= 2 (total a+b density); rank-1 means lambda_0/sum ~= 1.


def _eo_overlap(u, v):
    """Inner product of two embedding-basis vectors (EO basis is orthonormal)."""
    return abs(complex(np.vdot(u, v))) / (np.linalg.norm(u) * np.linalg.norm(v))


@pytest.fixture(scope="module")
def h2chain_pdmet(tmp_path_factory):
    """One H2 per cell (d=1.0 A, a=3.0 A: bound, closed-shell, weakly coupled) at
    Gamma, IAO+PAO, CASSCF(2,2) + NEVPT2 over 3 singlet roots (e_shift).
    """
    from pyscf.pbc import gto, scf, df
    from pdmet import dmet

    work = tmp_path_factory.mktemp("h2chain_nto")
    gdf_file = os.path.join(work, "gdf.h5")

    d_hh, a_x, vac = 1.0, 3.0, 20.0
    cell = gto.Cell()
    cell.atom = [
        ["H", (0.0, 0.5 * vac, 0.5 * vac)],
        ["H", (d_hh, 0.5 * vac, 0.5 * vac)],
    ]
    cell.a = np.diag([a_x, vac, vac])
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.spin = 0
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    if not os.path.exists(gdf_file):
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_file
        gdf.build()

    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.with_df._cderi = gdf_file
    kmf.exxdiv = None
    kmf.run()

    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="CASSCF")
    pdmet_obj.lobasis.minao = {"H": "gth-szv"}
    pdmet_obj.emb.impCluster = [1, 2]  # the H2 molecule
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (2, 2)  # sigma_g, sigma_u
    pdmet_obj.solver.e_shift = 0.2  # keep singlets
    pdmet_obj.solver.nevpt2_roots = [0, 1, 2]
    pdmet_obj.solver.nevpt2_nroots = 3
    pdmet_obj.solver.nto = True  # explicit-flag contract: NEVPT2 no longer implies NTOs
    pdmet_obj.solver.nroots = 1
    pdmet_obj.initialize()
    pdmet_obj.one_shot()
    return pdmet_obj


def _sigma_excited_state(pdmet_obj, floor=1e-3):
    """Index of the most asymmetric (transition-like) excited state, or None."""
    best, best_asym = None, 0.0
    for state in range(1, len(pdmet_obj.ntos_per_root)):
        lam = pdmet_obj.ntos_per_root[state]["lambdas"]
        if not len(lam) or lam[0] < floor:
            continue
        T = pdmet_obj.t_dm1s[state]
        asym = np.linalg.norm(T - T.T) / np.linalg.norm(T)
        if asym > best_asym:
            best, best_asym = state, asym
    return best, best_asym


@pytest.mark.slow
def test_h2_chain_is_closed_shell_not_biradical(h2chain_pdmet):
    """Ground state is closed-shell sigma_g^2, not a biradical (which would make
    every transition density symmetric -> donor==acceptor).
    """
    # state 0 = ground density; its NTO weights are the occupations, dominant ~2.
    lam0 = h2chain_pdmet.ntos_per_root[0]["lambdas"]
    assert lam0[0] / lam0.sum() > 0.7, (
        f"ground state looks biradical/open-shell: weights={lam0}"
    )


@pytest.mark.slow
def test_h2_chain_sigma_to_sigmastar_is_rank1(h2chain_pdmet):
    """The sigma_g -> sigma_u excitation is (near-)rank-1 with one dominant pair."""
    state, asym = _sigma_excited_state(h2chain_pdmet)
    assert state is not None, "no transition-like excited state found"
    assert asym > 0.7, f"transition density not asymmetric enough (asym={asym:.3f})"

    info = h2chain_pdmet.ntos_per_root[state]
    lam = info["lambdas"]

    # rank-1: one dominant singular value carries ~all of sum(lambda)
    assert lam[0] / lam.sum() > 0.95, f"not rank-1; lambdas={lam}"
    # total-density convention: the dominant weight is ~2 (= per-spin sigma~1)
    assert 1.5 < lam[0] < 2.05, f"dominant lambda off expected ~2: {lam[0]}"
    # hole and particle are DIFFERENT orbitals (sigma_g vs sigma_u)
    assert _eo_overlap(info["V_hole"][:, 0], info["U_part"][:, 0]) < 0.3


@pytest.mark.slow
def test_h2_chain_hole_is_bonding_particle_is_antibonding(h2chain_pdmet):
    """hole in the OCCUPIED space (sigma_g), particle in the VIRTUAL space
    (sigma_u) -- the basis-free way to assert bonding vs antibonding.
    """
    state, _ = _sigma_excited_state(h2chain_pdmet)
    info = h2chain_pdmet.ntos_per_root[state]
    hole = info["V_hole"][:, 0]
    part = info["U_part"][:, 0]

    # embedded mean-field MOs in the (orthonormal) embedding basis
    mf = h2chain_pdmet.qcsolver.mf
    C = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ) > 0
    C_occ, C_virt = C[:, occ], C[:, ~occ]

    def frac_in(C_sub, vec):
        proj = C_sub.conj().T @ vec
        return float(np.vdot(proj, proj).real) / float(np.vdot(vec, vec).real)

    assert frac_in(C_occ, hole) > 0.8, "hole NTO is not an occupied (bonding) orbital"
    assert frac_in(C_virt, part) > 0.8, (
        "particle NTO is not a virtual (antibonding) orbital"
    )


@pytest.mark.slow
def test_h2_chain_transition_density_survives_nevpt2(h2chain_pdmet):
    """Regression guard: NEVPT2 canonicalizes mc_ci in place, so NTOs must be
    taken in a pre-pass BEFORE it (else trans_rdm1 goes symmetric/diagonal and
    donor==acceptor). A real transition must stay asymmetric (asym ~ sqrt(2)).
    """
    state, _ = _sigma_excited_state(h2chain_pdmet)
    assert state is not None
    T = h2chain_pdmet.t_dm1s[state]
    asym = np.linalg.norm(T - T.T) / np.linalg.norm(T)
    assert asym > 1.0, (
        f"transition density is near-symmetric (asym={asym:.3f}) -- NTOs likely "
        "computed after NEVPT2 mutated mc_ci (see _nevpt2_fci_roots pre-pass)."
    )


# --------------------------------------------------------------------------- #
#  Manual entry point: `python tests/test_nto.py [-m "not slow"] ...`          #
#  Just forwards to pytest so fixtures/markers still work.                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))


# --------------------------------------------------------------------------- #
#  5. Both sources requested: contract R3 (NEVPT2 mc_ci wins, no doubling)    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def h2chain_pdmet_both(tmp_path_factory):
    """nroots=3 SA-CASSCF AND nevpt2_roots=[0,1,2]: analysis must come from
    the NEVPT2 mc_ci only -- one list of 3, never 6."""
    from pyscf.pbc import gto, scf, df
    from pdmet import dmet

    work = tmp_path_factory.mktemp("h2chain_both")
    gdf_file = os.path.join(work, "gdf.h5")

    d_hh, a_x, vac = 1.0, 3.0, 20.0
    cell = gto.Cell()
    cell.atom = [
        ["H", (0.0, 0.5 * vac, 0.5 * vac)],
        ["H", (d_hh, 0.5 * vac, 0.5 * vac)],
    ]
    cell.a = np.diag([a_x, vac, vac])
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.spin = 0
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    if not os.path.exists(gdf_file):
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_file
        gdf.build()

    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.with_df._cderi = gdf_file
    kmf.exxdiv = None
    kmf.run()

    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="CASSCF")
    pdmet_obj.lobasis.minao = {"H": "gth-szv"}
    pdmet_obj.emb.impCluster = [1, 2]
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (2, 2)
    pdmet_obj.solver.e_shift = 0.2
    pdmet_obj.solver.nroots = 3
    pdmet_obj.solver.state_percent = [1.0, 0.0, 0.0]
    pdmet_obj.solver.nevpt2_roots = [0, 1, 2]
    pdmet_obj.solver.nevpt2_nroots = 3
    pdmet_obj.solver.nto = True
    pdmet_obj.solver.ndo = True
    pdmet_obj.initialize()
    pdmet_obj.one_shot()
    return pdmet_obj


@pytest.mark.slow
def test_single_source_no_doubling(h2chain_pdmet_both):
    """SA block must be skipped when nevpt2_roots is set (else every state
    index is silently wrong)."""
    assert len(h2chain_pdmet_both.ntos_per_root) == 3, "both sources ran"
    assert len(h2chain_pdmet_both.ndos_per_root) == 3, "both sources ran"


@pytest.mark.slow
def test_both_sources_analysis_is_pristine(h2chain_pdmet_both):
    """The mc_ci source was analyzed pre-canonicalization: sigma->sigma*
    transition density asymmetric, |kappa| ~ 1."""
    lams = [h2chain_pdmet_both.ntos_per_root[s]["lambdas"] for s in (1, 2)]
    state = 1 + int(np.argmax([lam[0] if len(lam) else 0.0 for lam in lams]))

    T = h2chain_pdmet_both.t_dm1s[state]
    assert np.linalg.norm(T - T.T) / np.linalg.norm(T) > 0.5
    assert abs(h2chain_pdmet_both.ndos_per_root[state]["kappa"][0]) > 0.5
    assert abs(h2chain_pdmet_both.ndos_per_root[state]["p_A"] - 1.0) < 0.3
