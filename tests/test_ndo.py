"""NDO tests: numpy-only eigh core + invariants, settings validation, the
texcited presentation layer (on a stub pDMET -- no SCF needed), and an
IAO+PAO H4 pDMET integration exercising the state-average source.

    pytest tests/test_ndo.py -v               # all
    pytest tests/test_ndo.py -v -m "not slow" # skip the heavy DMET run

Reference: Plasser, Wormit, Dreuw, J. Chem. Phys. 141, 024106 (2014), Sec. VI.
"""

import os
import types
import numpy as np
import pytest

pytest.importorskip("pyscf")  # whole stack imports pyscf at load

from pdmet.qcsolvers.casbase import BaseCASSolver
from pdmet.settings import SolverSettings
from pdmet.tools import texcited


# --------------------------------------------------------------------------- #
#  1. Pure-math core: eigendecomposition convention and invariants            #
# --------------------------------------------------------------------------- #


def _random_symmetric(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


def test_kappa_sorted_by_magnitude_and_W_orthonormal():
    """kappa sorted by |kappa| desc; W columns orthonormal."""
    D = _random_symmetric(7, 0)
    kappa, W, D_det, D_att, stats = BaseCASSolver._decompose_ndo_cas(D)

    assert (np.diff(np.abs(kappa)) <= 1e-12).all(), "not sorted by |kappa|"
    assert np.allclose(W.T @ W, np.eye(W.shape[1]), atol=1e-10)
    assert W.shape == (7, len(kappa))


def test_eigendecomposition_reconstructs_delta():
    """W diag(kappa) W^T rebuilds Delta -- guards ordering/filtering bugs."""
    D = _random_symmetric(6, 1)
    kappa, W, *_ = BaseCASSolver._decompose_ndo_cas(D)
    assert np.allclose((W * kappa) @ W.T, D, atol=1e-10)


def test_detachment_attachment_split():
    """D_det + D_att == Delta (Eqs. 70-73); D_det <= 0 <= D_att as operators;
    traces match the promotion numbers p_D / p_A (Eq. 72)."""
    D = _random_symmetric(8, 2)
    kappa, W, D_det, D_att, stats = BaseCASSolver._decompose_ndo_cas(D)

    assert np.allclose(D_det + D_att, D, atol=1e-10)
    assert (np.linalg.eigvalsh(D_det) <= 1e-10).all(), "detachment not <= 0"
    assert (np.linalg.eigvalsh(D_att) >= -1e-10).all(), "attachment not >= 0"
    assert abs(np.trace(D_det) - stats["p_D"]) < 1e-10
    assert abs(np.trace(D_att) - stats["p_A"]) < 1e-10


def test_traceless_delta_conserves_electrons():
    """Tr(Delta) = 0  =>  p_A == -p_D (Eq. 67: no electrons gained or lost)."""
    D0 = _random_symmetric(6, 3)
    D = D0 - np.eye(6) * (np.trace(D0) / 6)  # force Tr = 0

    kappa, W, D_det, D_att, stats = BaseCASSolver._decompose_ndo_cas(D)
    assert abs(stats["p_A"] + stats["p_D"]) < 1e-10


def test_one_sided_delta_has_zero_pr_without_crashing():
    """A PSD Delta has no detachment side: p_D == 0 and PR_D == 0.0 (the
    zero-guard in the stats), with no ZeroDivisionError."""
    rng = np.random.default_rng(4)
    B = rng.standard_normal((5, 3))
    D = B @ B.T  # PSD: all eigenvalues >= 0

    kappa, W, D_det, D_att, stats = BaseCASSolver._decompose_ndo_cas(D)
    assert (kappa > 0).all()
    assert stats["p_D"] == 0.0
    assert stats["PR_D"] == 0.0
    assert stats["PR_A"] > 0.0
    assert np.allclose(D_det, 0.0)


def test_numerical_zero_kappas_are_dropped():
    """|kappa| <= thresh entries are pruned as floating-point noise."""
    D = np.diag([1.0, -0.5, 1e-15, 0.0])
    kappa, W, *_ = BaseCASSolver._decompose_ndo_cas(D, thresh=1e-12)
    assert len(kappa) == 2
    assert np.allclose(sorted(kappa), [-0.5, 1.0])


def test_cis_limit_ndos_equal_ntos():
    """Rank-2 single-excitation model (Eq. 78): for Delta = aa^T - dd^T with
    orthonormal donor d / acceptor a, kappa = {+1, -1} and the NDOs coincide
    with the NTOs of T = |a><d| (up to sign)."""
    rng = np.random.default_rng(5)
    d = rng.standard_normal(6)
    d /= np.linalg.norm(d)
    a = rng.standard_normal(6)
    a -= (a @ d) * d  # orthogonalize
    a /= np.linalg.norm(a)

    Delta = np.outer(a, a) - np.outer(d, d)
    kappa, W, D_det, D_att, stats = BaseCASSolver._decompose_ndo_cas(Delta)

    assert len(kappa) == 2
    assert np.allclose(sorted(kappa), [-1.0, 1.0], atol=1e-12)
    assert abs(stats["p_A"] - 1.0) < 1e-12 and abs(stats["p_D"] + 1.0) < 1e-12

    # NTO side of the same excitation
    lam, V_hole, U_part = BaseCASSolver._decompose_nto_cas(np.outer(a, d))
    assert abs(lam[0] - 1.0) < 1e-12

    w_att = W[:, np.argmax(kappa)]  # kappa = +1 column
    w_det = W[:, np.argmin(kappa)]  # kappa = -1 column
    assert abs(abs(w_det @ V_hole[:, 0]) - 1.0) < 1e-10, "NDO(det) != NTO(hole)"
    assert abs(abs(w_att @ U_part[:, 0]) - 1.0) < 1e-10, "NDO(att) != NTO(particle)"


# --------------------------------------------------------------------------- #
#  2. Settings validation (contract: mirror of the nto block)                 #
# --------------------------------------------------------------------------- #


def test_ndo_requires_multiroot():
    # a single-root run has no difference density to decompose
    with pytest.raises(ValueError, match="multi-root"):
        SolverSettings(ndo=True).validate()
    with pytest.raises(ValueError, match="multi-root"):
        SolverSettings(ndo_export=True).validate()


def test_ndo_flag_ok_without_nevpt2():
    SolverSettings(ndo=True, nroots=2).validate()  # must not raise
    SolverSettings(ndo=True, state_average_=[0.5, 0.5], nroots=2).validate()


def test_ndo_npairs_must_be_positive():
    s = SolverSettings(ndo=True, nroots=2, ndo_npairs=0)
    with pytest.raises(ValueError, match="ndo_npairs"):
        s.validate()


def test_ndo_kappa_floor_nonnegative():
    s = SolverSettings(ndo=True, nroots=2, ndo_kappa_floor=-1.0)
    with pytest.raises(ValueError, match="ndo_kappa_floor"):
        s.validate()


def test_valid_ndo_settings_pass_via_nevpt2():
    SolverSettings(
        ndo_export=True, nevpt2_roots=[0, 1], ndo_npairs=2, ndo_kappa_floor=1e-3
    ).validate()


def test_nto_and_ndo_flags_compose():
    SolverSettings(nto=True, ndo=True, nroots=3).validate()  # must not raise


# --------------------------------------------------------------------------- #
#  3. texcited presentation layer on a stub pDMET (no SCF required)           #
# --------------------------------------------------------------------------- #


def _stub_pdmet(nkpts=1, norb=4):
    """Duck-typed pDMET carrying one excited state with 1 detachment and 1
    attachment NDO -- enough for tables, floors, guards, rotation matrices.
    """
    W = np.zeros((norb, 2))
    W[0, 0] = 1.0  # detachment orbital
    W[1, 1] = 1.0  # attachment orbital
    kappa = np.array([-0.9, 0.9])  # |kappa| sorted desc (tie)
    info = {
        "kappa": kappa,
        "W": W,
        "D_det": (W * np.minimum(kappa, 0)) @ W.T,
        "D_att": (W * np.maximum(kappa, 0)) @ W.T,
        "p_D": -0.9,
        "p_A": 0.9,
        "PR_D": 1.0,
        "PR_A": 1.0,
    }
    empty = {
        "kappa": np.array([]),
        "W": np.zeros((norb, 0)),
        "D_det": np.zeros((norb, norb)),
        "D_att": np.zeros((norb, norb)),
        "p_D": 0.0,
        "p_A": 0.0,
        "PR_D": 0.0,
        "PR_A": 0.0,
    }
    obj = types.SimpleNamespace()
    obj.solver = SolverSettings(ndo=True, nroots=2)
    obj.ndos_per_root = [empty, info]  # state 0 = ground (Delta^00 = 0)
    obj.d_dm1s = [np.zeros((norb, norb)), (W * kappa) @ W.T]
    obj.local = types.SimpleNamespace(Nkpts=nkpts)
    obj.emb_orbs = np.eye(norb)[None, :, :]  # (1, nlo, Norb) identity EO->LO
    return obj


def test_get_ndos_table_and_entries(capsys):
    obj = _stub_pdmet()
    results = texcited.get_ndos(obj, state=1, n_orbs=2, outdir=None)
    out = capsys.readouterr().out

    assert "NDOs for state 1" in out
    assert "PR_D" in out and "PR_A" in out
    assert len(results) == 2
    types_seen = {r["type"] for r in results}
    assert types_seen == {"D", "A"}
    for r in results:
        assert {"orb", "kappa", "type", "coeff"} <= set(r)
        assert r["coeff"].shape == (4,)


def test_get_ndos_floor_prunes(capsys):
    obj = _stub_pdmet()
    results = texcited.get_ndos(obj, state=1, n_orbs=5, kappa_floor=0.95)
    assert results == []  # both |kappa| = 0.9 < 0.95


def test_ground_state_has_empty_ndos(capsys):
    """State 0 keeps a trivial entry so NTO/NDO lists stay index-aligned."""
    obj = _stub_pdmet()
    results = texcited.get_ndos(obj, state=0)
    assert results == []


def test_rotate_mat_ndo_shape_and_columns():
    obj = _stub_pdmet()
    R = texcited.rotate_mat_ndo(obj, state=1, n_orbs=2)
    assert R.shape == (4, 2)
    # identity EO->LO: columns are the W columns themselves
    assert abs(abs(R[0, 0]) - 1.0) < 1e-12
    assert abs(abs(R[1, 1]) - 1.0) < 1e-12


def test_rotate_mat_ndo_refuses_empty_state():
    obj = _stub_pdmet()
    with pytest.raises(AssertionError, match="no NDOs"):
        texcited.rotate_mat_ndo(obj, state=0)


def test_kpts_guard_blocks_ndo_cube_export(tmp_path):
    obj = _stub_pdmet(nkpts=2)
    # table works at any k-mesh ...
    assert texcited.get_ndos(obj, state=1, outdir=None) is not None
    # ... but cube export is Gamma-only
    with pytest.raises(NotImplementedError, match="Gamma-only"):
        texcited.get_ndos(obj, state=1, outdir=str(tmp_path))


def test_missing_analysis_gives_actionable_error():
    obj = _stub_pdmet()
    obj.ndos_per_root = None
    with pytest.raises(ValueError, match="solver.ndo=True"):
        texcited.get_ndos(obj, state=0)


def test_attach_detach_density_traces(monkeypatch):
    """AO-promoted D_det / D_att keep their traces (ao2eo unitary here)."""
    obj = _stub_pdmet()
    obj.local.get_ao2eo = lambda emb_orbs: np.eye(4)[None, :, :]
    D_det_ao, D_att_ao = texcited.get_attach_detach_density(obj, state=1)
    assert abs(np.trace(D_det_ao) - (-0.9)) < 1e-12
    assert abs(np.trace(D_att_ao) - 0.9) < 1e-12


def test_auto_export_respects_flags(capsys, monkeypatch):
    """auto_export prints NDO tables when ndo_export is on and skips cubes
    (with a note) at k > 1; silent when the flag is off."""
    obj = _stub_pdmet(nkpts=2)

    obj.solver.ndo_export = False
    texcited.auto_export(obj)
    assert "NDOs for state" not in capsys.readouterr().out

    obj.solver.ndo_export = True
    texcited.auto_export(obj)
    out = capsys.readouterr().out
    assert "NDOs for state 1" in out
    assert "Gamma-only" in out  # the k-mesh note, not an exception


# --------------------------------------------------------------------------- #
#  4. IAO+PAO integration (H4, Gamma-only): the STATE-AVERAGE source          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def h4_pdmet(tmp_path_factory):
    """H4 pDMET, multi-root CASCI (no NEVPT2), nto+ndo both on -- exercises
    the state-average analysis source and the NTO/NDO cross-checks.
    """
    from pyscf.pbc import gto, scf, df
    from pdmet import dmet

    work = tmp_path_factory.mktemp("h4_ndo")
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

    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="CASCI")
    pdmet_obj.lobasis.minao = {"H": "gth-szv"}
    pdmet_obj.emb.impCluster = [1, 2, 3, 4]
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (4, 4)
    pdmet_obj.solver.e_shift = 0.2
    pdmet_obj.solver.nroots = 4
    pdmet_obj.solver.state_percent = [1.0, 0.0, 0.0, 0.0]  # ground-state density
    pdmet_obj.solver.nto = True
    pdmet_obj.solver.ndo = True
    pdmet_obj.initialize()
    pdmet_obj.one_shot()
    return pdmet_obj


def _strongest_excited_state(pdmet_obj, floor=1e-3):
    best, best_lam = None, 0.0
    for state in range(1, len(pdmet_obj.ntos_per_root)):
        lam = pdmet_obj.ntos_per_root[state]["lambdas"]
        top = float(lam[0]) if len(lam) else 0.0
        if top > best_lam:
            best, best_lam = state, top
    return (best, best_lam) if best_lam >= floor else (None, best_lam)


@pytest.fixture(scope="module")
def excited_state(h4_pdmet):
    state, top = _strongest_excited_state(h4_pdmet)
    if state is None:
        pytest.skip(
            f"no excited state with a dominant NTO pair above floor "
            f"(strongest top-lambda={top:.2e}); tune cas / e_shift / nroots."
        )
    return state


@pytest.mark.slow
def test_ndos_per_root_populated_and_aligned(h4_pdmet):
    assert h4_pdmet.ndos_per_root is not None, "solver did not emit NDOs"
    assert h4_pdmet.ntos_per_root is not None, "solver did not emit NTOs"
    assert len(h4_pdmet.ndos_per_root) == len(h4_pdmet.ntos_per_root), (
        "NTO and NDO lists must stay index-aligned per state"
    )
    for info in h4_pdmet.ndos_per_root:
        assert {"kappa", "W", "D_det", "D_att", "p_D", "p_A", "PR_D", "PR_A"} <= set(
            info
        )


@pytest.mark.slow
def test_ground_state_delta_is_trivial(h4_pdmet):
    """Entry 0 is Delta^00 = 0: kept only for index alignment with the NTOs."""
    assert len(h4_pdmet.ndos_per_root[0]["kappa"]) == 0
    assert np.allclose(h4_pdmet.d_dm1s[0], 0.0, atol=1e-10)


@pytest.mark.slow
def test_integration_invariants_eo_basis(h4_pdmet):
    """Tr(Delta) == 0 (Eq. 67), p_A == -p_D, shapes, and reconstruction in EO."""
    Norb = h4_pdmet.emb_orbs.shape[-1]
    for state in range(1, len(h4_pdmet.ndos_per_root)):
        info = h4_pdmet.ndos_per_root[state]
        kappa, W = info["kappa"], info["W"]
        d_emb = h4_pdmet.d_dm1s[state]

        assert abs(np.trace(d_emb)) < 1e-8, "electron number not conserved"
        assert abs(info["p_A"] + info["p_D"]) < 1e-8
        assert W.shape == (Norb, len(kappa))
        assert np.allclose(info["D_det"] + info["D_att"], d_emb, atol=1e-8)
        assert abs(np.trace(info["D_att"]) - info["p_A"]) < 1e-8


@pytest.mark.slow
def test_diradical_ndos_expose_recoupling(h4_pdmet, excited_state):
    """Near-square H4 is a DIRADICAL (ground NO occupations ~ 2,1,1,0): its low
    excitations are open-shell recouplings, so the transition density is large
    while the total density barely moves (JCP 141, 024106). NTOs and NDOs MUST
    disagree here -- that disagreement is the diagnostic, not a bug.
    Verified against plain pyscf FCI at this geometry: lam ~ 0.95 per lane,
    |kappa| ~ 0.05.
    """
    lam0 = h4_pdmet.ntos_per_root[excited_state]["lambdas"][0]
    ndo = h4_pdmet.ndos_per_root[excited_state]
    kap0 = abs(ndo["kappa"][0]) if len(ndo["kappa"]) else 0.0

    assert lam0 > 0.5, "transition density should still see the excitation"
    assert kap0 < 0.2 * lam0, (
        f"top |kappa|={kap0:.3f} vs lambda={lam0:.3f}: for this diradical the "
        f"density change must be small (recoupling, not charge transfer)"
    )
    assert abs(ndo["p_A"]) < 0.3, f"promotion number p={ndo['p_A']:.3f} too large"


@pytest.mark.slow
def test_get_ndos_prints_and_returns(h4_pdmet, excited_state, capsys):
    results = h4_pdmet.get_ndos(state=excited_state, n_orbs=2, outdir=None)
    out = capsys.readouterr().out
    assert f"NDOs for state {excited_state}" in out
    assert len(results) >= 1
    for entry in results:
        assert {"orb", "kappa", "type", "coeff"} <= set(entry)
        assert entry["coeff"].shape == (h4_pdmet.emb_orbs.shape[-1],)


@pytest.mark.slow
def test_gamma_ndo_cube_export_writes_files(h4_pdmet, excited_state, tmp_path):
    outdir = os.path.join(tmp_path, "ndos")
    results = h4_pdmet.get_ndos(
        state=excited_state,
        n_orbs=2,
        kappa_floor=1e-6,
        outdir=outdir,
        grid=(20, 20, 20),
    )
    assert len(results) >= 1
    files = os.listdir(outdir)
    cubes = [f for f in files if f.endswith(".cube")]
    assert len(cubes) == len(results), f"expected one cube per NDO: {files}"
    for f in cubes:
        assert ("_det_" in f) or ("_att_" in f), f"role tag missing: {f}"


# --------------------------------------------------------------------------- #
#  5. Closed-shell H2 chain: the Eq. 78 limit (kappa tracks lambda)           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def h2chain_pdmet_sa(tmp_path_factory):
    """One H2 per cell (d=1.0 A, a=3.0 A: bound, closed-shell) at Gamma,
    IAO+PAO, multi-root CASCI(2,2) with nto+ndo -- the STATE-AVERAGE source,
    no NEVPT2. Complements test_nto.py's NEVPT2-source h2chain fixture.
    """
    from pyscf.pbc import gto, scf, df
    from pdmet import dmet

    work = tmp_path_factory.mktemp("h2chain_ndo")
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

    pdmet_obj = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="CASCI")
    pdmet_obj.lobasis.minao = {"H": "gth-szv"}
    pdmet_obj.emb.impCluster = [1, 2]  # the H2 molecule
    pdmet_obj.solver.twoS = 0
    pdmet_obj.solver.cas = (2, 2)  # sigma_g, sigma_u
    pdmet_obj.solver.e_shift = 0.2  # keep singlets
    pdmet_obj.solver.nroots = 3
    pdmet_obj.solver.state_percent = [1.0, 0.0, 0.0]  # ground-state density
    pdmet_obj.solver.nto = True
    pdmet_obj.solver.ndo = True
    pdmet_obj.initialize()
    pdmet_obj.one_shot()
    return pdmet_obj


@pytest.fixture(scope="module")
def h2_excited_state(h2chain_pdmet_sa):
    """The sigma_g -> sigma_u state: largest dominant NTO weight."""
    best, best_lam = None, 0.0
    for state in range(1, len(h2chain_pdmet_sa.ntos_per_root)):
        lam = h2chain_pdmet_sa.ntos_per_root[state]["lambdas"]
        top = float(lam[0]) if len(lam) else 0.0
        if top > best_lam:
            best, best_lam = state, top
    if best is None or best_lam < 1e-3:
        pytest.skip(f"no transition-carrying state (top lambda={best_lam:.2e})")
    return best


@pytest.mark.slow
def test_eq78_cross_check_closed_shell(h2chain_pdmet_sa, h2_excited_state):
    """Eq. 78 limit on a closed-shell single excitation: kappa ~ {+1, -1}
    (sigma_g emptied by one electron, sigma_u filled by one), p_A ~ 1, and the
    NDOs coincide with the NTOs. Thresholds are calibrated to the SPIN-TRACED
    convention: lambda lanes may merge to ~2 while |kappa| stays ~1, so the
    check is against absolute values, not the kappa/lambda ratio.
    """
    nto = h2chain_pdmet_sa.ntos_per_root[h2_excited_state]
    ndo = h2chain_pdmet_sa.ndos_per_root[h2_excited_state]

    kap0 = abs(ndo["kappa"][0])
    assert kap0 > 0.5, f"top |kappa|={kap0:.3f}: expected ~1 for sigma->sigma*"
    assert abs(ndo["p_A"] - 1.0) < 0.3, f"promotion number p={ndo['p_A']:.3f}"

    # detachment NDO == hole NTO (sigma_g); attachment NDO == particle (sigma_u)
    w_det = ndo["W"][:, np.argmin(ndo["kappa"])]
    w_att = ndo["W"][:, np.argmax(ndo["kappa"])]
    ov_hole = abs(w_det @ nto["V_hole"][:, 0])
    ov_part = abs(w_att @ nto["U_part"][:, 0])
    assert ov_hole > 0.9, f"det/hole overlap {ov_hole:.3f}"
    assert ov_part > 0.9, f"att/particle overlap {ov_part:.3f}"


@pytest.mark.slow
def test_eq78_electron_count_bookkeeping(h2chain_pdmet_sa, h2_excited_state):
    """Same state, global books: Tr(Delta) = 0 while p_A = -p_D ~ 1 -- one
    electron moved, none created."""
    ndo = h2chain_pdmet_sa.ndos_per_root[h2_excited_state]
    d_emb = h2chain_pdmet_sa.d_dm1s[h2_excited_state]
    assert abs(np.trace(d_emb)) < 1e-8
    assert abs(ndo["p_A"] + ndo["p_D"]) < 1e-8
    assert ndo["p_A"] > 0.7
