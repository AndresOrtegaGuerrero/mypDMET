"""
Round-trip test for the IAO+PAO chkfile (pdmet.save_lo / pdmet.load_lo).

Three checks, each isolated to one concern:

  1. Round-trip:   build → save → load reproduces ao2lo, lo_labels, minao
                   bit-for-bit.
  2. Lazy load:    the load path does NOT call make_iao_pao_kbasis (the
                   expensive build is genuinely skipped, not just
                   overwritten afterwards).
  3. Shape guard:  _verify_loaded_lo fires when the chkfile was built for
                   a different cell shape, instead of letting garbage
                   propagate into the OEH/Fock build downstream.

The test system is a 2-atom H cell at gamma — small enough that the
build path is fast (so #2 has something cheap to compare against), and
small enough that we can stress-test by reloading the chkfile against a
different basis (#3).
"""

import os
import sys
import tempfile

import numpy as np

# Allow running from tests/ without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyscf.pbc import gto, scf, df

from pdmet import dmet
from pdmet import localbasis as _localbasis_mod  # for monkey-patching in test #2


# ---------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------


def build_h2_cell(basis="gth-dzv"):
    """Tiny gamma-only H2 cell. Stays under 4 AOs for fast SCF."""
    cell = gto.Cell()
    cell.atom = "H 5 5 4; H 5 5 5"
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.a = np.eye(3) * 10
    cell.unit = "Angstrom"
    cell.verbose = 0
    cell.spin = 0
    cell.build()
    return cell


def converged_kmf(cell, gdf_file):
    """KRHF with cached GDF for repeatability."""
    kpts = cell.make_kpts([1, 1, 1])
    if not os.path.exists(gdf_file):
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_file
        gdf.build()
    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.with_df._cderi = gdf_file
    kmf.exxdiv = None
    kmf.max_cycle = 80
    kmf.kernel()
    return kmf


def make_pdmet(cell, kmf, lo_chkfile=None, minao="gth-szv"):
    """Construct an IAO+PAO pDMET, optionally loading an LO chkfile."""
    p = dmet.pDMET(cell, kmf, w90=None, lo_method="iao+pao", solver="HF")
    p.lobasis.minao = minao
    p.emb.impCluster = [1]
    p.emb.imp_orbital_filter = {"H": ["1s"]}
    if lo_chkfile is not None:
        p.load_lo(lo_chkfile)
    p.initialize()
    return p


# ---------------------------------------------------------------------
# Test 1: round-trip equality
# ---------------------------------------------------------------------


def test_round_trip(cell, kmf, chk):
    """build → save → load must reproduce ao2lo / lo_labels / minao exactly."""
    print("[1] round-trip: build → save → load == build")

    p_built = make_pdmet(cell, kmf)
    p_built.save_lo(chk)
    assert os.path.exists(chk), "save_lo did not write a chkfile"

    p_loaded = make_pdmet(cell, kmf, lo_chkfile=chk)

    # ao2lo must match to round-off; chkfile is float64, no algorithmic step
    # between save and load other than HDF5 I/O.
    np.testing.assert_allclose(
        p_loaded.local.ao2lo,
        p_built.local.ao2lo,
        atol=1e-12,
        err_msg="ao2lo drifted across save/load",
    )
    assert list(p_loaded.local.lo_labels) == list(p_built.local.lo_labels), (
        f"lo_labels drifted: built={p_built.local.lo_labels} "
        f"loaded={p_loaded.local.lo_labels}"
    )
    assert p_loaded.local.minao == p_built.local.minao, (
        f"minao drifted: built={p_built.local.minao!r} loaded={p_loaded.local.minao!r}"
    )
    print("    PASS")


# ---------------------------------------------------------------------
# Test 2: load skips the build
# ---------------------------------------------------------------------


def test_load_skips_build(cell, kmf, chk):
    """Loading a chkfile must NOT invoke make_iao_pao_kbasis."""
    print("[2] load-path bypasses make_iao_pao_kbasis")

    # Spy: replace the build with a tripwire. If anyone calls it during
    # the load path, raise loudly so the test fails with a clear message.
    original = _localbasis_mod.make_iao_pao_kbasis

    def tripwire(*args, **kwargs):
        raise RuntimeError(
            "make_iao_pao_kbasis was called on the load path — "
            "the build short-circuit is broken."
        )

    _localbasis_mod.make_iao_pao_kbasis = tripwire
    try:
        p = make_pdmet(cell, kmf, lo_chkfile=chk)
        # Sanity: load actually populated the data.
        assert p.local.ao2lo is not None and p.local.ao2lo.size > 0
    finally:
        # Always restore — otherwise other tests in the same process break.
        _localbasis_mod.make_iao_pao_kbasis = original
    print("    PASS")


# ---------------------------------------------------------------------
# Test 3: shape-mismatch detection
# ---------------------------------------------------------------------


def test_verify_shape_mismatch(chk, tmp_dir):
    """Loading a dzv-based chkfile against an szv cell must fail loudly."""
    print("[3] _verify_loaded_lo catches a chkfile/cell mismatch")

    # Build a *different* cell (smaller basis ⇒ fewer AOs).
    bad_cell = build_h2_cell(basis="gth-szv")
    bad_kmf = converged_kmf(bad_cell, os.path.join(tmp_dir, "gdf-h2-szv.h5"))

    p = dmet.pDMET(bad_cell, bad_kmf, w90=None, lo_method="iao+pao", solver="HF")
    p.lobasis.minao = "gth-szv"
    p.load_lo(chk)  # chkfile from the dzv cell — wrong shape on purpose

    try:
        p.initialize()
    except AssertionError as e:
        if "loaded ao2lo shape" not in str(e):
            raise AssertionError(
                f"got AssertionError, but not the expected message:\n  {e}"
            ) from None
        print("    PASS")
        return

    raise AssertionError(
        "_verify_loaded_lo did not catch the cell/chkfile shape mismatch"
    )


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def main():
    cell = build_h2_cell()

    with tempfile.TemporaryDirectory() as tmp:
        gdf = os.path.join(tmp, "gdf-h2.h5")
        chk = os.path.join(tmp, "iao_pao.chk")

        kmf = converged_kmf(cell, gdf)

        test_round_trip(cell, kmf, chk)
        test_load_skips_build(cell, kmf, chk)
        test_verify_shape_mismatch(chk, tmp)

    print("\nAll save/load checks passed.")


if __name__ == "__main__":
    main()
