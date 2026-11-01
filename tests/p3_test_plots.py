# OWNER: P3
from qdec.contracts import QUICK_GRID
from qdec.mocks import mock_service
from qdec.queue.synthetic import same_mean_lognormal
from qdec.results.plots import SYSTEM_COLOR, _colors, plot_backlog_explosion


def test_backlog_explosion_synthetic(tmp_path):
    df = same_mean_lognormal(900.0, {"light": 0.3, "heavy": 1.5}, 2000, seed=0)
    out = plot_backlog_explosion(df, tmp_path / "fig" / "bx.png", 1000.0)
    assert out.exists() and out.stat().st_size > 10_000


def test_backlog_explosion_on_mock_service(tmp_path):
    df = mock_service(QUICK_GRID[0].tag, n=2000)
    out = plot_backlog_explosion(df, tmp_path / "bx.png", 1000.0, note="MOCK DATA")
    assert out.exists()


def test_system_colors_are_fixed_and_others_do_not_collide():
    c = _colors(["pim_cache", "light", "heavy"])
    assert c["pim_cache"] == SYSTEM_COLOR["pim_cache"]
    assert len(set(c.values())) == 3
