from datetime import date

from nhlmodel.pipelines import daily, nightly, once


def test_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("NHL_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("NHL_DRY_RUN", raising=False)
    assert not once.already_done("x")
    once.mark_done("x")
    assert once.already_done("x") and not once.already_done("x", force=True)
    monkeypatch.setenv("NHL_DRY_RUN", "1")
    once.mark_done("y")                       # dry runs never block a real post
    assert not once.already_done("y")


def test_morning_runs_once(tmp_path, monkeypatch):
    monkeypatch.setenv("NHL_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("NHL_DRY_RUN", raising=False)
    calls = []
    monkeypatch.setattr(daily, "backfill_season", lambda s: calls.append(s))
    monkeypatch.setattr(daily, "project_slate", lambda d, rt: ([], None, None))
    d = date(2026, 10, 30)
    once.mark_done(f"morning_{d.isoformat()}")
    daily.morning(d)
    assert calls == []                        # skipped: already posted
    daily.morning(d, force=True)
    assert len(calls) == 1                    # --force runs again


def test_nightly_posts_once(tmp_path, monkeypatch):
    monkeypatch.setenv("NHL_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("NHL_DRY_RUN", raising=False)
    posts = []
    monkeypatch.setattr(nightly, "backfill_season", lambda s: 0)
    monkeypatch.setattr(nightly, "refresh_extras", lambda s: None)
    monkeypatch.setattr(nightly, "_post", lambda season, day: posts.append(day))
    nightly.run()
    nightly.run()
    assert len(posts) == 1
    nightly.run(force=True)
    assert len(posts) == 2
