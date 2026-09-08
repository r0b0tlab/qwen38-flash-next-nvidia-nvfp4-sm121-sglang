"""A producer cannot bind rows to another endpoint or an obsolete server epoch."""

import pytest
from scripts import benchmark_evidence as e


def test_bound_epoch_checks_exact_endpoint_and_live_identity(monkeypatch):
    assert hasattr(e, "verify_bound_epoch"), "bound producer has no epoch check"
    from scripts import runtime_context

    called = []
    monkeypatch.setattr(
        runtime_context, "verify_http_epoch", lambda m: called.append(m)
    )
    m = {"runtime_context": {"endpoint": "http://127.0.0.1:30080"}}
    assert e.verify_bound_epoch(m, "http://127.0.0.1:30080") is True
    assert called == [m]
    with pytest.raises(ValueError):
        e.verify_bound_epoch(m, "http://other:30080")
    assert e.verify_bound_epoch(None, "http://test-only") is False
