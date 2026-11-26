import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import httpx
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('hard_runner', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_config(tmp_path, model='openai/gpt-6-astra', cap=1227):
    auth = tmp_path / 'authorization.json'
    ledger = tmp_path / 'budget.sqlite'
    runner.write_json(auth, dict(source_available_units=1227, cumulative_cap_rub=cap, budget_path=str(ledger.resolve())))
    return dict(model=model, max_output=1024, temperature=0,
                input_rub_per_million=1779.92, output_rub_per_million=8899.58,
                cache_write_rub_per_million=2224.9, budget_path=str(ledger.resolve()),
                budget_authorization=str(auth), budget_authorization_sha256=runner.digest(auth),
                rub_limit=cap, attempt_limit=100, max_request_bytes=50000, max_input_tokens=51000,
                cost_reserve_multiplier=runner.MARGIN, bound_request_input=True,
                strict_dedup=True, max_attempts_per_key=1, strict_model=True)


def test_dedup_and_shared_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('API_KEY', 'mock-key')
    calls = []
    def handler(req):
        calls.append(req)
        model = json.loads(req.content)['model']
        return httpx.Response(200, json={'model': model, 'usage': {'prompt_tokens': 10, 'completion_tokens': 3},
            'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]})
    cfg = make_config(tmp_path)
    msgs = [{'role': 'user', 'content': 'test'}]
    c = runner.Client(tmp_path / 'budget.sqlite', cfg, transport=httpx.MockTransport(handler))
    try:
        first = c.call('stable', msgs)
        assert c.call('stable', msgs) == first
        assert len(calls) == 1
        exposure = c.db.execute('SELECT sum(charge) FROM reservations').fetchone()[0]
        assert exposure == pytest.approx(first['rub'] * runner.MARGIN)
        assert first['rub'] < runner.request_reserve(cfg, msgs)
    finally:
        c.db.close(); c.http.close()


def test_budget_blocks_before_network(tmp_path, monkeypatch):
    monkeypatch.setenv('API_KEY', 'mock-key')
    def forbidden(req):
        raise AssertionError('Network must not be called')
    cfg = make_config(tmp_path, cap=1)
    c = runner.Client(tmp_path / 'budget.sqlite', cfg, transport=httpx.MockTransport(forbidden))
    try:
        with pytest.raises(RuntimeError, match='spending limit'):
            c.call('new', [{'role': 'user', 'content': 'test'}])
        assert c.db.execute('SELECT count(*) FROM calls').fetchone()[0] == 0
    finally:
        c.db.close(); c.http.close()


def test_reasoning_is_not_double_charged(tmp_path):
    cfg = make_config(tmp_path)
    from client import usage_cost
    usage = {'prompt_tokens': 100, 'completion_tokens': 200,
             'completion_tokens_details': {'reasoning_tokens': 180}}
    assert usage_cost(cfg, usage) == pytest.approx((100*1779.92+200*8899.58)/1e6)
