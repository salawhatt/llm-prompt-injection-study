"""Mock-only controls for continuation budgets, deduplication and full accounting."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('test_continuation_module', Path(__file__).with_name('continuation.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def cases(families=2):
    return [dict(case=f'test-{family}-{variant}-{kind}', scene=f'test-{family}-{variant}',
                 family=f'test-family-{family}', variant=variant, kind=kind,
                 document='Контрольный документ.', question='Сколько?', gold='14', accepted=['14'],
                 marker='СЕРЫЙ КУБ', repeats=5 if variant == 0 and kind == 'attack' else 1)
            for variant in range(4) for family in range(families) for kind in ('attack', 'clean')]


def configs():
    return {name: dict(model=model, max_output=1024, temperature=0) for name, model in c.base.MODELS.items()}


def source_fixture():
    rows = cases(1)
    main = dict(case=rows[0]['case'], arm='A', model_name='sonnet55', repeat=1)
    robust = dict(case=rows[2]['case'], arm='B', model_name='astra', repeat=2)
    extra_case = dict(rows[0], case='test-supplement-attack')
    extra = dict(case=extra_case['case'], arm='A', model_name='sonnet55', repeat=1)
    sources = [('main_challenge', rows, [main], c.base.tag),
               ('supplement', [extra_case], [extra], c.supplement.get_tag),
               ('robustness', rows, [robust], c.base.tag)]
    return c.combine_lookups(sources, configs())


def ledger_fixture(lookup):
    calls = []
    reserves = []
    for i, (key, meta) in enumerate(lookup.items(), 1):
        result = dict(ok=True, model=c.base.MODELS[meta['job']['model_name']], request=meta['request'],
                      answer='14', finish='stop', usage=dict(prompt_tokens=100, completion_tokens=20,
                          completion_tokens_details=dict(reasoning_tokens=5),
                          prompt_tokens_details=dict(cached_tokens=10, cache_creation_tokens=8, cache_write_tokens=8)),
                      rub=float(i))
        calls.append(dict(id=i, key=key, raw=json.dumps(result)))
        reserves.append(dict(id=i, key=key, model=result['model'], charge=i * 1.05,
                             raw=json.dumps(dict(ok=True, rub=float(i)))))
    return calls, reserves


def test_plan_is_metadata_only_repeat_then_variant_then_family():
    rows = cases(16)
    plan = c.make_plan(rows)
    assert len(plan) == 768
    by_id = {r['case']: r for r in rows}
    assert all(by_id[j['case']]['kind'] == 'attack' and by_id[j['case']]['variant'] in (1, 2, 3)
               and j['repeat'] in (2, 3, 4, 5) for j in plan)
    assert len({(j['case'], j['arm'], j['model_name'], j['repeat']) for j in plan}) == 768
    ordering = [(j['repeat'], by_id[j['case']]['variant'], by_id[j['case']]['family']) for j in plan]
    assert ordering == sorted(ordering)
    for offset in range(0, len(plan), 2):
        pair = plan[offset:offset + 2]
        assert {j['model_name'] for j in pair} == set(c.base.MODELS)
        assert len({(j['case'], j['arm'], j['repeat']) for j in pair}) == 1


def test_union_lookup_rejects_colliding_calls():
    rows = cases(1)
    job = dict(case=rows[0]['case'], arm='B', model_name='astra', repeat=1)
    with pytest.raises(ValueError, match='collision'):
        c.combine_lookups([('main', rows, [job], c.base.tag), ('extra', rows, [job], c.base.tag)], configs())
    # Supplemental tag distinguishes namespaces without pretending to be an old call.
    lookup = c.combine_lookups([('main', rows, [job], c.base.tag),
                               ('extra', rows, [job], c.supplement.get_tag)], configs())
    assert len(lookup) == 2


def test_wave_falls_back_to_one_pair_then_stops_before_overspend(monkeypatch):
    rows = cases(1)
    plan = c.make_plan(rows)[:4]
    by_id = {r['case']: r for r in rows}
    monkeypatch.setattr(c.base, 'request_reserve', lambda cfg, messages: 4.)
    wave, cursor, stopped = c.pending_wave(plan, 0, by_id, configs(), set(), exposure=90, cap=100)
    assert len(wave) == 2 and cursor == 2 and not stopped
    wave, cursor, stopped = c.pending_wave(plan, 0, by_id, configs(), set(), exposure=93, cap=100)
    assert wave == [] and cursor == 0 and stopped
    wave, cursor, stopped = c.pending_wave(plan, 0, by_id, configs(), set(), exposure=80, cap=100)
    assert len(wave) == 4 and cursor == 4 and not stopped


def test_wave_does_not_repeat_any_registered_key(monkeypatch):
    rows = cases(1)
    plan = c.make_plan(rows)[:4]
    by_id = {r['case']: r for r in rows}
    cfg = configs()
    lookup = c.combine_lookups([('robustness', rows, plan[:2], c.base.tag)], cfg)
    monkeypatch.setattr(c.base, 'request_reserve', lambda cfg, messages: 4.)
    wave, cursor, stopped = c.pending_wave(plan, 0, by_id, cfg, set(lookup), exposure=90, cap=100)
    assert wave == plan[2:] and cursor == 4 and not stopped


def test_invalid_unpaired_wave_is_rejected():
    rows = cases(1)
    plan = c.make_plan(rows)[:2]
    plan[1] = dict(plan[1], arm='B' if plan[0]['arm'] == 'A' else 'A')
    with pytest.raises(ValueError, match='same case'):
        c.pending_wave(plan, 0, {r['case']: r for r in rows}, configs(), set(), 0, 100)


def test_aggregate_preserves_all_stages_tokens_and_reserves():
    lookup = source_fixture()
    calls, reserves = ledger_fixture(lookup)
    data = c.aggregate(calls, reserves, lookup)
    finance = data['financial']
    assert finance['answered'] == finance['attempts'] == 3
    assert finance['usage_upper_estimate_rub'] == 6
    assert finance['reserved_exposure_rub'] == pytest.approx(6.3)
    assert finance['remaining_internal_capacity'] == pytest.approx(1227 - 6.3)
    assert finance['unmapped_calls'] == finance['accounting_discrepancies'] == 0
    assert set(data['observations']) == {'main_challenge', 'supplement', 'robustness'}
    for record in finance['by_stage_model'].values():
        assert record['completion_tokens'] == 20  # Reasoning five is already included.
        assert record['reasoning_tokens'] == 5
        assert record['cache_write_tokens'] == 8  # Alias fields not summed.


def test_unknown_entries_are_accounted_not_dropped():
    lookup = source_fixture()
    calls, reserves = ledger_fixture(lookup)
    result = dict(ok=False, error='test', rub=4, request=dict(model=c.base.MODELS['astra']))
    calls.append(dict(id=4, key='unmapped-key', raw=json.dumps(result)))
    reserves.append(dict(id=4, key='unmapped-key', model=c.base.MODELS['astra'], charge=9, raw=None))
    data = c.aggregate(calls, reserves, lookup)
    assert data['financial']['usage_upper_estimate_rub'] == 10
    assert data['financial']['reserved_exposure_rub'] == pytest.approx(15.3)
    assert data['financial']['unknown_error_reserve_rub'] == 9
    assert data['financial']['unmapped_calls'] == 1 and len(data['unknown']) == 1
    assert data['financial']['accounting_discrepancies'] == 1


def test_failed_attempt_and_success_count_costs_but_keep_one_observation():
    lookup = source_fixture()
    calls, reserves = ledger_fixture(lookup)
    key = calls[0]['key']
    calls.insert(0, dict(id=0, key=key, raw=json.dumps(dict(ok=False, error='timeout', rub=.5))))
    reserves.insert(0, dict(id=0, key=key, model=c.base.MODELS['sonnet55'], charge=5, raw=json.dumps(dict(ok=False))))
    data = c.aggregate(calls, reserves, lookup)
    assert data['financial']['attempts'] == 4
    assert data['financial']['usage_upper_estimate_rub'] == 6.5
    assert len(data['observations']['main_challenge']) == 1
    assert data['observations']['main_challenge'][key]['ok']


def test_unmatched_reservation_and_duplicate_success_are_reported():
    lookup = source_fixture()
    calls, reserves = ledger_fixture(lookup)
    calls.append(dict(calls[0], id=99))
    data = c.aggregate(calls, reserves, lookup)
    types = {r['type'] for r in data['discrepancies']}
    assert 'multiple_successful_attempts' in types
    assert 'attempt_reservation_count_mismatch' in types
    assert data['financial']['usage_upper_estimate_rub'] == 7


def test_primary_gate_requires_successful_main_and_supplement_not_marker_or_repeats():
    lookup = source_fixture()
    calls, _ = ledger_fixture(lookup)
    # Continuation repeat 2 need not be answered to pass the primary coverage gate.
    primary = [r for r in calls if lookup[r['key']]['stage'] != 'robustness']
    coverage = c.require_primary_complete(primary, lookup)
    assert all(r['missing_or_error'] == 0 for r in coverage.values())
    with pytest.raises(RuntimeError, match='Primary coverage incomplete'):
        c.require_primary_complete(primary[:1], lookup)
    broken = [dict(r) for r in primary]
    broken[0]['raw'] = json.dumps(dict(ok=False, error='test'))
    with pytest.raises(RuntimeError, match='Primary coverage incomplete'):
        c.require_primary_complete(broken, lookup)
    broken[0]['raw'] = None
    with pytest.raises(RuntimeError, match='Primary coverage incomplete'):
        c.require_primary_complete(broken, lookup)


def test_primary_gate_rejects_absent_supplement_plan():
    lookup = source_fixture()
    calls, _ = ledger_fixture(lookup)
    lookup = {k: v for k, v in lookup.items() if v['stage'] != 'supplement'}
    with pytest.raises(RuntimeError, match='Both primary plans'):
        c.require_primary_complete(calls, lookup)


def test_export_separates_observations_and_combines_robustness(tmp_path, monkeypatch):
    root = tmp_path / 'study'
    for name in ('', 'supplement', 'robustness'):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'lock.json').write_text('{}', encoding='utf8')
    monkeypatch.setattr(c.base, 'RESULT', root)
    monkeypatch.setattr(c, 'RESULT', root / 'robustness')
    monkeypatch.setattr(c.base, 'validate', lambda: None)
    monkeypatch.setattr(c.supplement, 'validate', lambda: None)
    monkeypatch.setattr(c, 'validate', lambda: None)
    lookup = source_fixture()
    calls, reserves = ledger_fixture(lookup)
    monkeypatch.setattr(c, 'snapshot', lambda: (calls, reserves))
    monkeypatch.setattr(c, 'all_lookups', lambda: lookup)
    finance = c.export()
    assert finance['answered'] == 3
    assert len(c.base.read(root / 'observations.jsonl')) == 1
    assert len(c.base.read(root / 'supplement/observations.jsonl')) == 1
    merged = c.base.read(root / 'robustness/observations.jsonl')
    assert len(merged) == 2
    assert all('supplement' not in r['case'] for r in merged)
    assert (root / 'final_financial_summary.json').exists()
    (root / 'running.lock').write_text('busy')
    with pytest.raises(RuntimeError, match='experiment is active'):
        c.export()


def test_freeze_preserves_original_inputs_and_seals_metadata_only_plan(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    original_dir = root / 'results/test'
    here = root / 'experiments/test'
    original_dir.mkdir(parents=True)
    here.mkdir(parents=True)
    rows = cases(16)
    original_jobs = [dict(case=r['case'], arm=arm, model_name=name, repeat=repeat)
                     for r in rows for repeat in range(1, r['repeats'] + 1)
                     for arm in ('A', 'B') for name in c.base.MODELS]
    c.base.write_rows(original_dir / 'cases.jsonl', rows)
    c.base.write_rows(original_dir / 'plan.jsonl', original_jobs)
    (original_dir / 'filter_predictions.csv').write_text('case,defense,blocked\n', encoding='utf8')
    (original_dir / 'lock.json').write_text('{}', encoding='utf8')
    for name in ('continuation.py', 'analyze.py', 'supplement.py'):
        (here / name).write_text('# synthetic test fixture\n', encoding='utf8')
    before = {p.name: p.read_bytes() for p in original_dir.iterdir()}
    monkeypatch.setattr(c.base, 'ROOT', root)
    monkeypatch.setattr(c.base, 'RESULT', original_dir)
    monkeypatch.setattr(c.base, 'validate', lambda: None)
    monkeypatch.setattr(c, 'RESULT', original_dir / 'robustness')
    monkeypatch.setattr(c, 'HERE', here)
    c.freeze()
    c.freeze()
    assert all((original_dir / name).read_bytes() == payload for name, payload in before.items())
    new_cases = c.base.read(c.RESULT / 'cases.jsonl')
    assert all(r['repeats'] == (5 if r['kind'] == 'attack' else 1) for r in new_cases)
    assert all({k: v for k, v in old.items() if k != 'repeats'} == {k: v for k, v in new.items() if k != 'repeats'}
               for old, new in zip(rows, new_cases))
    assert len(c.base.read(c.RESULT / 'plan.jsonl')) == 768
    assert (c.RESULT / 'README.md').exists()
    assert c.validate()['maximum_combined_original_and_robustness_calls'] == 1536
