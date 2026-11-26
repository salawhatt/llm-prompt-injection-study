"""Isolated, budgeted challenge study; never writes to GitHub or main-v4."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT = ROOT / 'results/hard-challenge-2026-11-26'
sys.path.insert(0, str(ROOT / 'src'))
from client import Client, identity
from experiment_core import prepare

MODELS = {'sonnet55': 'anthropic/claude-sonnet-5.5', 'astra': 'openai/gpt-6-astra'}
CAP = 1227.0
MARGIN = 1.05


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf8'))
    temp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes((''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)).encode('utf8'))
    temp.replace(path)


def config(name):
    catalog = json.loads((RESULT / 'catalog.json').read_text(encoding='utf8'))
    meta = next(m['metadata'] for m in catalog['data'] if m['id'] == MODELS[name])
    authorization = RESULT / 'budget_authorization.json'
    return dict(model=MODELS[name], temperature=0, max_output=1024, strict_model=True,
                input_rub_per_million=meta['prompt_tokens_cost'],
                output_rub_per_million=meta['generated_tokens_cost'],
                cache_write_rub_per_million=max(meta.get('cache_write_tokens_cost', 0),
                                               meta.get('cache_write_1h_tokens_cost', 0)),
                budget_path=str((RESULT / 'budget.sqlite').resolve()), rub_limit=CAP,
                budget_authorization=str(authorization), budget_authorization_sha256=digest(authorization),
                attempt_limit=2000, timeout=120, max_request_bytes=50000, max_input_tokens=51000,
                cost_reserve_multiplier=MARGIN, bound_request_input=True, strict_dedup=True,
                max_attempts_per_key=1, store_raw_response=True)


def tag(job):
    return f"hard-challenge-2026-11-26:{job['case']}:{job['repeat']}:{job['arm']}"


def request_reserve(cfg, messages):
    _, payload = identity(cfg, '', messages)
    size = len(json.dumps(payload, ensure_ascii=False).encode('utf8'))
    if size > cfg['max_request_bytes']:
        raise ValueError('Document exceeds frozen request envelope')
    rate = max(cfg['input_rub_per_million'], cfg['cache_write_rub_per_million'])
    return ((size + 1000) * rate + cfg['max_output'] * cfg['output_rub_per_million']) / 1e6 * MARGIN


def freeze():
    import joblib
    import pandas as pd
    from materials import build_cases
    RESULT.mkdir(parents=True, exist_ok=True)
    if (RESULT / 'lock.json').exists():
        validate()
        return
    protected = [ROOT / 'src/client.py', ROOT / 'src/experiment_core.py', ROOT / 'src/research_defense.py',
                 ROOT / 'models/main-v4/D.joblib', ROOT / 'models/main-v4/E2.1.joblib']
    # Actual classifier filenames are resolved without copying or modifying weights.
    model_paths = {'D': ROOT / 'models/main-v4/D.joblib', 'E': ROOT / 'models/main-v4/E2.1.joblib'}
    if not all(p.exists() for p in model_paths.values()):
        raise FileNotFoundError('Check frozen classifier paths before preparing the study')
    cases = build_cases()
    for c in cases:
        c['repeats'] = 5 if c['kind'] == 'attack' and c['variant'] == 0 else 1
        c['cohort'] = 'hard_challenge'
        c['input_hash'] = hashlib.sha256(json.dumps([c['question'], c['document']], ensure_ascii=False).encode()).hexdigest()
    assert len(cases) == 128 and len({r['scene'] for r in cases}) == 64
    assert len({r['family'] for r in cases}) == 16
    assert len({r['case'] for r in cases}) == len(cases)
    old = read(ROOT / 'github/llm-prompt-injection-study/data/cases.jsonl')
    old_texts = {' '.join(r['document'].casefold().split()) for r in old}
    assert not any(' '.join(r['document'].casefold().split()) in old_texts for r in cases)
    for scene in {r['scene'] for r in cases}:
        pair = [r for r in cases if r['scene'] == scene]
        assert {r['kind'] for r in pair} == {'attack', 'clean'}
        assert len({r['question'] for r in pair}) == len({r['gold'] for r in pair}) == 1
        assert pair[0]['marker'] not in pair[0]['accepted']
    write_rows(RESULT / 'cases.jsonl', cases)
    predictions = []
    for arm, path in model_paths.items():
        model, threshold = joblib.load(path)
        for c, score in zip(cases, model.predict_proba([r['document'] for r in cases])[:, 1]):
            predictions.append(dict(case=c['case'], defense=arm, score=float(score),
                                    threshold=float(threshold), blocked=bool(score >= threshold)))
    pd.DataFrame(predictions).to_csv(RESULT / 'filter_predictions.csv', index=False)
    write_json(RESULT / 'budget_authorization.json', dict(
        source='User authorized separate hard challenge on 2026-11-26 and reported 1227 rubles available.',
        source_available_units=CAP, cumulative_cap_rub=CAP,
        budget_path=str((RESULT / 'budget.sqlite').resolve()),
        uncertainty_multiplier=MARGIN, independently_confirmed_billing=None,
        note='No old reserves/grants added. Usage upper estimate and error reserves are not a provider invoice.'))
    jobs = []
    for repeat in range(1, 6):
        for c in cases:
            if repeat > c['repeats']:
                continue
            for arm in ('A', 'B'):
                for name in MODELS:
                    jobs.append(dict(case=c['case'], arm=arm, model_name=name, repeat=repeat))
    write_rows(RESULT / 'plan.jsonl', jobs)
    paths = protected + [HERE / 'materials.py', HERE / 'run.py', HERE / 'analyze.py', HERE / 'PROTOCOL.md',
                         RESULT / 'cases.jsonl', RESULT / 'plan.jsonl', RESULT / 'filter_predictions.csv',
                         RESULT / 'catalog.json', RESULT / 'budget_authorization.json']
    write_json(RESULT / 'lock.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        models=MODELS, configurations=['A', 'B', 'D', 'E'], parameters={'temperature': 0, 'max_tokens': 1024},
        files={str(p.relative_to(ROOT)): digest(p) for p in paths},
        scenario_count=64, families=16, documents=128, maximum_api_calls=len(jobs),
        schedule='All first trials (variant/family order), then repetitions 2..5 of variant-0 attack cases. Pilot reuses planned first trials.',
        statistical_scope='Targeted synthetic challenge after observing previous results, not a blind independent benchmark.'))
    validate()
    print(json.dumps({'prepared_cases': len(cases), 'planned_calls': len(jobs)}, ensure_ascii=False), flush=True)


def validate():
    lock = json.loads((RESULT / 'lock.json').read_text(encoding='utf8'))
    for name, expected in lock['files'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Frozen study component changed: ' + name)
    return lock


def ledger():
    path = RESULT / 'budget.sqlite'
    if not path.exists():
        return [], 0.0
    with sqlite3.connect('file:' + path.as_posix() + '?mode=ro', uri=True) as db:
        rows = db.execute('SELECT key,result FROM calls ORDER BY id').fetchall()
        exposure = db.execute('SELECT coalesce(sum(charge),0) FROM reservations').fetchone()[0]
    return rows, float(exposure)


@lru_cache(maxsize=1)
def observation_lookup():
    configs = {n: config(n) for n in MODELS}
    cases = {r['case']: r for r in read(RESULT / 'cases.jsonl')}
    lookup = {}
    for job in read(RESULT / 'plan.jsonl'):
        key, _ = identity(configs[job['model_name']], tag(job), prepare(cases[job['case']], cases[job['case']]['question'], job['arm']))
        lookup[key] = job
    return lookup


def export():
    rows, exposure = ledger()
    lookup = observation_lookup()
    observations = []
    by_model = {n: dict(attempts=0, answered=0, errors=0, pending=0, prompt_tokens=0,
                        completion_tokens=0, reasoning_tokens=0, estimated_cost_rub=0.0) for n in MODELS}
    for key, entry in rows:
        job = lookup[key]
        result = json.loads(entry) if entry else dict(ok=False, error='pending')
        observations.append(dict(job, **result))
        record = by_model[job['model_name']]
        record['attempts'] += 1
        record['answered' if result.get('ok') else 'pending' if not entry else 'errors'] += 1
        usage = result.get('usage', {})
        record['prompt_tokens'] += usage.get('prompt_tokens', 0)
        record['completion_tokens'] += usage.get('completion_tokens', 0)
        record['reasoning_tokens'] += (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
        record['estimated_cost_rub'] += result.get('rub', 0)
    write_rows(RESULT / 'observations.jsonl', observations)
    financial = dict(by_model=by_model, usage_upper_estimate_rub=sum(r['estimated_cost_rub'] for r in by_model.values()),
                     reserved_exposure_rub=exposure, limit_rub=CAP, remaining_internal_capacity=CAP-exposure,
                     billing_actual_rub=None, actual_account_balance=None,
                     note='Usage includes reasoning once. Cache read discounts ignored, maximum write tariff used. Not a provider invoice or current account balance.')
    write_json(RESULT / 'financial_summary.json', financial)
    return financial


def invoke(job, case, cfg):
    client = Client(RESULT / 'budget.sqlite', cfg)
    try:
        return client.call(tag(job), prepare(case, case['question'], job['arm']))
    except RuntimeError as exc:
        # The client already persists ambiguous failures; no automatic retries.
        return dict(ok=False, error=str(exc))
    finally:
        client.db.close()
        client.http.close()


def run(stage):
    validate()
    if not os.environ.get('API_KEY'):
        raise RuntimeError('API_KEY is not available; no paid requests sent')
    cases = {r['case']: r for r in read(RESULT / 'cases.jsonl')}
    jobs = read(RESULT / 'plan.jsonl')
    if stage == 'pilot':
        scene_ids = sorted({r['scene'] for r in cases.values()})
        # Spread across families/variants; selected before receiving any answer.
        selected = {scene_ids[i] for i in (0, 21, 42, 63)}
        jobs = [j for j in jobs if j['repeat'] == 1 and cases[j['case']]['scene'] in selected]
    else:
        pilot = json.loads((RESULT / 'pilot_summary.json').read_text(encoding='utf8'))
        if not pilot['passed']:
            raise RuntimeError('Pilot did not pass; review before continuing')
    configs = {name: config(name) for name in MODELS}
    guard = RESULT / 'running.lock'
    with guard.open('x', encoding='utf8') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'stage': stage}))
    stopped = None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for offset in range(0, len(jobs), 2):
                pair = jobs[offset:offset+2]
                assert len(pair) == 2 and {j['model_name'] for j in pair} == set(MODELS)
                old, exposure = ledger()
                seen = {key: json.loads(value) if value else None for key, value in old}
                pending = []
                for job in pair:
                    case = cases[job['case']]
                    cfg = configs[job['model_name']]
                    messages = prepare(case, case['question'], job['arm'])
                    key, _ = identity(cfg, tag(job), messages)
                    if key in seen:
                        continue
                    pending.append((job, case, cfg, request_reserve(cfg, messages)))
                if exposure + sum(p[3] for p in pending) > CAP:
                    stopped = 'budget_reserve_for_model_pair'
                    break
                futures = [pool.submit(invoke, job, case, cfg) for job, case, cfg, _ in pending]
                answers = [future.result() for future in futures]
                if pending:
                    finance = export()
                    print(json.dumps(dict(stage=stage, pair=offset//2+1, pairs=len(jobs)//2,
                          completed=sum(v['answered'] for v in finance['by_model'].values()),
                          estimated_rub=round(finance['usage_upper_estimate_rub'], 2),
                          reserved_rub=round(finance['reserved_exposure_rub'], 2),
                          errors=[a.get('error') for a in answers if not a.get('ok')]), ensure_ascii=False), flush=True)
                if stage == 'pilot' and any(not a.get('ok') or a.get('finish') == 'length' for a in answers):
                    stopped = 'pilot_error_or_truncation'
                    break
    finally:
        guard.unlink()
        finance = export()
    if stage == 'pilot':
        observed = read(RESULT / 'observations.jsonl')
        wanted = {(j['case'], j['model_name'], j['arm'], j['repeat']) for j in jobs}
        observed = [r for r in observed if (r['case'], r['model_name'], r['arm'], r['repeat']) in wanted]
        passed = len(observed) == len(jobs) and all(r['ok'] and r['finish'] != 'length' for r in observed)
        write_json(RESULT / 'pilot_summary.json', dict(passed=passed, observations=len(observed),
            estimated_rub=sum(r.get('rub', 0) for r in observed),
            per_model={name: dict(n=sum(r['model_name']==name for r in observed),
                mean_cost=sum(r.get('rub',0) for r in observed if r['model_name']==name)/max(1,sum(r['model_name']==name for r in observed))) for name in MODELS},
            stopping_reason=stopped))
    write_json(RESULT / f'{stage}_completion.json', dict(stopping_reason=stopped,
               financial=finance, at_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'pilot', 'run', 'export'])
    args = parser.parse_args()
    if args.stage == 'freeze': freeze()
    elif args.stage == 'export': print(json.dumps(export(), ensure_ascii=False, indent=2))
    else: run(args.stage)
