"""Predeclared supplemental cases, sharing the original 1227-ruble ledger."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('challenge_base', HERE / 'run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
RESULT = base.RESULT / 'supplement'


def get_tag(job):
    return f"hard-challenge-2026-11-26:supplement:{job['case']}:{job['repeat']}:{job['arm']}"


def freeze():
    import joblib
    import pandas as pd
    from supplement_materials import build_cases
    base.validate()
    RESULT.mkdir(exist_ok=True)
    if (RESULT / 'lock.json').exists():
        validate()
        return
    cases = build_cases()
    assert len(cases) == 16 and len({c['scene'] for c in cases}) == 8
    old = base.read(base.RESULT / 'cases.jsonl')
    assert not {c['case'] for c in cases} & {c['case'] for c in old}
    assert not {c['document'] for c in cases} & {c['document'] for c in old}
    for c in cases:
        c['repeats'] = 1
    base.write_rows(RESULT / 'cases.jsonl', cases)
    predictions = []
    for arm, name in [('D', 'D'), ('E', 'E2.1')]:
        model, threshold = joblib.load(base.ROOT / f'models/main-v4/{name}.joblib')
        for c, score in zip(cases, model.predict_proba([r['document'] for r in cases])[:, 1]):
            predictions.append(dict(case=c['case'], defense=arm, score=float(score),
                                    threshold=float(threshold), blocked=bool(score >= threshold)))
    pd.DataFrame(predictions).to_csv(RESULT / 'filter_predictions.csv', index=False)
    jobs = [dict(case=c['case'], arm=arm, model_name=name, repeat=1)
            for c in cases for arm in ('A', 'B') for name in base.MODELS]
    base.write_rows(RESULT / 'plan.jsonl', jobs)
    paths = [HERE / 'supplement.py', HERE / 'supplement_materials.py', HERE / 'SUPPLEMENT_PROTOCOL.md',
             base.RESULT / 'lock.json', RESULT / 'cases.jsonl', RESULT / 'plan.jsonl', RESULT / 'filter_predictions.csv']
    base.write_json(RESULT / 'lock.json', dict(files={str(p.relative_to(base.ROOT)): base.digest(p) for p in paths},
                   maximum_calls=len(jobs), budget='Same shared ledger and same 1227-ruble total; no additional authorization/cap.',
                   analysis='Separate supplemental cohort; do not pool as independent with main hard-challenge.'))
    validate()


def validate():
    base.validate()
    info = json.loads((RESULT / 'lock.json').read_text(encoding='utf8'))
    for path, value in info['files'].items():
        if base.digest(base.ROOT / path) != value:
            raise ValueError('Frozen supplement changed: ' + path)


def lookup():
    cases = {r['case']: r for r in base.read(RESULT / 'cases.jsonl')}
    out = {}
    for job in base.read(RESULT / 'plan.jsonl'):
        c = cases[job['case']]
        key, _ = base.identity(base.config(job['model_name']), get_tag(job), base.prepare(c, c['question'], job['arm']))
        out[key] = job
    return out


def export():
    entries, exposure = base.ledger()
    original = base.observation_lookup()
    extra = lookup()
    all_jobs = original | extra
    observations = []
    totals = {}
    for key, raw in entries:
        job = all_jobs[key]
        result = json.loads(raw) if raw else dict(ok=False, error='pending')
        scope = 'supplement' if key in extra else 'main_challenge'
        k = scope + '/' + job['model_name']
        row = totals.setdefault(k, dict(attempts=0, answered=0, errors=0, prompt_tokens=0,
                                       completion_tokens=0, estimated_rub=0.0))
        row['attempts'] += 1
        row['answered' if result.get('ok') else 'errors'] += 1
        row['estimated_rub'] += result.get('rub', 0)
        for field in ('prompt_tokens', 'completion_tokens'):
            row[field] += result.get('usage', {}).get(field, 0)
        if key in extra:
            observations.append(dict(job, **result))
    base.write_rows(RESULT / 'observations.jsonl', observations)
    finance = dict(by_stage_model=totals, usage_upper_estimate_rub=sum(r['estimated_rub'] for r in totals.values()),
                   reserved_exposure_rub=exposure, limit_rub=base.CAP,
                   remaining_internal_capacity=base.CAP-exposure, billing_actual_rub=None,
                   actual_account_balance=None, note='Combined usage estimates, not a provider invoice. Same shared cap.')
    base.write_json(base.RESULT / 'combined_financial_summary.json', finance)
    return finance


def invoke(job, case):
    client = base.Client(base.RESULT / 'budget.sqlite', base.config(job['model_name']))
    try:
        return client.call(get_tag(job), base.prepare(case, case['question'], job['arm']))
    except RuntimeError as exc:
        return dict(ok=False, error=str(exc))
    finally:
        client.db.close(); client.http.close()


def run():
    validate()
    if not os.environ.get('API_KEY'):
        raise RuntimeError('No API_KEY; no request sent')
    if not (base.RESULT / 'run_completion.json').exists():
        raise RuntimeError('Finish original planned queue before the supplement')
    guard = base.RESULT / 'running.lock'
    with guard.open('x', encoding='utf8') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'stage': 'supplement'}))
    stopped = None
    try:
        cases = {r['case']: r for r in base.read(RESULT / 'cases.jsonl')}
        jobs = base.read(RESULT / 'plan.jsonl')
        keys = lookup()
        reverse = {tuple((j['case'], j['arm'], j['model_name'], j['repeat'])): key for key, j in keys.items()}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i in range(0, len(jobs), 2):
                old, exposure = base.ledger()
                seen = {k for k, _ in old}
                pair = [j for j in jobs[i:i+2] if reverse[(j['case'],j['arm'],j['model_name'],j['repeat'])] not in seen]
                reserve = sum(base.request_reserve(base.config(j['model_name']), base.prepare(
                    cases[j['case']], cases[j['case']]['question'], j['arm'])) for j in pair)
                if exposure + reserve > base.CAP:
                    stopped = 'budget_reserve_for_model_pair'
                    break
                futures = [pool.submit(invoke, j, cases[j['case']]) for j in pair]
                results = [f.result() for f in futures]
                if pair:
                    finance = export()
                    print(json.dumps(dict(pair=i//2+1, pairs=len(jobs)//2,
                         estimated_rub=round(finance['usage_upper_estimate_rub'], 2),
                         reserved_rub=round(finance['reserved_exposure_rub'], 2),
                         errors=[r.get('error') for r in results if not r.get('ok')]), ensure_ascii=False), flush=True)
    finally:
        guard.unlink()
        finance = export()
    base.write_json(RESULT / 'completion.json', dict(stopping_reason=stopped, finance=finance))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['freeze', 'run', 'export'])
    stage = p.parse_args().stage
    if stage == 'freeze': freeze()
    elif stage == 'run': run()
    else: print(json.dumps(export(), ensure_ascii=False, indent=2))
