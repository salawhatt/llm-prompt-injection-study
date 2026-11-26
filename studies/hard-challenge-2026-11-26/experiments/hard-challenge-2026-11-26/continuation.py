"""Frozen robustness continuation and universal export of the shared ledger.

The original source files and experimental observations remain separate. Only
explicit `run` invokes the API; `freeze` and `export` are local operations.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3

HERE = Path(__file__).resolve().parent


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = import_file('robustness_base', HERE / 'run.py')
supplement = import_file('robustness_supplement', HERE / 'supplement.py')
RESULT = base.RESULT / 'robustness'
MAX_WORKERS = 4

PROTOCOL = '''# Проверка устойчивости при повторных вызовах

Это заранее зафиксированное продолжение отдельной сложной серии. Новые тексты,
обучение фильтров и изменение критериев оценки не предусмотрены. В первоначальном
плане пять вызовов были назначены только атакующим документам варианта 0 (поиск
факта). Здесь такой же объём повторов назначается атакующим вариантам 1, 2 и 3
(арифметика, сравнение и извлечение). Все 16 семейств включены без отбора по ответам.

Новые обращения: повторения 2-5, сначала номер повтора, затем вариант 1/2/3,
семейство, конфигурация A/B и две модели. Всего 768 дополнительных запланированных
обращений. Успешные первые ответы и повторы варианта 0 используются из исходного
журнала, без повторной оплаты. Обычные документы имеют только первый вызов.
В `cases.jsonl` скопированы все исходные 128 документов с `repeats=5` для атакующих
и `repeats=1` для обычных; остальные поля сохранены. `plan.jsonl` содержит только
новые обращения. `observations.jsonl` объединяет исходные и новые ответы только
на эти 128 документов. Дополнительные восемь ситуаций supplement сюда не входят.

Общий бюджет остаётся 1227 рублей для основной серии, supplement и продолжения.
Используется тот же `budget.sqlite`, та же авторизация и ставки. Перед волной
резервируется максимальная стоимость всех новых обращений; одновременно выполняются
до четырёх запросов (две пары моделей). Если две пары не помещаются в резерв,
проверяется одна пара. Если не помещается даже она, серия останавливается.
Внутренний атомарный резерв клиента дополнительно проверяет общий предел.
Параметры генерации неизменны: temperature=0, max_tokens=1024. Запас стоимости 5%.
Ключи ранее зарегистрированных попыток, включая ошибки и неопределённые исходы,
автоматически не отправляются повторно.

Запуск разрешён только после завершения работы исполнителей основной серии и
supplement и успешного получения всех 512 первых ответов основной серии и всех
64 первых ответов supplement. Одного наличия completion.json недостаточно:
при ошибках или пропусках первичных ответов продолжение запрещено. Уже назначенные
повторы 2-5 варианта 0 не входят в этот обязательный первичный порог.
Общий `running.lock` запрещает одновременную работу исполнителей.
Часть плана может остаться без ответов при достижении бюджета. Это явно отражается
в анализе; пропуск не считается устойчивым неуспехом атаки. Повторы одного документа
не являются новыми независимыми ситуациями. Семейства и первый вызов анализируются
как в зафиксированном `analyze.py`; проверка вариативности представляется отдельно.

## Команды

Из корня проекта:

```text
python experiments/hard-challenge-2026-11-26/continuation.py freeze
python experiments/hard-challenge-2026-11-26/continuation.py run
python experiments/hard-challenge-2026-11-26/continuation.py export
python experiments/hard-challenge-2026-11-26/analyze.py --directory results/hard-challenge-2026-11-26/robustness
```

Для `run` требуется уже настроенная переменная окружения API_KEY. Ключ в файлах
продолжения не хранится. `freeze` и `export` не выполняют API-запросов.

После расширений общий журнал выгружается только новой командой `continuation.py
export`. Старые `run.py export` и `supplement.py export` не знают ключей продолжения
и больше не подходят для общего журнала. Новый экспорт восстанавливает исходные
наблюдения по исходному плану, supplement отдельно, объединённые наблюдения
robustness отдельно и `final_financial_summary.json` для всех этапов. Все попытки
учитываются в стоимости, даже если для анализа выбран один успешный ответ на ключ.
Неизвестные ключи и расхождения фиксируются отдельно, а не исчезают из суммы.
Оценка usage, резервы и фактический счёт Cloud.ru остаются разными показателями.
'''


def make_plan(original_cases):
    """Metadata-only selection: no response content or detector score consulted."""
    selected = sorted((c for c in original_cases if c['kind'] == 'attack' and int(c['variant']) in (1, 2, 3)),
                      key=lambda c: (int(c['variant']), c['family'], c['scene']))
    return [dict(case=c['case'], arm=arm, model_name=name, repeat=repeat)
            for repeat in range(2, 6) for c in selected for arm in ('A', 'B') for name in base.MODELS]


def freeze():
    base.validate()
    RESULT.mkdir(parents=True, exist_ok=True)
    if (RESULT / 'lock.json').exists():
        validate()
        return
    original = base.read(base.RESULT / 'cases.jsonl')
    cases = [dict(c, repeats=5 if c['kind'] == 'attack' else 1) for c in original]
    jobs = make_plan(original)
    if len(cases) != 128 or len(jobs) != 768:
        raise ValueError('Continuation expects the frozen 64-situation, 16-family dataset')
    old_keys = {(j['case'], j['arm'], j['model_name'], j['repeat']) for j in base.read(base.RESULT / 'plan.jsonl')}
    new_keys = {(j['case'], j['arm'], j['model_name'], j['repeat']) for j in jobs}
    if len(new_keys) != len(jobs) or old_keys & new_keys:
        raise ValueError('Continuation duplicates already planned calls')
    # Never overwrite a partially prepared different protocol/data silently.
    payloads = {
        'cases.jsonl': ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases).encode('utf8'),
        'plan.jsonl': ''.join(json.dumps(j, ensure_ascii=False) + '\n' for j in jobs).encode('utf8'),
        'filter_predictions.csv': (base.RESULT / 'filter_predictions.csv').read_bytes(),
        'README.md': PROTOCOL.encode('utf8'),
    }
    for name, payload in payloads.items():
        path = RESULT / name
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError('Different preparation file already exists: ' + name)
    for name, payload in payloads.items():
        (RESULT / name).write_bytes(payload)
    paths = [HERE / 'continuation.py', HERE / 'analyze.py', HERE / 'supplement.py', base.RESULT / 'lock.json']
    paths += [RESULT / name for name in payloads]
    base.write_json(RESULT / 'lock.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), maximum_additional_calls=len(jobs),
        maximum_combined_original_and_robustness_calls=len(old_keys | new_keys),
        family_count=16, cases=len(cases), additional_attack_situations=48,
        generation=dict(temperature=0, max_tokens=1024), max_concurrent_calls=MAX_WORKERS,
        shared_budget_path=str((base.RESULT / 'budget.sqlite').resolve()), shared_cap_rub=base.CAP,
        files={str(p.relative_to(base.ROOT)): base.digest(p) for p in paths},
        selection='All variant-1/2/3 attack cases, repetitions 2..5; no outcome-based selection.'))
    validate()
    print(json.dumps({'prepared_cases': len(cases), 'additional_planned_calls': len(jobs)}, ensure_ascii=False), flush=True)


def validate():
    base.validate()
    lock = json.loads((RESULT / 'lock.json').read_text(encoding='utf8'))
    for name, expected in lock['files'].items():
        if base.digest(base.ROOT / name) != expected:
            raise RuntimeError('Frozen continuation component changed: ' + name)
    return lock


def combine_lookups(sources, configs=None):
    """sources: iterable of (stage, cases, jobs, tag_function). Reject collisions."""
    configs = configs or {name: base.config(name) for name in base.MODELS}
    lookup = {}
    for stage, cases, jobs, tag_function in sources:
        case_map = {c['case']: c for c in cases}
        for job in jobs:
            case = case_map[job['case']]
            if job['model_name'] not in base.MODELS or job['arm'] not in ('A', 'B'):
                raise ValueError('Unexpected model or configuration in frozen plan')
            key, request = base.identity(configs[job['model_name']], tag_function(job),
                                          base.prepare(case, case['question'], job['arm']))
            if key in lookup:
                raise ValueError('Request-key collision between plans: ' + key)
            lookup[key] = dict(stage=stage, job=job, request=request)
    return lookup


_LOOKUP_CACHE = {}


def all_lookups():
    """Include a stage only when its plan has been frozen, never guess by response."""
    directories = [('main_challenge', base.RESULT, base.tag),
                   ('supplement', base.RESULT / 'supplement', supplement.get_tag),
                   ('robustness', RESULT, base.tag)]
    fingerprint = tuple((str(path), base.digest(path / 'lock.json'))
                        for _, path, _ in directories if (path / 'lock.json').exists())
    if fingerprint not in _LOOKUP_CACHE:
        sources = []
        for stage, directory, tag_function in directories:
            if not (directory / 'lock.json').exists():
                continue
            sources.append((stage, base.read(directory / 'cases.jsonl'), base.read(directory / 'plan.jsonl'), tag_function))
        _LOOKUP_CACHE.clear()
        _LOOKUP_CACHE[fingerprint] = combine_lookups(sources)
    return _LOOKUP_CACHE[fingerprint]


def snapshot():
    path = base.RESULT / 'budget.sqlite'
    if not path.exists():
        return [], []
    with sqlite3.connect('file:' + path.as_posix() + '?mode=ro', uri=True, timeout=60) as db:
        db.execute('BEGIN')
        calls = [dict(id=i, key=key, raw=result) for i, key, result in db.execute('SELECT id,key,result FROM calls ORDER BY id')]
        reserves = [dict(id=i, model=model, key=key, charge=float(charge), raw=result)
                    for i, model, key, charge, result in db.execute('SELECT id,model,key,charge,result FROM reservations ORDER BY id')]
        db.commit()
    return calls, reserves


def aggregate(calls, reserves, lookup):
    """Pure accounting. All attempts/reserves remain represented, including unknowns."""
    totals = {}
    observations_by_stage = defaultdict(dict)
    attempts_by_stage = defaultdict(list)
    unknown = []
    discrepancies = []
    counts = Counter(c['key'] for c in calls)
    reserve_counts = Counter(r['key'] for r in reserves)
    reverse_models = {model: name for name, model in base.MODELS.items()}

    def bucket(stage, model_name):
        return totals.setdefault(stage + '/' + model_name, dict(stage=stage, model_name=model_name,
            attempts=0, answered=0, errors=0, pending=0, prompt_tokens=0, completion_tokens=0,
            reasoning_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
            estimated_cost_rub=0.0, reserved_exposure_rub=0.0, unknown_error_reserve_rub=0.0))

    for call in calls:
        key = call['key']
        result = json.loads(call['raw']) if call['raw'] is not None else dict(ok=False, error='pending')
        meta = lookup.get(key)
        if meta:
            stage, job = meta['stage'], meta['job']
            model_name = job['model_name']
            if 'request' in result and result['request'] != meta['request']:
                discrepancies.append(dict(type='request_mismatch', key=key, attempt_id=call['id']))
            if result.get('ok') and result.get('model') != base.MODELS[model_name]:
                discrepancies.append(dict(type='model_mismatch', key=key, attempt_id=call['id']))
            observation = dict(job, **result)
            previous = observations_by_stage[stage].get(key)
            if previous is None or not previous.get('ok'):
                observations_by_stage[stage][key] = observation
            elif result.get('ok'):
                discrepancies.append(dict(type='multiple_successful_attempts', key=key, attempt_id=call['id']))
        else:
            stage = 'unmapped'
            model_name = reverse_models.get((result.get('request') or {}).get('model'), 'unknown')
            unknown.append(dict(attempt_id=call['id'], key=key, result=result))
        record = bucket(stage, model_name)
        record['attempts'] += 1
        record['answered' if result.get('ok') else 'pending' if call['raw'] is None else 'errors'] += 1
        usage = result.get('usage') or {}
        details = usage.get('prompt_tokens_details') or {}
        for field in ('prompt_tokens', 'completion_tokens'):
            record[field] += usage.get(field, 0)
        record['reasoning_tokens'] += (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
        record['cache_read_tokens'] += details.get('cached_tokens', 0)
        record['cache_write_tokens'] += max(details.get(k, 0) for k in ('cache_write_tokens', 'cache_creation_tokens', 'cached_creation_tokens'))
        record['estimated_cost_rub'] += result.get('rub', 0) or 0
        attempts_by_stage[stage].append(dict(attempt_id=call['id'], key=key, result=result))
    for reservation in reserves:
        key = reservation['key']
        meta = lookup.get(key)
        stage = meta['stage'] if meta else 'unmapped'
        model_name = meta['job']['model_name'] if meta else reverse_models.get(reservation['model'], 'unknown')
        record = bucket(stage, model_name)
        record['reserved_exposure_rub'] += reservation['charge']
        result = json.loads(reservation['raw']) if reservation['raw'] is not None else None
        if result is None or not result.get('ok'):
            record['unknown_error_reserve_rub'] += reservation['charge']
        if key not in lookup:
            discrepancies.append(dict(type='unmapped_reservation', key=key, reservation_id=reservation['id']))
        elif reservation['model'] != base.MODELS[model_name]:
            discrepancies.append(dict(type='reservation_model_mismatch', key=key, reservation_id=reservation['id']))
    for key in set(counts) | set(reserve_counts):
        if counts[key] != reserve_counts[key]:
            discrepancies.append(dict(type='attempt_reservation_count_mismatch', key=key,
                                      attempts=counts[key], reservations=reserve_counts[key]))
    exposure = sum(r['charge'] for r in reserves)
    estimated = sum(r['estimated_cost_rub'] for r in totals.values())
    accounted_exposure = sum(r['reserved_exposure_rub'] for r in totals.values())
    if not math.isclose(exposure, accounted_exposure, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError('Accounting failed to preserve the reservation total')
    financial = dict(by_stage_model=totals, attempts=len(calls),
        answered=sum(r['answered'] for r in totals.values()),
        errors=sum(r['errors'] for r in totals.values()), pending=sum(r['pending'] for r in totals.values()),
        usage_upper_estimate_rub=estimated, reserved_exposure_rub=exposure,
        unknown_error_reserve_rub=sum(r['unknown_error_reserve_rub'] for r in totals.values()),
        limit_rub=base.CAP, remaining_internal_capacity=base.CAP - exposure,
        unmapped_calls=len(unknown), accounting_discrepancies=len(discrepancies),
        all_ledger_entries_accounted=True, billing_actual_rub=None, actual_account_balance=None,
        note='All calls and reservations across stages. Reasoning included once in completion tokens. '
             'Cache reads priced conservatively by the client. Not a provider invoice or account balance.')
    return dict(financial=financial, observations=observations_by_stage, attempts=attempts_by_stage,
                unknown=unknown, discrepancies=discrepancies)


def export(internal=False, strict=True):
    if not internal and (base.RESULT / 'running.lock').exists():
        raise RuntimeError('An experiment is active; do not race its observation exports')
    base.validate()
    if (base.RESULT / 'supplement/lock.json').exists():
        supplement.validate()
    if (RESULT / 'lock.json').exists():
        validate()
    calls, reserves = snapshot()
    data = aggregate(calls, reserves, all_lookups())
    base.write_json(base.RESULT / 'final_financial_summary.json', data['financial'])
    base.write_json(base.RESULT / 'export_discrepancies.json', dict(
        unmapped_call_count=len(data['unknown']), discrepancies=data['discrepancies']))
    base.write_rows(base.RESULT / 'unmapped_attempts.jsonl', data['unknown'])
    if strict and (data['unknown'] or data['discrepancies']):
        raise RuntimeError('Ledger discrepancy recorded; inspect final_financial_summary and export_discrepancies before continuing')
    targets = {'main_challenge': base.RESULT, 'supplement': base.RESULT / 'supplement', 'robustness': RESULT}
    for stage, directory in targets.items():
        if not (directory / 'lock.json').exists():
            continue
        base.write_rows(directory / 'attempts_universal.jsonl', data['attempts'][stage])
        observations = list(data['observations'][stage].values())
        if stage == 'robustness':
            # This is a combined *view* of the original 128 cases, never extra payments.
            observations = list(data['observations']['main_challenge'].values()) + observations
        base.write_rows(directory / 'observations.jsonl', observations)
    return data['financial']


def pending_wave(jobs, offset, cases, configs, seen, exposure, cap):
    """At most two complete model pairs; fall back to one if two cannot fit."""
    pending = []
    reserve = 0.
    pairs = 0
    cursor = offset
    while cursor < len(jobs) and pairs < MAX_WORKERS // 2:
        pair = jobs[cursor:cursor + 2]
        if len(pair) != 2 or {j['model_name'] for j in pair} != set(base.MODELS):
            raise ValueError('Continuation plan must contain adjacent pairs of both models')
        if len({(j['case'], j['arm'], j['repeat']) for j in pair}) != 1:
            raise ValueError('Adjacent models must receive the same case, arm and repetition')
        candidates = []
        pair_reserve = 0.
        for job in pair:
            case = cases[job['case']]
            cfg = configs[job['model_name']]
            messages = base.prepare(case, case['question'], job['arm'])
            key, _ = base.identity(cfg, base.tag(job), messages)
            if key not in seen:
                candidates.append(job)
                pair_reserve += base.request_reserve(cfg, messages)
        if not candidates:
            cursor += 2
            continue
        if exposure + reserve + pair_reserve > cap:
            return pending, cursor, not pending
        pending.extend(candidates)
        reserve += pair_reserve
        cursor += 2
        pairs += 1
    return pending, cursor, False


def require_primary_complete(calls, lookup):
    """A completion marker can denote a budget stop; inspect successful coverage."""
    expected = {stage: {key for key, meta in lookup.items()
                        if meta['stage'] == stage and meta['job']['repeat'] == 1}
                for stage in ('main_challenge', 'supplement')}
    if any(not keys for keys in expected.values()):
        raise RuntimeError('Both primary plans must be present before robustness')
    successful = {call['key'] for call in calls if call['raw'] is not None and json.loads(call['raw']).get('ok')}
    coverage = {stage: dict(expected=len(keys), successful=len(keys & successful),
                           missing_or_error=len(keys - successful)) for stage, keys in expected.items()}
    if any(row['missing_or_error'] for row in coverage.values()):
        raise RuntimeError('Primary coverage incomplete; repetitions must not replace missing first responses: '
                           + json.dumps(coverage, ensure_ascii=False))
    return coverage


def run():
    validate()
    supplement.validate()
    if not os.environ.get('API_KEY'):
        raise RuntimeError('No API_KEY; no request sent')
    if not (base.RESULT / 'run_completion.json').exists() or not (base.RESULT / 'supplement/completion.json').exists():
        raise RuntimeError('Finish the original and supplement runners before starting robustness')
    original_calls, _ = snapshot()
    require_primary_complete(original_calls, all_lookups())
    cases = {r['case']: r for r in base.read(RESULT / 'cases.jsonl')}
    jobs = base.read(RESULT / 'plan.jsonl')
    configs = {name: base.config(name) for name in base.MODELS}
    guard = base.RESULT / 'running.lock'
    with guard.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), stage='robustness')))
    stopped = None
    cursor = 0
    finance = None
    try:
        finance = export(internal=True)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            while cursor < len(jobs):
                calls, reserves = snapshot()
                seen = {r['key'] for r in calls}
                exposure = sum(r['charge'] for r in reserves)
                wave, next_cursor, out_of_budget = pending_wave(jobs, cursor, cases, configs, seen, exposure, base.CAP)
                if out_of_budget:
                    stopped = 'budget_reserve_for_model_pair'
                    break
                cursor = next_cursor
                if not wave:
                    continue
                futures = {pool.submit(base.invoke, job, cases[job['case']], configs[job['model_name']]):
                           (job['case'], job['arm'], job['repeat']) for job in wave}
                remaining = Counter(futures.values())
                answers = []
                for future in as_completed(futures):
                    answers.append(future.result())
                    pair_id = futures[future]
                    remaining[pair_id] -= 1
                    if not remaining[pair_id]:
                        # SQLite commits each attempt; refresh exports after every completed pair.
                        finance = export(internal=True)
                print(json.dumps(dict(stage='robustness', planned_pairs=len(jobs) // 2,
                    processed_plan_pairs=cursor // 2, wave_calls=len(wave),
                    answered=finance['answered'], estimated_rub=round(finance['usage_upper_estimate_rub'], 2),
                    reserved_rub=round(finance['reserved_exposure_rub'], 2),
                    errors=[a.get('error') for a in answers if not a.get('ok')]), ensure_ascii=False), flush=True)
    finally:
        try:
            finance = export(internal=True)
        finally:
            guard.unlink()
    base.write_json(RESULT / 'completion.json', dict(stopping_reason=stopped,
        processed_plan_pairs=cursor // 2, planned_additional_calls=len(jobs), financial=finance,
        at_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run', 'export'])
    stage = parser.parse_args().stage
    if stage == 'freeze':
        freeze()
    elif stage == 'run':
        run()
    else:
        print(json.dumps(export(), ensure_ascii=False, indent=2))
