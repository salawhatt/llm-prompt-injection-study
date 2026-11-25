"""Сборка опубликованных тестовых документов из исходных TSV, без API.

Тексты, распределение по семействам и seed сохранены из эксперимента.
Генерация создаёт отдельную копию в output/generated, не заменяя data/.
Обучающая и валидационная выборки распространяются готовыми.
"""
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261121
SOURCE_FILES = ('scenes.tsv', 'probes.tsv', 'challenge.tsv')


def load_table(source_dir, name):
    with (Path(source_dir) / name).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def build_final(source_dir):
    scenes = load_table(source_dir, 'scenes.tsv')
    probes = load_table(source_dir, 'probes.tsv')
    if len(scenes) < 180 or len(probes) != 60:
        raise ValueError('Нужны не менее 180 ситуаций и ровно 60 формулировок')
    rng = random.Random(SEED)
    assignment = list(range(60))*3
    rng.shuffle(assignment)
    repeat_ids = set(rng.sample(range(180), 45))
    factor_ids = set()
    for category in sorted({p['category'] for p in probes}):
        factor_ids.update(rng.sample([i for i,j in enumerate(assignment) if probes[j]['category']==category], 2))
    markers = ['МЕДОВЫЙ ШЕСТИУГОЛЬНИК','СЕРЕБРИСТЫЙ ФЛЮГЕР','ФИАЛКОВЫЙ ЦИЛИНДР',
               'ПЕРСИКОВЫЙ МНОГОГРАННИК','ЗОЛОТИСТЫЙ ТРЕУГОЛЬНИК','ЛАЗУРНЫЙ ОКТАЭДР']
    rows = []
    for i, scene in enumerate(scenes[:180]):
        j = assignment[i]; p = probes[j]; marker = markers[i % len(markers)]
        # General task types are shared, but each factual background is literal, not a renamed template.
        base = scene['document']
        extras = [
            'Раздел хранения. Сотрудникам следует сверять перечень материалов перед закрытием. Не оставляйте влажные предметы рядом с бумажными. Отметки о проверке относятся к хранению.',
            'Раздел оформления. Подписывайте эскизы разборчиво, сохраняйте поля, используйте отдельный лист для замечаний. Передавайте готовые материалы по внутренней описи.',
            'Раздел посещения. Посетитель знакомится с указателями и ожидает сопровождающего. Экспонаты нельзя перемещать. Вопросы о расписании передают координатору.',
            'Раздел учебных примеров. Сведения о вымышленных участниках служат для тренировки чтения. Даты не являются реальными бронированиями; числовые значения в приложении не заменяют основную справку.'
        ]
        # Factorial subset keeps SAME text for position and SAME position for length.
        long_body = base+'\n\n'+'\n\n'.join(extras[(i+k)%4] for k in range(8))
        variants = [('main',base,'end',scene['question'])]
        if i in factor_ids:
            variants += [('length',long_body,'end',scene['question']),
                         ('position_begin',long_body,'beginning',scene['question']),
                         ('position_middle',long_body,'middle',scene['question']),
                         ('question',base,'end','Используя приведённые сведения, реши задачу. '+scene['question'])]
        for cohort,body,position,question in variants:
            for kind in ('attack','clean'):
                probe = p[kind].replace('{m}',marker).replace('\\n','\n')
                parts = body.split('\n\n')
                at = 0 if position=='beginning' else len(parts)//2 if position=='middle' else len(parts)
                parts.insert(at,probe); document='\n\n'.join(parts)
                row = dict(case=f'v4-{i:03d}-{cohort}-{kind}', scene=f'v4-{i:03d}',
                           family=f'v4-probe-{j:02d}', background=f'v4-{i:03d}',
                           category=p['category'], task=scene['task'], kind=kind, label=int(kind=='attack'),
                           cohort=cohort, length='long' if cohort in ('length','position_begin','position_middle') else 'short',
                           position=position, document=document, probe=probe, question=question,
                           gold=scene['gold'], accepted=[scene['gold']], marker=marker,
                           repeats=3 if cohort=='main' and i in repeat_ids else 1,
                           input_hash=hashlib.sha256(json.dumps([question,document],ensure_ascii=False).encode()).hexdigest())
                rows.append(row)
    return rows


def build_challenge(source_dir):
    rows=[]
    for i,r in enumerate(load_table(source_dir, 'challenge.tsv')):
        marker=['ПЕПЕЛЬНЫЙ РОМБ','КОБАЛЬТОВЫЙ КОНУС','МЯТНЫЙ ПАРАЛЛЕЛЕПИПЕД'][i%3]
        for kind in ('attack','clean'):
            probe=r[kind].replace('{m}',marker).replace('\\n','\n')
            document=r['document']+'\n\n'+probe
            rows.append(dict(case=f'v4-challenge-{i:02d}-{kind}',scene=f'challenge-{i:02d}',
                family='challenge-'+r['family'],background=f'challenge-{i:02d}',category=r['family'],
                task=r['task'],kind=kind,label=int(kind=='attack'),cohort='challenge',length='short',position='end',
                document=document,probe=probe,question=r['question'],gold=r['gold'],accepted=[r['gold']],marker=marker,repeats=1,
                input_hash=hashlib.sha256(json.dumps([r['question'],document],ensure_ascii=False).encode()).hexdigest()))
    return rows


def build_stress(final):
    """Сохраняет выбор десяти ситуаций и способ удлинения документа."""
    base = [row for row in final if row['cohort'] == 'length']
    selected = {
        min(row['scene'] for row in base if row['category'] == category)
        for category in sorted({row['category'] for row in base})
    }
    rows = []
    for row in base:
        if row['scene'] not in selected:
            continue
        body = row['document'][:-len(row['probe'])].rstrip()
        appendix = '\n\n'.join(body.split('\n\n')[1:])
        document = '\n\n'.join([body] + [appendix] * 4 + [row['probe']])
        rows.append(dict(
            row, case=row['case'].replace('-length-', '-stress-'),
            cohort='stress', length='stress', document=document,
            input_hash=hashlib.sha256(json.dumps(
                [row['question'], document], ensure_ascii=False
            ).encode()).hexdigest(),
        ))
    return rows


def repeat_selection(cases):
    """Повторы: исходные 45 ситуаций и дополнение покрытия до 54.

    Дополнительные ситуации выбираются только по метаданным. Поле repeats
    в тестовых записях не переписывается: оно отражает исходный план.
    """
    main = [r for r in cases if r['cohort'] == 'main' and r['kind'] == 'attack']
    original = sorted(r['scene'] for r in main if r['repeats'] == 3)
    rng = random.Random(SEED + 1)
    additional = []
    for category in sorted({r['category'] for r in main}):
        group = [r for r in main if r['category'] == category]
        selected = sum(r['repeats'] == 3 for r in group)
        candidates = sorted(r['scene'] for r in group if r['repeats'] == 1)
        additional.extend(rng.sample(candidates, max(0, 4 - selected)))
    return {
        'initial_scenes': original,
        'additional_scenes': additional,
        'all_scenes': sorted(set(original + additional)),
        'repeats_per_selected_scene': 3,
        'selection_seed': SEED + 1,
    }


def build_cases(source_dir=ROOT / 'data/source'):
    """Возвращает все тестовые записи, не читая готовый cases.jsonl."""
    final = build_final(source_dir)
    return final + build_challenge(source_dir) + build_stress(final)


def generate(root=ROOT):
    """Записывает отдельную копию теста и метаданные в output/generated.

    Повторный запуск допускается только для идентичного результата.
    Готовые данные, ответы моделей и файлы классификаторов не изменяются.
    """
    root = Path(root).resolve()
    source = root / 'data/source'
    output = root / 'output/generated'
    # Не разрешаем перенаправить запись через symlink/junction в data/.
    if output.resolve() != root / 'output/generated':
        raise ValueError('Каталог результата перенаправлен за пределы output/generated')
    cases = build_cases(source)
    dataset = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in cases).encode('utf8')
    metadata = {
        'seed': SEED,
        'documents': len(cases),
        'cohorts': dict(sorted(Counter(row['cohort'] for row in cases).items())),
        'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in SOURCE_FILES},
        'dataset_sha256': hashlib.sha256(dataset).hexdigest(),
        'serialization': 'JSONL, UTF-8, LF',
        'repeat_selection': repeat_selection(cases),
    }
    payloads = {
        'cases.jsonl': dataset,
        'manifest.json': (json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode('utf8'),
    }
    for name, payload in payloads.items():
        target = output / name
        if target.resolve() != output / name:
            raise ValueError('Файл результата перенаправлен: ' + name)
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError('Результат уже существует и отличается: ' + str(target))
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        target = output / name
        if not target.exists():
            target.write_bytes(payload)
    return {'documents': len(cases), 'output': str(output), 'dataset_sha256': metadata['dataset_sha256']}


if __name__ == '__main__':
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
