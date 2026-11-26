"""Package only the separate challenge study after its runners have stopped.

create: stopped-ledger audit, SQLite backup, allowlist, secret patterns, SHA256, CRC.
verify: verify an existing archive without extracting it. self-test: memory only.
No API calls, credential-file reads, model training, Git changes or implicit exports.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = 'hard-challenge-2026-11-26'
RESULT = ROOT / 'results' / STUDY
EXPORTS = ROOT / 'exports'
TARGET = EXPORTS / (STUDY + '.zip')
EXTRA_FILES = ('src/client.py', 'src/experiment_core.py', 'src/research_defense.py',
               'src/research_data.py', 'models/main-v4/D.joblib',
               'models/main-v4/E2.1.joblib', 'requirements.txt')
DENIED_PARTS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ipynb_checkpoints'}
PATTERNS = (
    ('private_key', rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    ('github_token', rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    ('api_token', rb'\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{24,}\b'),
    ('google_api_token', rb'\bAIza[0-9A-Za-z_-]{30,}\b'),
    ('jwt', rb'\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b'),
    ('literal_bearer', rb'(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{24,}'),
    ('dpapi_blob', rb'(?i)01000000d08c9ddf0115d1118c7a00c04fc297eb[0-9a-f]{40,}'),
    ('literal_credential', rb'''(?i)["']?(?:api_key|access_token|secret_key)["']?\s*[:=]\s*["'][A-Za-z0-9_./+=-]{24,}["']'''),
)

README = '''# Сохранённые результаты сложных проверок LLM

Отдельная серия от 26 ноября 2026 года: Claude Sonnet 5.5 и GPT-6 Astra,
защиты A/B/D/E. Архив содержит новый эксперимент, сохранённые ответы,
журнал SQLite, анализ, графики и используемые классификаторы.

Это архив отдельного эксперимента, а не полная копия исходного проекта.
Неполные группы и остановки по бюджету показаны в результатах. Повторы
не считаются новыми независимыми ситуациями. Данные `robustness` включают
ответы основного набора: их нельзя повторно прибавлять к числу API-вызовов.

## Воспроизведение сохранённых результатов

Распакуйте архив в отдельную папку и откройте терминал в её корне.
Используется Python 3.11. Создайте окружение, активируйте его и выполните:

```text
python -m venv .venv
python -m pip install -r requirements-analysis.txt
python experiments/hard-challenge-2026-11-26/analyze.py
python experiments/hard-challenge-2026-11-26/analyze.py --directory results/hard-challenge-2026-11-26/supplement
python experiments/hard-challenge-2026-11-26/analyze.py --directory results/hard-challenge-2026-11-26/robustness --no-plots
python experiments/hard-challenge-2026-11-26/post_analysis.py
```

Активация Windows: `.venv\\Scripts\\activate`. Linux/macOS: `source .venv/bin/activate`.
Эти команды не вызывают API и не обучают модели. Они пересчитывают таблицы
и графики по сохранённым наблюдениям. Итог: `results/hard-challenge-2026-11-26/SUMMARY_RU.md`.

Для проверки самого ZIP используйте `python experiments/hard-challenge-2026-11-26/package_results.py verify --archive ПУТЬ_К_ZIP`.
Файл `ARCHIVE_MANIFEST.json` содержит SHA256 всех остальных файлов архива.
SQLite помещён через штатный backup, без файлов WAL/SHM. Его контрольная
сумма может отличаться от физической копии исходной базы, сохраняя те же записи.

## Границы комплекта

Классификатор D использует стандартный sklearn Pipeline. E использует
`research_defense.Detector`; для загрузки сохранены `research_defense.py`,
`research_data.py` и `experiment_core.py`. Обучающие материалы прежних серий
не включены. Сохранённый расчёт метрик использует зафиксированные решения фильтров.

`preservation.json` и `validate_artifacts.py` относятся к проверке сохранности
474 файлов исходного проекта и его Git HEAD. Эти прежние файлы в архив не входят,
поэтому полная такая проверка предназначена для исходного проекта.
Исполнители API и их протоколы сохранены для проверки реализации; авторизация
содержит исходные локальные пути и не является разрешением или готовой
конфигурацией нового запуска на другом компьютере. API-ключей в архиве нет.
Для анализа достаточно команд выше, `run.ps1` и `run-extra.ps1` запускать не нужно.

Проверка секретов ищет характерные токены и закрытые ключи по шаблонам;
она не сравнивает файлы с настоящим секретом и не читает его хранилище.
Оценка стоимости по usage и резерв отличаются от фактического счёта провайдера.
'''


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    parts = path.parts
    return (bool(parts) and not path.is_absolute() and '..' not in parts
            and '\\' not in name and ':' not in name
            and not any(p in DENIED_PARTS or p.startswith('.pytest-') for p in parts)
            and not any(p.lower().startswith('.env') or p.lower().endswith('.dpapi') for p in parts)
            and path.name not in ('running.lock', 'packaging.lock')
            and not path.name.endswith(('.pyc', '.tmp', '.sqlite-wal', '.sqlite-shm')))


def scan(data, name):
    if not safe_name(name):
        raise ValueError('Excluded archive path: ' + name)
    for label, expression in PATTERNS:
        if re.search(expression, data):
            # Do not expose the matching token or surrounding text.
            raise ValueError(f'Secret-pattern match ({label}) in {name}; archive not published')


def source_files():
    files = {}
    for directory, suffixes in ((HERE, {'.py', '.md', '.ps1'}),
                                (RESULT, {'.json', '.jsonl', '.csv', '.tsv', '.md', '.txt', '.png', '.svg'})):
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(ROOT).as_posix()
            if not safe_name(relative) or not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('Source path escapes the selected directory: ' + relative)
            files[relative] = path
    for name in EXTRA_FILES:
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Missing or redirected dependency: ' + name)
        files[name] = path
    return files


def minimal_requirements():
    wanted = {'numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn', 'joblib', 'httpx'}
    lines = []
    for line in (ROOT / 'requirements.txt').read_text(encoding='utf8').splitlines():
        if line.split('==')[0] in wanted:
            lines.append(line)
    if {line.split('==')[0] for line in lines} != wanted:
        raise ValueError('Exact analysis dependency versions are missing')
    return ('\n'.join(lines) + '\n').encode('utf8')


def offline_audit():
    spec = importlib.util.spec_from_file_location('packaging_validation', HERE / 'validate_artifacts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.validate()
    if not report['passed'] or report['ledger_snapshot']['pending']:
        raise RuntimeError('Artifact validation failed or requests remain pending; archive not created')
    for relative in ('run_completion.json', 'supplement/completion.json', 'robustness/completion.json',
                     'final_financial_summary.json', 'SUMMARY_RU.md', 'export_discrepancies.json'):
        if not (RESULT / relative).is_file():
            raise RuntimeError('Finish all runners and offline exports before packaging: ' + relative)
    finance = json.loads((RESULT / 'final_financial_summary.json').read_text(encoding='utf8'))
    actual = report['ledger_snapshot']
    for key, expected in [('attempts', actual['attempts']), ('pending', 0),
                          ('usage_upper_estimate_rub', actual['estimated_cost_rub']),
                          ('reserved_exposure_rub', actual['reserved_exposure_rub'])]:
        if not math.isclose(float(finance.get(key, -1)), float(expected), rel_tol=1e-10, abs_tol=1e-8):
            raise RuntimeError('Financial export is stale: ' + key)
    if finance.get('unmapped_calls') or finance.get('accounting_discrepancies'):
        raise RuntimeError('Resolve ledger export discrepancies before packaging')
    for suffix in ('', 'supplement', 'robustness'):
        for name in ('observations.jsonl', 'cases.jsonl', 'filter_predictions.csv', 'summary.csv', 'RESULTS.md'):
            if not (RESULT / suffix / name).is_file():
                raise RuntimeError('Missing saved analysis artifact: ' + suffix + '/' + name)
    return report


def backup_sqlite(destination):
    source = RESULT / 'budget.sqlite'
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True, timeout=60)) as src:
        src.execute('PRAGMA query_only=ON')
        with closing(sqlite3.connect(destination)) as dst:
            src.backup(dst)
            if dst.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('SQLite backup integrity check failed')
            pending = dst.execute('SELECT count(*) FROM calls WHERE result IS NULL').fetchone()[0]
            if pending:
                raise RuntimeError('SQLite backup contains pending calls')
            dst.commit()


def verify_archive(archive):
    archive = Path(archive)
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or not all(safe_name(name) for name in names):
            raise ValueError('Archive has duplicate or excluded paths')
        if zipped.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        manifest = json.loads(zipped.read('ARCHIVE_MANIFEST.json'))
        expected = set(manifest['files']) | {'ARCHIVE_MANIFEST.json'}
        if set(names) != expected:
            raise ValueError('Archive membership differs from SHA256 manifest')
        for name in names:
            data = zipped.read(name)
            scan(data, name)
            if name != 'ARCHIVE_MANIFEST.json':
                record = manifest['files'][name]
                if len(data) != record['bytes'] or sha_bytes(data) != record['sha256']:
                    raise ValueError('Archive SHA256 verification failed: ' + name)
    return dict(archive=str(archive.resolve()), sha256=digest(archive), bytes=archive.stat().st_size,
                files=len(names), crc_verified=True, sha256_verified=True,
                secret_patterns_checked=True, secret_store_read=False)


def create_archive():
    # Acquiring the same exclusive guard prevents a new runner starting mid-backup.
    guard = RESULT / 'running.lock'
    if guard.exists():
        raise RuntimeError('An experiment is active (running.lock); packaging is forbidden')
    if TARGET.exists():
        raise FileExistsError('Archive already exists; preserve it before creating a different package')
    owner = json.dumps(dict(pid=os.getpid(), stage='packaging', created_utc=datetime.now(timezone.utc).isoformat()))
    with guard.open('x', encoding='utf8') as stream:
        stream.write(owner)
    try:
        audit = offline_audit()
        files = source_files()
        EXPORTS.mkdir(parents=True, exist_ok=True)
        if EXPORTS.resolve() != ROOT.resolve() / 'exports':
            raise ValueError('Export directory must stay inside this project')
        with tempfile.TemporaryDirectory(prefix='challenge-package-', dir=EXPORTS) as temporary:
            staging = Path(temporary).resolve()
            # Check the absolute temporary directory before its recursive cleanup.
            if not staging.is_relative_to(EXPORTS.resolve()):
                raise ValueError('Temporary directory escaped exports')
            sqlite_copy = staging / 'budget.sqlite'
            backup_sqlite(sqlite_copy)
            files['results/' + STUDY + '/budget.sqlite'] = sqlite_copy
            memory = {'README.md': README.encode('utf8'),
                      'requirements-analysis.txt': minimal_requirements(),
                      'PACKAGING_VALIDATION.json': (json.dumps(audit, ensure_ascii=False, indent=2) + '\n').encode('utf8')}
            manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), study=STUDY,
                scope='Separate saved-results archive, not a complete original-project checkout.',
                serialized_dependencies={'D': 'sklearn.pipeline.Pipeline', 'E': 'research_defense.Detector',
                    'E_local_imports': ['research_data', 'experiment_core']},
                sqlite_method='sqlite3.Connection.backup; no WAL/SHM copied; source is read-only',
                sqlite_source_sha256=digest(RESULT / 'budget.sqlite'),
                completion=audit['experiment_complete'], files={})
            temporary_zip = staging / 'package.zip'
            with zipfile.ZipFile(temporary_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
                for name in sorted(set(files) | set(memory)):
                    data = memory[name] if name in memory else files[name].read_bytes()
                    scan(data, name)
                    manifest['files'][name] = dict(bytes=len(data), sha256=sha_bytes(data))
                    zipped.writestr(name, data)
                data = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf8')
                scan(data, 'ARCHIVE_MANIFEST.json')
                zipped.writestr('ARCHIVE_MANIFEST.json', data)
            report = verify_archive(temporary_zip)
            # Check once more before publishing; never overwrite a pre-existing ZIP.
            if TARGET.exists():
                raise FileExistsError('Archive appeared while packaging; not replacing it')
            temporary_zip.rename(TARGET)
            report['archive'] = str(TARGET)
            return report
    finally:
        if guard.exists() and guard.read_text(encoding='utf8') == owner:
            guard.unlink()


def self_test():
    for name in ('.cloudru-key.dpapi', '.env', '.git/config', 'x/__pycache__/a.pyc',
                 '../escape.py', 'running.lock', 'x/.pytest-temp/test.py'):
        assert not safe_name(name), name
    for payload in (b'-----BEGIN ' + b'PRIVATE KEY-----', b'Bearer ' + b'a' * 32,
                    b'ghp_' + b'a' * 36, b'api_key = "' + b'x' * 32 + b'"'):
        try:
            scan(payload, 'test.txt')
        except ValueError:
            pass
        else:
            raise AssertionError('Secret-pattern negative control missed')
    scan(b'api_key=os.environ["API_KEY"]', 'source.py')
    source = io.BytesIO()
    with zipfile.ZipFile(source, 'w') as zipped:
        zipped.writestr('example.txt', b'known data')
    source.seek(0)
    with zipfile.ZipFile(source) as zipped:
        assert zipped.testzip() is None and sha_bytes(zipped.read('example.txt')) == sha_bytes(b'known data')
    return dict(self_test='passed', archive_created=False, credentials_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'verify', 'self-test'])
    parser.add_argument('--archive', type=Path, default=TARGET)
    arguments = parser.parse_args()
    if arguments.action == 'create':
        result = create_archive()
    elif arguments.action == 'verify':
        result = verify_archive(arguments.archive)
    else:
        result = self_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))
