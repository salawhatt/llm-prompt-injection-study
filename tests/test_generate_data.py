"""Проверки восстановления реального тестового набора из исходных таблиц."""
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from generate_data import SOURCE_FILES, build_cases, generate, repeat_selection


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


@pytest.fixture
def isolated_root(tmp_path):
    source = tmp_path / 'data/source'
    source.mkdir(parents=True)
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / 'data/source' / name, source / name)
    return tmp_path


def test_generated_cases_match_published_test(isolated_root):
    # Здесь нет cases.jsonl и ответов: восстановление использует только TSV.
    actual = build_cases(isolated_root / 'data/source')
    expected = read(ROOT / 'data/cases.jsonl')
    assert actual == expected
    assert len(actual) == 588
    assert Counter(row['cohort'] for row in actual) == {
        'main': 360, 'length': 40, 'position_begin': 40,
        'position_middle': 40, 'question': 40, 'challenge': 48, 'stress': 20,
    }


def test_matched_controls_and_input_hashes():
    pairs = defaultdict(list)
    for row in build_cases():
        pairs[row['scene'], row['cohort']].append(row)
        encoded = json.dumps([row['question'], row['document']], ensure_ascii=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == row['input_hash']
    for group in pairs.values():
        assert {row['kind'] for row in group} == {'attack', 'clean'}
        assert len({row['question'] for row in group}) == 1
        assert len({row['gold'] for row in group}) == 1


def test_repeat_selection_matches_saved_responses():
    selection = repeat_selection(build_cases())
    responses = read(ROOT / 'data/responses.jsonl')
    actual = {row['scene'] for row in responses if row['cohort'] == 'main' and row['repeat'] > 1}
    assert len(selection['initial_scenes']) == 45
    assert len(selection['additional_scenes']) == 9
    assert set(selection['initial_scenes']).isdisjoint(selection['additional_scenes'])
    assert set(selection['all_scenes']) == actual
    assert len(actual) == 54


def test_generation_is_reproducible_and_preserves_data(isolated_root):
    before = {p.name: p.read_bytes() for p in (isolated_root / 'data/source').iterdir()}
    first = generate(isolated_root)
    second = generate(isolated_root)
    assert first == second
    dataset = isolated_root / 'output/generated/cases.jsonl'
    manifest = json.loads((dataset.parent / 'manifest.json').read_text(encoding='utf8'))
    assert read(dataset) == build_cases(isolated_root / 'data/source')
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest['dataset_sha256']
    assert before == {p.name: p.read_bytes() for p in (isolated_root / 'data/source').iterdir()}


def test_generation_does_not_overwrite_modified_output(isolated_root):
    generate(isolated_root)
    target = isolated_root / 'output/generated/cases.jsonl'
    target.write_text('modified', encoding='utf8')
    with pytest.raises(FileExistsError):
        generate(isolated_root)
    assert target.read_text(encoding='utf8') == 'modified'


def test_generation_refuses_symlink_output(isolated_root):
    output = isolated_root / 'output/generated'
    output.mkdir(parents=True)
    protected = isolated_root / 'data/protected.jsonl'
    protected.write_text('protected', encoding='utf8')
    try:
        (output / 'cases.jsonl').symlink_to(protected)
    except OSError:
        pytest.skip('Система не разрешает создавать символические ссылки')
    with pytest.raises(ValueError):
        generate(isolated_root)
    assert protected.read_text(encoding='utf8') == 'protected'
