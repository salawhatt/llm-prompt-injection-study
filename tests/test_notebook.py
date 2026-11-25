"""Публикуется только исходный блокнот, без ключей и результатов сеанса."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_valid_source_and_has_no_outputs():
    notebook = json.loads((ROOT / 'notebooks/demo.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert code_cells
    assert len({c['id'] for c in notebook['cells']}) == len(notebook['cells'])
    for cell in code_cells:
        assert cell['outputs'] == [] and cell['execution_count'] is None
        compile(''.join(cell['source']), cell['id'], 'exec')


def test_live_request_only_in_explicit_button_callback():
    notebook = json.loads((ROOT / 'notebooks/demo.ipynb').read_text(encoding='utf-8'))
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(source)
    send = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'send_one')
    requests = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'request_once']
    assert len(requests) == 1 and requests[0] in list(ast.walk(send))
    # No top-level invocation of the callback, including via interactive_output.
    assert 'send_button.on_click(send_one)' in source
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == 'send_one' for n in ast.walk(tree))
