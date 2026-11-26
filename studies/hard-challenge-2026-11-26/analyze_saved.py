"""Reproduce the published challenge from saved answers, entirely offline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil


ROOT = Path(__file__).resolve().parent
STUDY = "hard-challenge-2026-11-26"
RESULT_PATH = Path("results") / STUDY
STAGES = ("", "supplement", "robustness")
MODELS = {"sonnet55": "anthropic/claude-sonnet-5.5", "astra": "openai/gpt-6-astra"}
INPUT_NAMES = ("cases.jsonl", "observations.jsonl", "filter_predictions.csv")
SUPPORT_FILES = ("final_financial_summary.json", "robustness/completion.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def input_paths(root=ROOT):
    base = Path(root).resolve() / RESULT_PATH
    return [base / stage / name for stage in STAGES for name in INPUT_NAMES] + [base / name for name in SUPPORT_FILES]


def verify_snapshot(root=ROOT):
    """Check original hashes and fixed observation counts without ML imports."""
    root = Path(root).resolve()
    manifest = json.loads((root / "ARCHIVE_MANIFEST.json").read_text(encoding="utf8"))
    require(manifest.get("study") == STUDY, "Unexpected archive study")
    files = manifest["files"]
    require(bool(files), "Empty archive manifest")
    for name, metadata in files.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name,
                "Unsafe archive manifest path")
        path = root.joinpath(*relative.parts)
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Archive path escapes snapshot")
        require(path.is_file(), "Missing archived file: " + name)
        require(path.stat().st_size == metadata["bytes"] and sha256(path) == metadata["sha256"],
                "Archived file changed: " + name)
    for path in input_paths(root):
        require(path.relative_to(root).as_posix() in files, "Analysis input is not covered by archive manifest")

    base = root / RESULT_PATH
    expected = {"": (128, 64, 16, 768), "supplement": (16, 8, 4, 64), "robustness": (128, 64, 16, 1302)}
    merged = {}
    stages = {}
    main_cases = None
    for stage in STAGES:
        cases = read_rows(base / stage / "cases.jsonl")
        observations = read_rows(base / stage / "observations.jsonl")
        count_cases, count_scenes, count_families, count_observations = expected[stage]
        require(len(cases) == count_cases and len(observations) == count_observations, "Unexpected stage size: " + stage)
        case_map = {row["case"]: row for row in cases}
        require(len(case_map) == len(cases), "Duplicate case identity")
        pairs = defaultdict(list)
        for case in cases:
            pairs[case["scene"]].append(case)
        require(len(pairs) == count_scenes and len({r["family"] for r in cases}) == count_families, "Unexpected situation/family count")
        for pair in pairs.values():
            require(len(pair) == 2 and {r["kind"] for r in pair} == {"attack", "clean"}, "Missing matched control")
            require(all(pair[0][field] == pair[1][field] for field in ("question", "gold", "accepted", "marker", "family", "variant")),
                    "Matched pair changes task or evaluation")
        if stage == "":
            main_cases = case_map
        elif stage == "robustness":
            require(set(case_map) == set(main_cases), "Repeated set changes case identities")
            for key, case in case_map.items():
                require({k: v for k, v in case.items() if k != "repeats"} ==
                        {k: v for k, v in main_cases[key].items() if k != "repeats"}, "Repeated set changes document")

        with (base / stage / "filter_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        filter_keys = [(r["case"], r["defense"]) for r in predictions]
        require(len(filter_keys) == len(set(filter_keys)) == len(cases) * 2 and
                set(filter_keys) == {(key, arm) for key in case_map for arm in ("D", "E")}, "Incomplete filter decisions")
        saved = {}
        paired_models = defaultdict(set)
        for record in observations:
            key = (record["case"], record["model_name"], record["arm"], record["repeat"])
            require(key not in saved, "Duplicate response within a stage")
            require(key[0] in case_map and key[1] in MODELS and key[2] in ("A", "B"), "Unexpected response identity")
            require(type(key[3]) is int and 1 <= key[3] <= case_map[key[0]]["repeats"], "Response outside repetition plan")
            require(record.get("ok") is True and record.get("model") == MODELS[key[1]], "Invalid response/model mapping")
            require(record["request"]["model"] == MODELS[key[1]] and record["request"]["temperature"] == 0 and
                    record["request"]["max_tokens"] == 1024, "Generation parameters changed")
            saved[key] = record
            paired_models[(key[0], key[2], key[3])].add(key[1])
            if key in merged:
                require(merged[key] == record, "Conflicting copies in combined robustness view")
            merged[key] = record
        require(all(names == set(MODELS) for names in paired_models.values()), "Unpaired model responses")
        first = sum(key[3] == 1 for key in saved)
        require(first == len(cases) * 4, "Primary trials are incomplete")
        stages[stage or "main"] = dict(documents=len(cases), situations=len(pairs), families=count_families,
                                       saved_observations=len(saved), first_responses=first)
    per_model = Counter(row["model_name"] for row in merged.values())
    require(len(merged) == 1366 and per_model == {"sonnet55": 683, "astra": 683}, "Unique response total changed")
    return dict(original_files_checked=len(files), distinct_situations=72, unique_observations=len(merged),
                observations_by_model=dict(per_model), stages=stages,
                repeated_trials_complete=False, api_requests=0)


def copy_inputs(root=ROOT):
    """Copy only analysis inputs; never write into the archived results tree."""
    root = Path(root).resolve()
    source = root / RESULT_PATH
    output = root / "output" / "reproduced"
    require(output.resolve().is_relative_to(root), "Output escapes snapshot")
    for path in input_paths(root):
        target = output / path.relative_to(source)
        require(target.resolve().is_relative_to(root / "output"), "Output link escapes output directory")
        require(not target.is_symlink(), "Refusing a symbolic output file")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        require(not temporary.is_symlink(), "Refusing a symbolic temporary file")
        shutil.copyfile(path, temporary)
        temporary.replace(target)
    return output


def load_analytics(root=ROOT):
    """Load only the two frozen offline modules, without adding runtime to sys.path."""
    directory = Path(root).resolve() / "experiments" / STUDY
    modules = []
    for filename in ("analyze.py", "post_analysis.py"):
        spec = importlib.util.spec_from_file_location("published_challenge_" + filename[:-3], directory / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def reproduce(root=ROOT, make_plots=True):
    verification = verify_snapshot(root)
    before = {path: sha256(path) for path in input_paths(root)}
    output = copy_inputs(root)
    analysis, post = load_analytics(root)
    summaries = {}
    for stage in STAGES:
        summaries[stage or "main"] = analysis.run(output / stage, make_plots=make_plots and stage != "robustness", draws=10000)
    synthesis = post.run(output, make_plots=make_plots)
    require(all(sha256(path) == digest for path, digest in before.items()), "Archived analysis input was modified")
    return dict(output=str(output), verification=verification, analyses=summaries, synthesis=synthesis)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check the immutable archive and response pairing without computing intervals")
    parser.add_argument("--no-plots", action="store_true", help="Recalculate tables and reports without plotting")
    args = parser.parse_args(argv)
    result = verify_snapshot() if args.verify_only else reproduce(make_plots=not args.no_plots)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
