"""Офлайн-воспроизведение исследования из сохранённых ответов."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import study
from research_defense import Detector
from experiment_core import counts

def integrity():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf8"))
    for name, digest in manifest["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Контрольная сумма не совпала: {name}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["report", "verify", "demo", "train", "generate"])
    args = parser.parse_args()
    integrity()
    if args.stage == "generate":
        from generate_data import generate
        print(generate())
        return
    if args.stage == "train":
        train = study.read(ROOT / "data/train.jsonl")
        val = study.read(ROOT / "data/validation.jsonl")
        model = Detector(mode="paragraph", c=1).fit(train)
        threshold = study.load(ROOT)[3]["E"][1]
        pred = model.predict_proba([r["document"] for r in val])[:, 1] >= threshold
        print(counts([r["label"] for r in val], pred))
        print("Повторное обучение проверено на валидации. Исходная модель не заменена.")
        return
    cases, responses, annotations, filters = study.load(ROOT)
    frame, predictions = study.reconstruct(cases, responses, annotations, filters)
    if args.stage == "demo":
        study.show_example(frame, cases, "v4-003", "main")
        return
    main_frame = frame[(frame.cohort == "main") & (frame["repeat"] == 1)]
    comparisons = study.comparisons(main_frame)
    checks = study.verify(ROOT, cases, responses, frame, predictions, comparisons)
    print(f"Проверок пройдено: {len(checks)}")
    if args.stage == "verify":
        return
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    tables = {"outcomes": frame, "main_metrics": study.metrics(main_frame),
              "clean_metrics": study.clean_metrics(main_frame),
              "filter_metrics": study.detector_metrics(predictions),
              "paired_comparisons": comparisons, "factors": study.factor_table(frame),
              "repeats": study.repeats(frame),
              "challenge": study.metrics(frame[(frame.cohort == "challenge") & (frame["repeat"] == 1)])}
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    study.plot_main(main_frame, output)
    study.plot_clean(main_frame, output)
    study.plot_filters(predictions, output)
    study.plot_categories(main_frame, output)
    study.plot_length(predictions, output)
    print(tables["main_metrics"].to_string(index=False))
    print("Таблицы и графики сохранены в output/")

if __name__ == "__main__":
    main()
