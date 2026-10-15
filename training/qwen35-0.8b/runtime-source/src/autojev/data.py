import argparse
import hashlib
import importlib
import json
import math
import random
import re
import string
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import httpx
from PIL import Image, ImageDraw

from autojev.events import record
from autojev.types import Content, Example, JSONValue, Label, Question, Target

NIMBLE_REVISION = "d2387fc0b32d1173bfc995395c076a25a2a107c9"
SEED = 20260920


@dataclass(frozen=True)
class Source:
    repository: str
    revision: str
    config: str | None = None


SOURCES = {
    "boolq": Source("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5"),
    "squad2": Source("rajpurkar/squad_v2", "3ffb306f725f7d2ce8394bc1873b24868140c412"),
    "paws": Source("google-research-datasets/paws", "161ece9501cf0a11f3e48bd356eaa82de46d6a09", "labeled_final"),
    "civil_comments": Source("google/civil_comments", "f2970eb3a55777454c94069077cc8d9b5866312d"),
    "aegis2": Source("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"),
    "pubmedqa": Source("qiaojin/PubMedQA", "9001f2853fb87cab8d220904e0de81ac6973b318", "pqa_labeled"),
}
ARCHIVES = {
    "vitaminc": ("https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip", "vitaminc.zip"),
    "multinli": ("https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip", "multinli_1.0.zip"),
    "massive": (
        "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
        "massive-1.1.tar.gz",
    ),
}
ARCHIVE_SHA256 = {
    "vitaminc": "49d82dc1690cbee420d18e2c26f687a7937710bb211845d2571430dfd4dc0337",
    "multinli": "049f507b9e36b1fcb756cfd5aeb3b7a0cfcb84bf023793652987f7e7e0957822",
    "massive": "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
}
TASKS = (
    "vitaminc-dev",
    "massive-en-US",
    "massive-de-DE",
    "boolq",
    "squad2",
    "paws",
    "multinli",
    "civil_comments",
    "aegis2",
    "pubmedqa",
)
TRAIN_QUOTAS = dict(zip(TASKS, (16000, 10000, 10000, 7000, 14000, 12000, 12000, 8000, 6000, 0), strict=False))
TEST_QUOTAS = dict(zip(TASKS, (920, 540, 540, 460, 460, 385, 460, 460, 385, 385), strict=False))
DEV_QUOTAS = dict(zip(TASKS, (275, 160, 160, 140, 140, 115, 140, 140, 115, 115), strict=False))
TEMPERATURE_QUOTAS = dict(zip(TASKS, (185, 110, 110, 90, 90, 75, 90, 90, 75, 75), strict=False))
KNOWN_SPLITS = {
    "vitaminc": "dev",
    "massive": "test",
    "boolq": "validation",
    "squad2": "validation",
    "paws": "test",
    "multinli": "dev_matched",
    "civil_comments": "test",
    "aegis2": "test",
    "pubmedqa": "train",
}
DEVELOPMENT_SPLITS = {
    **KNOWN_SPLITS,
    "massive": "dev",
    "paws": "validation",
    "civil_comments": "validation",
    "aegis2": "validation",
}
FINAL_SPLITS = {**KNOWN_SPLITS, "vitaminc": "test"}


class Converter(Protocol):
    SOURCE_URL: str
    LICENSE: str

    def record(self, raw: dict[str, JSONValue], subset: str = "") -> dict[str, JSONValue] | None: ...


class Serializer(Protocol):
    def jsonl_line(self, value: dict[str, JSONValue]) -> str: ...


class Arguments(argparse.Namespace):
    cache_root: Path
    source_cache: Path | None
    nimble_source: Path | None
    output: Path
    manifest: Path
    seed: int
    max_length: int
    train_scale: float
    image_train: int
    image_eval: int


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def ranking(value: str, seed: int) -> str:
    return digest(f"{seed}:{value}")


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=180) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_bytes(1024 * 1024):
                stream.write(chunk)
    temporary.replace(path)


def checkout(path: Path | None, raw: Path) -> Path:
    if path is not None:
        if (path / ".revision").read_text().strip() != NIMBLE_REVISION:
            raise ValueError("The supplied Nimble checkout is not the pinned revision")
        return path.resolve()
    path = raw / "nimble-source"
    if not path.exists():
        archive = raw / f"nimble-{NIMBLE_REVISION}.tar.gz"
        download(f"https://codeload.github.com/bespokelabsai/nimble/tar.gz/{NIMBLE_REVISION}", archive)
        with tarfile.open(archive) as bundle:
            bundle.extractall(raw, filter="data")
        (raw / f"nimble-{NIMBLE_REVISION}").rename(path)
        (path / ".revision").write_text(NIMBLE_REVISION + "\n")
    return path.resolve()


class RawData:
    def __init__(self, args: Arguments) -> None:
        self.cache = args.cache_root
        self.supplied = args.source_cache
        self.folder = args.output / "raw"
        self.files: dict[str, dict[str, str]] = {}

    def remember(self, path: Path, source: str) -> None:
        key = f"{source}/{path.name}"
        if key not in self.files:
            self.files[key] = {"name": path.name, "sha256": checksum(path)}

    def archive(self, source: str) -> Path:
        url, name = ARCHIVES[source]
        cached = self.supplied / name if self.supplied else self.folder / name
        if not cached.exists():
            cached = self.folder / name
            download(url, cached)
        self.remember(cached, source)
        if self.files[f"{source}/{cached.name}"]["sha256"] != ARCHIVE_SHA256[source]:
            raise ValueError(f"The {source} archive differs from the pinned checksum")
        return cached

    def rows(self, source: str, split: str, locale: str = "") -> Iterator[dict[str, JSONValue]]:
        if source == "massive":
            with tarfile.open(self.archive(source)) as archive:
                member = next(m for m in archive.getmembers() if m.name.endswith(f"/data/{locale}.jsonl"))
                stream = archive.extractfile(member)
                assert stream is not None
                for line in stream:
                    row = cast(dict[str, JSONValue], json.loads(line))
                    if row["partition"] == split:
                        yield row
            return
        if source in ("vitaminc", "multinli"):
            member_name = (
                f"vitaminc/{split}.jsonl" if source == "vitaminc" else f"multinli_1.0/multinli_1.0_{split}.jsonl"
            )
            with zipfile.ZipFile(self.archive(source)) as zip_archive, zip_archive.open(member_name) as zip_stream:
                for line in zip_stream:
                    yield cast(dict[str, JSONValue], json.loads(line))
            return
        cached = self.supplied / source / f"{split}.jsonl" if self.supplied else self.folder / source / f"{split}.jsonl"
        if cached.exists():
            self.remember(cached, source)
            with cached.open() as text_stream:
                for index, text_line in enumerate(text_stream):
                    row = cast(dict[str, JSONValue], json.loads(text_line))
                    if source == "civil_comments":
                        row["row_index"] = index
                    yield row
            return
        from datasets import Dataset, load_dataset

        spec = SOURCES[source]
        paths = sorted((self.cache / "datasets").glob(f"**/{spec.revision}/*-{split}*.arrow"))
        datasets: list[Dataset]
        if paths:
            datasets = [Dataset.from_file(str(path)) for path in paths]
            for path in paths:
                self.remember(path, source)
        else:
            datasets = [
                cast(
                    Dataset,
                    load_dataset(
                        spec.repository,
                        spec.config,
                        split=split,
                        revision=spec.revision,
                        cache_dir=str(self.folder / "hf"),
                    ),
                )
            ]
        index = 0
        for dataset in datasets:
            for value in dataset:
                row = cast(dict[str, JSONValue], value)
                if source == "civil_comments":
                    row["row_index"] = index
                index += 1
                yield row


class Families:
    def __init__(self, raw: RawData) -> None:
        self.parents: dict[str, str] = {}
        self.paws: dict[tuple[str, str], str] = {}
        for split in ("train", "validation", "test"):
            for row in raw.rows("paws", split):
                values: list[str] = []
                for key in ("sentence1", "sentence2"):
                    text = normalized(str(row[key]))
                    values.extend((digest(text), digest(" ".join(sorted(re.findall(r"\w+", text))))))
                roots = [self.find(value) for value in values]
                root = min(roots)
                for other in roots:
                    self.parents[other] = root
                self.paws[split, str(row["id"])] = root

    def find(self, value: str) -> str:
        self.parents.setdefault(value, value)
        while value != self.parents[value]:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def key(self, source: str, split: str, raw: dict[str, JSONValue]) -> str:
        if source == "paws":
            return self.find(self.paws[split, str(raw["id"])])
        field = {"massive": "id", "vitaminc": "case_id", "multinli": "promptID", "pubmedqa": "pubid"}.get(source)
        if field:
            return str(raw[field])
        field = {"boolq": "passage", "squad2": "context", "aegis2": "prompt", "civil_comments": "text"}[source]
        return digest(normalized(str(raw[field])))


def task_source(task: str) -> tuple[str, str]:
    if task.startswith("massive-"):
        return "massive", task.removeprefix("massive-")
    return ("vitaminc", "") if task == "vitaminc-dev" else (task, "")


def convert(
    raw: dict[str, JSONValue], task: str, split: str, converter: Converter, families: Families
) -> tuple[Example, dict[str, JSONValue]] | None:
    source, locale = task_source(task)
    value = converter.record(raw | {"partition": "test"} if source == "massive" else raw, locale)
    if value is None:
        return None
    model_input = cast(dict[str, JSONValue], value["input"])
    questions = cast(dict[str, Question], model_input["questions"])
    question = questions["decision"]
    reference = cast(dict[str, JSONValue], value["reference"])
    label = cast(Label, reference["target"])
    target: Target = label
    distribution = reference.get("distribution")
    if split == "train" and source != "pubmedqa" and isinstance(distribution, dict):
        if question["type"] == "noul":
            target = float(cast(float, distribution["true"]))
        elif question["type"] == "choice":
            target = [float(cast(float, distribution[key])) for key in question["criteria"]]
    provenance: dict[str, JSONValue] = {
        "dataset": source,
        "split": split,
        "upstream_id": value["id"],
        "url": converter.SOURCE_URL,
        "license": converter.LICENSE,
        "nimble_revision": NIMBLE_REVISION,
    }
    if source in SOURCES:
        provenance["revision"] = SOURCES[source].revision
    if distribution is not None:
        provenance["human_distribution"] = distribution
    identifier = f"{task}:{split}:{value['id']}"
    if split == "train" and source != "pubmedqa":
        identifier += ":" + digest(json.dumps(model_input["state"], sort_keys=True, ensure_ascii=False))[:16]
    result: Example = {
        "id": identifier,
        "suite": task,
        "family": families.key(source, split, raw),
        "state": cast(Content, model_input["state"]),
        "question": question,
        "target": target,
        "label": label,
        "source": provenance,
    }
    return result, value


def choose(rows: list[Example], quota: int, seed: int, excluded: set[str] | None = None) -> list[Example]:
    groups: dict[str, list[Example]] = defaultdict(list)
    for row in rows:
        if excluded is None or row["family"] not in excluded:
            groups[row["family"]].append(row)
    selected: list[Example] = []
    for family in sorted(groups, key=lambda key: ranking(key, seed)):
        group = groups[family]
        if len(selected) + len(group) <= quota:
            selected.extend(sorted(group, key=lambda row: row["id"]))
    return selected


def text_parts(row: Example) -> list[str]:
    state = row["state"]
    if isinstance(state, str):
        return [state]
    if isinstance(state, dict):
        return [value for value in state.values() if isinstance(value, str)]
    return [json.dumps(state, sort_keys=True)]


def state_key(row: Example) -> str:
    return digest(normalized(json.dumps(row["state"], sort_keys=True, ensure_ascii=False)))


class NearDuplicates:
    def __init__(self) -> None:
        self.parts: list[set[str]] = []
        self.index: dict[str, list[int]] = defaultdict(list)

    @staticmethod
    def shingles(text: str) -> set[str]:
        words = re.findall(r"\w+", normalized(text))
        return {" ".join(words[index : index + 5]) for index in range(len(words) - 4)} if len(words) >= 20 else set()

    def add(self, row: Example) -> None:
        for text in text_parts(row):
            values = self.shingles(text)
            if values:
                index = len(self.parts)
                self.parts.append(values)
                for anchor in sorted(values, key=digest)[:8]:
                    self.index[anchor].append(index)

    def matches(self, row: Example) -> bool:
        for text in text_parts(row):
            values = self.shingles(text)
            if not values:
                continue
            candidates = {index for anchor in sorted(values, key=digest)[:8] for index in self.index.get(anchor, [])}
            for index in candidates:
                other = self.parts[index]
                if len(values & other) / len(values | other) >= 0.85:
                    return True
        return False


def write_rows(path: Path, rows: list[Example]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows)
    if path.exists() and path.read_text() != payload:
        raise FileExistsError(f"Refusing to replace a different frozen dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return checksum(path)


def images(count: int, split: str, seed: int, output: Path) -> list[Example]:
    rng = random.Random(seed)
    colors = {"red": "#df3030", "blue": "#3068da", "green": "#259951", "orange": "#ed911d"}
    shapes = ("circle", "square", "triangle")
    rows: list[Example] = []
    directory = output / "images" / split
    directory.mkdir(parents=True, exist_ok=True)
    for scene in range(math.ceil(count / 4)):
        picture = Image.new("RGB", (224, 224), "#f9f6ee")
        drawing = ImageDraw.Draw(picture)
        objects: list[tuple[str, str]] = []
        for x in (55, 169):
            color, shape = rng.choice(list(colors)), rng.choice(shapes)
            y, radius = rng.randint(85, 135), rng.randint(25, 40)
            box = (x - radius, y - radius, x + radius, y + radius)
            if shape == "circle":
                drawing.ellipse(box, fill=colors[color])
            elif shape == "square":
                drawing.rectangle(box, fill=colors[color])
            else:
                drawing.polygon(
                    ((x, y - radius), (x + radius, y + radius), (x - radius, y + radius)), fill=colors[color]
                )
            objects.append((color, shape))
        image_path = directory / f"{scene:04}.png"
        picture.save(image_path)
        for side, (color, shape) in zip(("left", "right"), objects, strict=False):
            for attribute, label, options in (("color", color, list(colors)), ("shape", shape, list(shapes))):
                rows.append(
                    {
                        "id": f"images:{split}:{scene}:{side}:{attribute}",
                        "suite": "image_retention",
                        "family": f"{split}:{scene}",
                        "state": "Use the picture to answer the question.",
                        "images": [str(image_path)],
                        "label": label,
                        "target": label,
                        "question": {
                            "type": "choice",
                            "instructions": f"What is the {attribute} of the {side} object?",
                            "criteria": dict.fromkeys(options),
                        },
                        "source": {
                            "dataset": "generated_geometry",
                            "split": split,
                            "seed": seed,
                            "image_sha256": checksum(image_path),
                            "license": "Apache-2.0",
                        },
                    }
                )
    return rows[:count]


def validate(rows: list[Example]) -> None:
    identifiers: set[str] = set()
    for row in rows:
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate example ID: {row['id']}")
        identifiers.add(row["id"])
        question, target, label = row["question"], row["target"], row["label"]
        if not row["family"]:
            raise ValueError("Missing family")
        if question["type"] == "choice":
            options = list(question["criteria"])
            if not isinstance(label, str) or label not in options:
                raise ValueError("Invalid Choice hard label")
            if isinstance(target, list):
                if (
                    len(target) != len(options)
                    or abs(sum(target) - 1) > 1e-6
                    or any(not math.isfinite(p) or p < 0 for p in target)
                ):
                    raise ValueError("Invalid soft Choice target")
            elif target not in options:
                raise ValueError("Invalid Choice target")
        elif question["type"] == "noul":
            if type(label) is not bool or not isinstance(target, bool | float) or not 0 <= target <= 1:
                raise ValueError("Invalid Noul label/target")
        else:
            raise ValueError("The accuracy experiment admits only Choice and Noul examples")


def token_lengths(rows: list[Example], cache: Path) -> None:
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    from autojev.model import BASE_MODEL, BASE_REVISION, decision_messages, open_image

    processor = cast(
        Qwen3VLProcessor,
        AutoProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION, cache_dir=str(cache / "hub")),
    )
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}
    codes = [
        code for code in string.ascii_uppercase if len(processor.tokenizer.encode(code, add_special_tokens=False)) == 1
    ]
    if len(codes) != 26:
        raise ValueError("The benchmark requires the model's first 26 answer codes")
    text_rows = [row for row in rows if not row.get("images")]
    for start in range(0, len(text_rows), 512):
        batch = text_rows[start : start + 512]
        texts = [
            processor.apply_chat_template(
                decision_messages(row, codes), tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for row in batch
        ]
        encoded = cast(list[list[int]], processor.tokenizer(texts, padding=False, truncation=False)["input_ids"])
        for row, tokens in zip(batch, encoded, strict=False):
            row["source"]["input_tokens"] = len(tokens)
    for row in rows:
        if not row.get("images"):
            continue
        text = processor.apply_chat_template(
            decision_messages(row, codes), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded_images = processor(text=[text], images=[open_image(image) for image in row.get("images", [])])
        row["source"]["input_tokens"] = len(cast(list[list[int]], encoded_images["input_ids"])[0])


def build(args: Arguments) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    raw = RawData(args)
    dependency = checkout(args.nimble_source, raw.folder)
    sys.path.insert(0, str(dependency))
    serializer = cast(Serializer, importlib.import_module("nimble.datasets.public_benchmarks"))
    families = Families(raw)
    folds: dict[str, list[Example]] = {name: [] for name in ("known", "test", "dev", "temperature", "train")}
    report: dict[str, object] = {"seed": args.seed, "nimble_revision": NIMBLE_REVISION, "tasks": {}, "sources": {}}
    task_reports = cast(dict[str, object], report["tasks"])
    source_reports = cast(dict[str, object], report["sources"])
    protected: dict[str, set[str]] = defaultdict(set)
    record("data_started", seed=args.seed, tasks=list(TASKS))
    for task in TASKS:
        source, locale = task_source(task)
        converter = cast(Converter, importlib.import_module(f"nimble.datasets.public_sources.{source}"))
        published = json.loads((dependency / f"docs/assets/public-benchmarks/subsets/{task}-manifest.json").read_text())
        known_ids = set(published["ids"])
        pools: dict[str, list[Example]] = defaultdict(list)
        exact_known: list[dict[str, JSONValue]] = []
        stats: Counter[str] = Counter()
        for split in sorted({KNOWN_SPLITS[source], FINAL_SPLITS[source], DEVELOPMENT_SPLITS[source]}):
            for value in raw.rows(source, split, locale):
                result = convert(value, task, split, converter, families)
                if result is None:
                    stats["invalid_heldout_rows"] += 1
                    continue
                row, upstream = result
                protected[source].add(row["family"])
                if split == KNOWN_SPLITS[source] and upstream["id"] in known_ids:
                    folds["known"].append(row)
                    exact_known.append(upstream)
                pools[split].append(row)
            unique_ids: dict[str, Example] = {}
            for candidate in pools[split]:
                previous_row = unique_ids.get(candidate["id"])
                if previous_row is not None:
                    if previous_row["state"] != candidate["state"] or previous_row["label"] != candidate["label"]:
                        raise ValueError("Conflicting rows share an upstream example ID")
                    stats["duplicate_heldout_ids_removed"] += 1
                unique_ids.setdefault(candidate["id"], candidate)
            pools[split] = list(unique_ids.values())
        known_payload = "".join(
            serializer.jsonl_line(row) for row in sorted(exact_known, key=lambda row: str(row["id"]))
        )
        if digest(known_payload) != published["dataset_sha256"]:
            raise ValueError(f"Known benchmark content changed for {task}")

        excluded = {row["family"] for row in folds["known"] if row["source"]["dataset"] == source}
        for fold, quota, split in (
            ("test", TEST_QUOTAS[task], FINAL_SPLITS[source]),
            ("dev", DEV_QUOTAS[task], DEVELOPMENT_SPLITS[source]),
            ("temperature", TEMPERATURE_QUOTAS[task], DEVELOPMENT_SPLITS[source]),
        ):
            if locale == "de-DE":
                wanted = {row["family"] for row in folds[fold] if row["suite"] == "massive-en-US"}
                selected = [row for row in pools[split] if row["family"] in wanted]
                if len(selected) != len(wanted):
                    raise ValueError("A MASSIVE translation is missing")
            else:
                selected = choose(
                    pools[split], quota, args.seed + {"test": 1, "dev": 2, "temperature": 3}[fold], excluded
                )
            folds[fold].extend(selected)
            excluded.update(row["family"] for row in selected)
            stats[fold] = len(selected)
        stats["known"] = len(exact_known)
        source_reports[source] = {
            "url": converter.SOURCE_URL,
            "license": converter.LICENSE,
            "revision": SOURCES[source].revision if source in SOURCES else None,
            "converter_sha256": checksum(dependency / f"nimble/datasets/public_sources/{source}.py"),
        }
        task_reports[task] = dict(stats)
        record("data_task_partitioned", task=task, counts=dict(stats))

    protected_states: set[str] = set()
    protected_near = NearDuplicates()
    for fold in ("known", "test", "dev", "temperature"):
        if fold != "known":
            bad_families = {
                (str(row["source"]["dataset"]), row["family"])
                for row in folds[fold]
                if state_key(row) in protected_states or protected_near.matches(row)
            }
            folds[fold] = [
                row for row in folds[fold] if (str(row["source"]["dataset"]), row["family"]) not in bad_families
            ]
            report[f"{fold}_duplicate_families_removed"] = len(bad_families)
        for row in folds[fold]:
            protected_states.add(state_key(row))
            protected_near.add(row)

    token_lengths([row for name, rows in folds.items() if name != "train" for row in rows], args.cache_root)
    early_partitions: dict[str, object] = {}
    for name in ("known", "test", "dev", "temperature"):
        rows = folds[name]
        overlong = {
            (str(row["source"]["dataset"]), row["family"])
            for row in rows
            if cast(int, row["source"]["input_tokens"]) > args.max_length
        }
        if name == "known" and overlong:
            raise ValueError("The known benchmark exceeds the token limit; never silently change its records")
        folds[name] = [row for row in rows if (str(row["source"]["dataset"]), row["family"]) not in overlong]
        report[f"{name}_overlong_families_excluded"] = len(overlong)
        validate(folds[name])
        early_partitions[name] = {
            "rows": len(folds[name]),
            "sha256": write_rows(args.output / f"{name}.jsonl", folds[name]),
            "tasks": dict(Counter(row["suite"] for row in folds[name])),
        }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    (args.manifest.parent / "evaluation-partitions.json").write_text(json.dumps(early_partitions, indent=2) + "\n")
    record("evaluation_data_frozen", partitions=early_partitions)

    for task in TASKS:
        quota = int(TRAIN_QUOTAS[task] * args.train_scale)
        if not quota:
            continue
        source, locale = task_source(task)
        converter = cast(Converter, importlib.import_module(f"nimble.datasets.public_sources.{source}"))
        candidates: list[Example] = []
        stats = Counter[str]()
        pool_fraction = min(
            1.0, quota * 5 / {"vitaminc": 370653, "multinli": 392702, "civil_comments": 1804874}.get(source, 1)
        )
        for value in raw.rows(source, "train", locale):
            family = families.key(source, "train", value)
            if family in protected[source]:
                stats["heldout_family_rows_excluded"] += 1
                continue
            if int(ranking(family, args.seed), 16) / 2**256 >= pool_fraction:
                continue
            result = convert(value, task, "train", converter, families)
            if result is None:
                stats["invalid_training_rows"] += 1
                continue
            row, _ = result
            candidates.append(row)
        bad = {row["family"] for row in candidates if state_key(row) in protected_states or protected_near.matches(row)}
        candidates = [row for row in candidates if row["family"] not in bad]
        unique: dict[str, Example] = {}
        conflicts: set[str] = set()
        for row in candidates:
            key = state_key(row)
            if key in unique and unique[key]["label"] != row["label"]:
                conflicts.add(key)
            unique.setdefault(key, row)
        candidates = [row for key, row in unique.items() if key not in conflicts]
        selected = choose(candidates, quota, args.seed + 4)
        if len(selected) < 0.8 * quota:
            raise ValueError(f"Too few training rows for {task}: {len(selected)} / {quota}")
        folds["train"].extend(selected)
        stats.update(
            {
                "train": len(selected),
                "near_duplicate_families_excluded": len(bad),
                "conflicting_state_labels_excluded": len(conflicts),
            }
        )
        cast(dict[str, object], task_reports[task]).update(dict(stats))
        record("data_training_task", task=task, counts=dict(stats))

    folds["train"].extend(images(args.image_train, "train", args.seed + 100, args.output))
    folds["image_test"] = images(args.image_eval, "test", args.seed + 101, args.output)
    token_lengths(folds["train"] + folds["image_test"], args.cache_root)
    for name in ("train", "image_test"):
        rows = folds[name]
        overlong = {
            (str(row["source"]["dataset"]), row["family"])
            for row in rows
            if cast(int, row["source"]["input_tokens"]) > args.max_length
        }
        if name == "known" and overlong:
            raise ValueError("The known benchmark exceeds the model token limit; never silently change its records")
        folds[name] = [row for row in rows if (str(row["source"]["dataset"]), row["family"]) not in overlong]
        report[f"{name}_overlong_families_excluded"] = len(overlong)
    random.Random(args.seed).shuffle(folds["train"])
    partitions: dict[str, object] = {}
    previous: dict[tuple[str, str], str] = {}
    for name, rows in folds.items():
        validate(rows)

        keys = {(str(row["source"]["dataset"]), row["family"]) for row in rows}
        for family_key in keys:
            if family_key in previous:
                raise ValueError(f"Family crossed {previous[family_key]} and {name}")
            previous[family_key] = name
        path = args.output / f"{name}.jsonl"
        partitions[name] = {
            "rows": len(rows),
            "families": len(keys),
            "sha256": write_rows(path, rows),
            "tasks": dict(Counter(row["suite"] for row in rows)),
            "tokens": sum(cast(int, row["source"]["input_tokens"]) for row in rows),
            "maximum_input_tokens": max(cast(int, row["source"]["input_tokens"]) for row in rows),
            "question_types": dict(Counter(row["question"]["type"] for row in rows)),
        }
        identifiers = [{"id": row["id"], "dataset": row["source"]["dataset"], "family": row["family"]} for row in rows]
        identity_path = args.manifest.parent / "splits" / f"{name}.json"
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(json.dumps(identifiers, separators=(",", ":")) + "\n")
    report.update(
        {
            "partitions": partitions,
            "raw_files": raw.files,
            "generator_sha256": checksum(Path(__file__)),
            "quotas": {
                "train": TRAIN_QUOTAS,
                "test": TEST_QUOTAS,
                "dev": DEV_QUOTAS,
                "temperature": TEMPERATURE_QUOTAS,
            },
            "max_input_tokens": args.max_length,
            "train_scale": args.train_scale,
            "family_policy": "Whole source families; shared MASSIVE IDs; normalized passages/prompts; PAWS connected sentence/token-bag components.",
            "near_duplicate_policy": "Five-word shingle Jaccard >= 0.85, candidates retrieved by eight minimum SHA256 anchors; text parts below 20 words use exact checks only.",
            "known_suite": "The 3247 original Choice/Noul records are verified against pinned Nimble checksums before adding provenance.",
            "limitations": [
                "PAWS original generation mapping is unavailable (official URL returned 403); sentence/token-bag components and shingle checks are heuristic.",
                "Near-duplicate retrieval is approximate and does not prove absence of every paraphrase or pretraining overlap.",
                "PubMedQA labeled data are reserved for evaluation; no in-domain PubMedQA training in the first corpus.",
                "Image retention is controlled geometry, not a general vision benchmark.",
                "Fresh means unused in this project's tuning, not necessarily unseen by the base model or Jev.",
            ],
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    record("data_frozen", manifest_sha256=checksum(args.manifest), partitions=partitions)
    print(json.dumps(partitions, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed, family-separated decisions from pinned public sources.\n\nThe benchmark's converters are an external, revision-pinned dependency. This\nmodule selects records, checks separation, and records their provenance."
    )
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/huggingface")
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--nimble-source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("progress/data-manifest.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--train-scale", type=float, default=1.0)
    parser.add_argument("--image-train", type=int, default=1000)
    parser.add_argument("--image-eval", type=int, default=200)
    build(parser.parse_args(namespace=Arguments()))


if __name__ == "__main__":
    main()
