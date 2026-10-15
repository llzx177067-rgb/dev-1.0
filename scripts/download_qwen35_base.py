"""Download and verify the exact Qwen 3.5 0.8B Base revision for local training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, __version__, snapshot_download

REPO = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    if algorithm == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_ranges(item: dict, dest: Path) -> None:
    """Fetch verified-size chunks when a proxy stalls an unrestricted large GET."""
    import httpx

    target = dest / item["name"]
    if target.exists() and target.stat().st_size == item["size"] and digest(target, "sha256") == item["lfs_sha256"]:
        return
    chunk_size = 32 * 1024 * 1024
    ranges = [(begin, min(begin + chunk_size, item["size"]) - 1) for begin in range(0, item["size"], chunk_size)]
    parts = dest / "download-parts"
    parts.mkdir(exist_ok=True)

    def fetch(pair: tuple[int, int]) -> Path:
        begin, end = pair
        part = parts / f"{begin:012d}-{end:012d}.part"
        if part.exists() and part.stat().st_size == end - begin + 1:
            return part
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{item['name']}?download=true&jev_part={begin}-{end}"
        for attempt in range(5):
            try:
                response = httpx.get(url, headers={"Range": f"bytes={begin}-{end}"}, follow_redirects=True, timeout=60)
                response.raise_for_status()
                expected_range = f"bytes {begin}-{end}/{item['size']}"
                if response.status_code != 206 or response.headers.get("content-range") != expected_range or len(response.content) != end - begin + 1:
                    raise RuntimeError(f"Invalid range response for {begin}-{end}")
                partial = part.with_suffix(".partial")
                partial.write_bytes(response.content)
                partial.replace(part)
                return part
            except (httpx.HTTPError, RuntimeError) as error:
                if attempt == 4:
                    raise
                print(json.dumps({"event": "range_retry", "range": [begin, end], "attempt": attempt + 1, "error_type": type(error).__name__}), flush=True)
                time.sleep(min(2 ** attempt, 10))
        raise AssertionError("Unreachable")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, pair) for pair in ranges]
        for done_count, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done_count % 8 == 0 or done_count == len(ranges):
                print(json.dumps({"event": "range_progress", "complete_chunks": done_count, "total_chunks": len(ranges)}), flush=True)
    combined = target.with_suffix(".assembled")
    with combined.open("wb") as output:
        for begin, end in ranges:
            with (parts / f"{begin:012d}-{end:012d}.part").open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if combined.stat().st_size != item["size"] or digest(combined, "sha256") != item["lfs_sha256"]:
        raise RuntimeError("Assembled weight LFS SHA256 mismatch")
    combined.replace(target)
    for begin, end in ranges:
        (parts / f"{begin:012d}-{end:012d}.part").unlink()
    parts.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=Path.home() / ".cache/huggingface/jev-qwen35-08b-base-dc7cdfe2")
    parser.add_argument("--range-download-large", action="store_true")
    args = parser.parse_args()
    dest = args.local_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(dest)
    print(json.dumps({"event": "start", "repo": REPO, "revision": REVISION, "path": str(dest), "free_bytes": usage.free}), flush=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise RuntimeError(f"Unexpected revision: {info.sha}")
    files = []
    for sibling in info.siblings:
        lfs = sibling.lfs
        files.append({"name": sibling.rfilename, "size": sibling.size, "blob_id": sibling.blob_id,
                      "lfs_sha256": lfs.sha256 if lfs is not None else None})
    total = sum(x["size"] or 0 for x in files)
    if usage.free < total + 512 * 1024 * 1024:
        raise RuntimeError("Not enough disk space for pinned model snapshot")
    metadata = {"repo": REPO, "revision": info.sha, "files": files, "total_bytes": total}
    (dest / "remote-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "remote_metadata", "file_count": len(files), "total_bytes": total}), flush=True)
    started = time.time()
    large = [x for x in files if (x["size"] or 0) > 256 * 1024 * 1024]
    small_patterns = [x["name"] for x in files if x not in large]
    snapshot_download(REPO, revision=REVISION, local_dir=dest, max_workers=4, token=False,
                      allow_patterns=small_patterns if args.range_download_large else None)
    if args.range_download_large:
        for item in large:
            download_ranges(item, dest)
    verified = []
    for item in files:
        path = dest / item["name"]
        size = path.stat().st_size
        sha = digest(path, "sha256")
        git_blob = digest(path, "sha1") if item["lfs_sha256"] is None else None
        expected_hash = item["lfs_sha256"] or item["blob_id"]
        actual_hash = sha if item["lfs_sha256"] else git_blob
        if size != item["size"] or actual_hash != expected_hash:
            raise RuntimeError(f"Integrity mismatch for {item['name']}: size/hash")
        verified.append({**item, "actual_size": size, "actual_sha256": sha, "actual_git_blob_sha1": git_blob, "verified": True})
        print(json.dumps({"event": "verified", "file": item["name"], "size": size, "sha256": sha}), flush=True)
    config = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"] or config.get("model_type") != "qwen3_5":
        raise RuntimeError("Unexpected model architecture")
    if config["text_config"]["hidden_size"] != 1024:
        raise RuntimeError("Unexpected text hidden size")
    index = json.loads((dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
    tensor_total = 0
    tensor_counts = {}
    tensor_names = set()
    for filename in sorted(set(index["weight_map"].values())):
        file = dest / filename
        with file.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        ranges = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            shape = tensor["shape"]
            count = 1
            for dimension in shape:
                count *= dimension
            tensor_total += count
            tensor_counts[tensor["dtype"]] = tensor_counts.get(tensor["dtype"], 0) + count
            begin, end = tensor["data_offsets"]
            if not (0 <= begin <= end <= file.stat().st_size - data_start):
                raise RuntimeError(f"Tensor offset outside file: {name}")
            if index["weight_map"].get(name) != filename:
                raise RuntimeError(f"Tensor not in index or points to another shard: {name}")
            ranges.append((begin, end))
            tensor_names.add(name)
        ranges.sort()
        if any(ranges[i][1] != ranges[i + 1][0] for i in range(len(ranges) - 1)):
            raise RuntimeError("Tensor data ranges have a gap or overlap")
        if ranges[0][0] != 0 or ranges[-1][1] != file.stat().st_size - data_start:
            raise RuntimeError("Tensor byte ranges do not cover the full shard")
    if tensor_names != set(index["weight_map"]):
        raise RuntimeError("Safetensors header and weight index tensor names differ")
    if tensor_total != 873438784:
        raise RuntimeError(f"Unexpected number of parameters: {tensor_total}")
    manifest = {"status": "verified", "repo": REPO, "revision": REVISION, "path": str(dest),
                "docker_read_only_mount": "/models/qwen35-0.8b-base", "local_snapshot_directory": str(dest),
                "huggingface_hub_version": __version__, "model_type": config["model_type"],
                "architectures": config["architectures"], "parameter_count": tensor_total, "parameters_by_dtype": tensor_counts,
                "tensor_count": len(tensor_names), "total_repository_bytes": total, "files": verified,
                "elapsed_download_and_verification_seconds": round(time.time() - started, 3)}
    (dest / "download-verification-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "manifest": str(dest / "download-verification-manifest.json"), "total_bytes": total,
                      "parameter_count": tensor_total, "elapsed_seconds": manifest["elapsed_download_and_verification_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
