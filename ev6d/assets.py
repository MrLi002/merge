"""Official YCB Google 16k downloads, unchanged meshes and asset provenance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

YCB_OBJECTS = ("003_cracker_box", "005_tomato_soup_can", "006_mustard_bottle", "010_potted_meat_can")
YCB_BASE = "https://ycb-benchmarks.s3.amazonaws.com/data/google/"
YCB_INDEX = "https://ycb-benchmarks.s3.amazonaws.com/index.html"

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def fetch_ycb(root, objects=YCB_OBJECTS):
    root, objects = Path(root), tuple(objects)
    if not objects or any(name not in YCB_OBJECTS for name in objects):
        raise ValueError("Select supported YCB object IDs")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "source_index": YCB_INDEX, "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Calli et al., The YCB Object and Model Set, 2015; official Google 16k meshes and textures.", "objects": {}}
    for name in objects:
        filename = f"{name}_google_16k.tgz"
        archive, url = root / filename, YCB_BASE + filename
        previous = manifest["objects"].get(name, {})
        if archive.exists() and previous.get("archive_sha256") and sha256(archive) != previous["archive_sha256"]:
            raise ValueError(f"Cached archive failed integrity check: {archive}")
        if not archive.exists():
            partial = archive.with_suffix(".tgz.part")
            with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024*1024)
            partial.replace(archive)
        extracted = []
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != name:
                    raise ValueError(f"Unsafe or unexpected archive path: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("Only ordinary model files are accepted")
                target = root.joinpath(*path.parts)
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Archive path escapes the model directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                extracted.append({"path": target.relative_to(root).as_posix(), "bytes": target.stat().st_size, "sha256": sha256(target)})
        model = root / name / "google_16k" / "textured.obj"
        if not model.exists():
            raise ValueError(f"Expected textured OBJ missing: {model}")
        manifest["objects"][name] = {"url": url, "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive),
                                      "files": extracted, "model": model.relative_to(root).as_posix()}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"YCB model ready: {name}", flush=True)
    return manifest
