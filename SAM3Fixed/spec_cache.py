"""
Union-prompt spec cache: multiple cameras can have different prompts/classes,
but they all use ONE SHARED engine baked with the UNION of every class
currently needed -- this fits how nvinfer batches multiple sources through
one model. Post-processing (in ds9_rtsp.py) filters detections back down to
the classes each individual camera actually wants, by cls_id.

engine_dir for a given union gets a stable name (classes sorted + normalized)
so the same set of classes always maps to the same folder, regardless of
which camera declared them first.
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SAM3_FIXED_DIR = _HERE  # export.py lives right here, next to this script
# generated engines go into the shared ../weight/ folder (repo root), not
# next to export.py, so engine_dir_for() below points into WEIGHT_DIR.
WEIGHT_DIR = os.path.join(_ROOT, "weight")


def normalize_class(cls):
    """Same convention as export.py: strip spaces, join with '-' (for the folder name)."""
    return "-".join(cls.strip().split())


def union_classes(camera_prompts):
    """camera_prompts: dict cam_id -> list[str] classes (can be empty -> ["object"]).
    Returns (union_sorted, cam_to_cls_ids) -- the union, stably sorted, plus a
    mapping of each camera to its list of cls_id (index into the union) it
    needs to filter for."""
    seen = set()
    for classes in camera_prompts.values():
        for c in (classes or ["object"]):
            seen.add(c.strip())
    union_sorted = sorted(seen)  # stable order, independent of camera declaration order

    cam_to_cls_ids = {}
    for cam_id, classes in camera_prompts.items():
        classes = classes or ["object"]
        cam_to_cls_ids[cam_id] = [union_sorted.index(c.strip()) for c in classes]

    return union_sorted, cam_to_cls_ids


def engine_dir_for(union_classes_sorted, mode="seg", imgsz=1008):
    suffix = "_".join(normalize_class(c) for c in union_classes_sorted)
    return os.path.join(WEIGHT_DIR, f"sam3_{imgsz}_{mode}_{suffix}")


def is_spec_ready(engine_dir):
    return os.path.exists(os.path.join(engine_dir, "sam3.engine"))


def ensure_spec(union_classes_sorted, mode="seg", imgsz=1008, device="cuda"):
    """Return engine_dir, building it via export.py if not already present.
    Does NOT bake anything export.py doesn't already do -- this is just
    check-or-build logic plus a naming convention keyed on the union."""
    engine_dir = engine_dir_for(union_classes_sorted, mode, imgsz)
    if is_spec_ready(engine_dir):
        print(f"[spec_cache] already available: {engine_dir}")
        return engine_dir

    print(f"[spec_cache] not yet specialized, building: classes={union_classes_sorted} "
          f"mode={mode} imgsz={imgsz} ...")
    cmd = [sys.executable, os.path.join(SAM3_FIXED_DIR, "export.py"),
           "--classes", *union_classes_sorted,
           "--mode", mode, "--imgsz", str(imgsz), "--device", device]
    subprocess.run(cmd, check=True, cwd=SAM3_FIXED_DIR)

    if not is_spec_ready(engine_dir):
        raise RuntimeError(f"export.py finished but no engine found at {engine_dir}")
    print(f"[spec_cache] build done: {engine_dir}")
    return engine_dir


if __name__ == "__main__":
    # example: 4 cams -- {truck}, {person}, {dog,person}, {} (-> object)
    camera_prompts = {
        0: ["truck"],
        1: ["person"],
        2: ["dog", "person"],
        3: [],
    }
    union, cam_map = union_classes(camera_prompts)
    print("union classes:", union)
    print("cam -> cls_ids:", cam_map)
    engine_dir = ensure_spec(union, mode="det", imgsz=504)
    print("engine_dir:", engine_dir)
