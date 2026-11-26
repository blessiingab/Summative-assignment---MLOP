from pathlib import Path
from PIL import Image, UnidentifiedImageError
from .config import CLASS_NAMES

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def validate_image_folder(folder, require_class_folders=True):
    """Check class-folder uploads and return a report; does not delete files."""
    folder = Path(folder)
    report = {"valid": [], "invalid": [], "unknown_folders": []}
    if not folder.exists():
        raise FileNotFoundError(folder)

    if require_class_folders:
        for class_dir in folder.iterdir():
            if not class_dir.is_dir():
                continue
            if class_dir.name not in CLASS_NAMES.values():
                report["unknown_folders"].append(class_dir.name)
                continue
            for path in class_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in VALID_EXTENSIONS:
                    report["invalid"].append((str(path), "unsupported extension"))
                    continue
                try:
                    with Image.open(path) as im:
                        im.verify()
                    report["valid"].append(str(path))
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    report["invalid"].append((str(path), str(exc)))
    else:
        for path in folder.rglob("*"):
            if path.is_file():
                try:
                    with Image.open(path) as im:
                        im.verify()
                    report["valid"].append(str(path))
                except Exception as exc:
                    report["invalid"].append((str(path), str(exc)))
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate uploaded cassava images")
    parser.add_argument("folder")
    args = parser.parse_args()
    result = validate_image_folder(args.folder)
    print("Valid:", len(result["valid"]))
    print("Invalid:", len(result["invalid"]))
    print("Unknown class folders:", result["unknown_folders"])
    for item in result["invalid"]:
        print("INVALID:", item)
