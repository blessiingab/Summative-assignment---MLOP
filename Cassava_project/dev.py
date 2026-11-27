from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
RELOAD_EXCLUDES = [
    str(PROJECT_ROOT / ".venv"),
    str(PROJECT_ROOT / "data"),
    str(PROJECT_ROOT / "models"),
]

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT)],
        reload_excludes=RELOAD_EXCLUDES,
        access_log=True,
        app_dir=str(PROJECT_ROOT),
    )
