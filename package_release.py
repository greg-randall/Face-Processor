"""Prepare a distributable Windows release archive."""

from __future__ import annotations

import shutil
import stat
import time
from pathlib import Path


def _on_rm_error(func, path, exc_info):
    """Force removal of read-only files when clearing staging directories."""
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    dist_dir = project_root / "dist" / "gui"
    exe_path = dist_dir / "gui.exe"
    internal_dir = dist_dir / "_internal"

    if not exe_path.exists() or not internal_dir.exists():
        raise SystemExit(
            "PyInstaller output not found. Run the build step before packaging."
        )

    build_dir = project_root / "build"
    staging_root = build_dir / "release_tmp"
    package_root = staging_root / "FaceProcessor_Windows"

    if staging_root.exists():
        shutil.rmtree(staging_root, onerror=_on_rm_error)

    package_root.mkdir(parents=True)

    shutil.copy2(exe_path, package_root / "FaceProcessor.exe")
    shutil.copytree(internal_dir, package_root / "_internal", dirs_exist_ok=True)
    shutil.copy2(project_root / "median_landmarks.json", package_root / "median_landmarks.json")

    build_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    archive_base = project_root / f"release_windows_{timestamp}"
    while archive_base.with_suffix(".zip").exists():
        timestamp += 1
        archive_base = project_root / f"release_windows_{timestamp}"
    archive_path = archive_base.with_suffix(".zip")
    if archive_path.exists():
        try:
            archive_path.unlink()
        except PermissionError:
            raise SystemExit(
                f"Unable to remove existing archive at {archive_path}. Please close any programs using it."
            )

    shutil.make_archive(str(archive_base), "zip", root_dir=staging_root, base_dir="FaceProcessor_Windows")

    shutil.rmtree(staging_root, onerror=_on_rm_error)

    for folder in (project_root / "dist", build_dir):
        if folder.exists():
            shutil.rmtree(folder, onerror=_on_rm_error)

    print(f"Created release archive at {archive_path}")


if __name__ == "__main__":
    main()
