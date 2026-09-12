"""Profile cache: ~/.polyserve/profiles/<hardware_hash>/<model_id>/<objective>.json"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from polyserve.hardware import hardware_hash
from polyserve.models import HardwareDescriptor, ModelSpec, Profile

logger = logging.getLogger(__name__)


def polyserve_home() -> Path:
    return Path(os.environ.get("POLYSERVE_HOME", Path.home() / ".polyserve"))


def profiles_dir() -> Path:
    return polyserve_home() / "profiles"


def logs_dir() -> Path:
    return polyserve_home() / "logs"


def profile_path(hw_hash: str, model: ModelSpec, objective: str, workload: str = "default",
                 power: str = "off", phases: str = "unified") -> Path:
    name = objective if workload == "default" else f"{objective}-{workload}"
    if power != "off":
        name += f"-power-{power}"
    if phases != "unified":
        name += f"-phases-{phases}"
    return profiles_dir() / hw_hash / model.safe_id / f"{name}.json"


def save(profile: Profile) -> Path:
    path = profile_path(profile.hardware_hash, ModelSpec(hf_id=profile.model_id), profile.objective, profile.workload,
                        profile.power_mode, profile.phases)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
    logger.info("saved profile %s", path)
    return path


def load(hw: HardwareDescriptor, model: ModelSpec, objective: str, workload: str = "default",
         power: str = "off", phases: str = "unified") -> Optional[Profile]:
    """Return a cached profile if present and still valid for this hardware + backend version."""
    path = profile_path(hardware_hash(hw), model, objective, workload, power, phases)
    if not path.exists():
        return None
    try:
        profile = Profile.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("ignoring unreadable profile %s: %s", path, exc)
        return None
    reason = invalid_reason(profile, hw)
    if reason:
        logger.info("cached profile invalid (%s); recalibrating", reason)
        return None
    return profile


def invalid_reason(profile: Profile, hw: HardwareDescriptor) -> Optional[str]:
    if profile.hardware_hash != hardware_hash(hw):
        return "hardware changed"
    current = hw.backend_version(profile.backend)
    if profile.backend_version != current:
        return f"{profile.backend} version changed ({profile.backend_version} -> {current})"
    if not hw.backend_available(profile.backend):
        return f"{profile.backend} no longer available"
    return None


def delete(hw: HardwareDescriptor, model: ModelSpec, objective: Optional[str] = None,
           workload: str = "default") -> int:
    base = profiles_dir() / hardware_hash(hw) / model.safe_id
    n = 0
    if objective:
        p = profile_path(hardware_hash(hw), model, objective, workload)
        if p.exists():
            p.unlink()
            n = 1
    elif base.exists():
        for p in base.glob("*.json"):
            p.unlink()
            n += 1
    return n


def iter_profiles() -> Iterator[Tuple[Path, Profile]]:
    root = profiles_dir()
    if not root.exists():
        return
    for path in sorted(root.glob("*/*/*.json")):
        try:
            yield path, Profile.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("skipping %s: %s", path, exc)


def list_profiles() -> List[Tuple[Path, Profile]]:
    return list(iter_profiles())
