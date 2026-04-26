from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os
import uuid

from ..config import settings


DEFAULT_CHANNEL_PROFILE: Dict[str, Any] = {
    "channel_name": "",
    "channel_link": "",
    "script_intro_line": "",
    "intro_line": "",
    "description_footer": "",
    "brand_notes": "",
    "social_links": [],
    "useful_links": [],
    "default_hashtags": [],
    "reusable_items": [],
}


class ChannelProfileStore:
    """Persistent channel profiles reused across generations."""

    def __init__(self, profile_path: str | None = None):
        configured = profile_path or os.getenv("CHANNEL_PROFILE_PATH")
        self.profile_path = Path(configured or "data/channel_profile.json")

    def _normalize_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        channel_name = str(payload.get("channel_name", "") or "").strip()
        channel_link = str(payload.get("channel_link", "") or "").strip()
        script_intro_line = str(payload.get("script_intro_line", "") or "").strip()
        intro_line = str(payload.get("intro_line", "") or "").strip()
        description_footer = str(payload.get("description_footer", "") or "").strip()
        brand_notes = str(payload.get("brand_notes", "") or "").strip()

        social_links_raw = payload.get("social_links") or []
        social_links = [str(item).strip() for item in social_links_raw if str(item).strip()]

        useful_links_raw = payload.get("useful_links") or []
        useful_links: List[Dict[str, str]] = []
        seen_links = set()
        for item in useful_links_raw:
            if isinstance(item, dict):
                key = str(item.get("key", "") or "").strip()
                value = str(item.get("value", "") or "").strip()
                if not key or not value:
                    continue
                dedupe_key = (key.lower(), value.lower())
                if dedupe_key in seen_links:
                    continue
                seen_links.add(dedupe_key)
                useful_links.append({"key": key, "value": value})
            else:
                value = str(item).strip()
                if not value:
                    continue
                if value.lower() in seen_links:
                    continue
                seen_links.add(value.lower())
                useful_links.append({"key": "Link", "value": value})

        hashtags_raw = payload.get("default_hashtags") or []
        default_hashtags: List[str] = []
        seen = set()
        for tag in hashtags_raw:
            clean = str(tag).strip()
            if not clean:
                continue
            if not clean.startswith("#"):
                clean = f"#{clean.lstrip('#')}"
            clean = clean.replace(" ", "")
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            default_hashtags.append(clean)

        reusable_items_raw = payload.get("reusable_items") or []
        reusable_items: List[Dict[str, str]] = []
        seen_items = set()
        for item in reusable_items_raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "") or "").strip()
            value = str(item.get("value", "") or "").strip()
            if not key or not value:
                continue
            dedupe_key = (key.lower(), value.lower())
            if dedupe_key in seen_items:
                continue
            seen_items.add(dedupe_key)
            reusable_items.append({"key": key, "value": value})

        return {
            "channel_name": channel_name,
            "channel_link": channel_link,
            "script_intro_line": script_intro_line,
            "intro_line": intro_line,
            "description_footer": description_footer,
            "brand_notes": brand_notes,
            "social_links": social_links,
            "useful_links": useful_links,
            "default_hashtags": default_hashtags,
            "reusable_items": reusable_items,
        }

    def _profile_with_meta(self, payload: Dict[str, Any], profile_id: Optional[str] = None, created_at: Optional[str] = None) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        normalized = self._normalize_profile(payload)
        return {
            "id": profile_id or str(uuid.uuid4()),
            "created_at": created_at or now,
            "updated_at": now,
            **normalized,
        }

    def _load_raw(self) -> Dict[str, Any]:
        if not self.profile_path.exists():
            return {"profiles": []}

        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except Exception:
            return {"profiles": []}

        # Legacy format: plain single-profile dict
        if isinstance(data, dict) and "profiles" not in data:
            profile = self._profile_with_meta(data, profile_id="default")
            return {"profiles": [profile]}

        if not isinstance(data, dict):
            return {"profiles": []}

        raw_profiles = data.get("profiles") or []
        profiles: List[Dict[str, Any]] = []
        for p in raw_profiles:
            if not isinstance(p, dict):
                continue
            normalized = self._profile_with_meta(
                p,
                profile_id=str(p.get("id") or str(uuid.uuid4())),
                created_at=str(p.get("created_at") or datetime.utcnow().isoformat()),
            )
            profiles.append(normalized)

        return {"profiles": profiles}

    def _save_raw(self, raw: Dict[str, Any]) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    # Backward-compatible single-profile methods
    def load(self) -> Dict[str, Any]:
        profiles = self.list_profiles()
        if not profiles:
            return DEFAULT_CHANNEL_PROFILE.copy()
        first = profiles[0].copy()
        first.pop("id", None)
        first.pop("created_at", None)
        first.pop("updated_at", None)
        return first

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profiles = self.list_profiles()
        if not profiles:
            created = self.create_profile(payload)
            out = created.copy()
        else:
            out = self.update_profile(profiles[0]["id"], payload)
        out.pop("id", None)
        out.pop("created_at", None)
        out.pop("updated_at", None)
        return out

    # Multi-profile API
    def list_profiles(self) -> List[Dict[str, Any]]:
        return self._load_raw().get("profiles", [])

    def get_profile(self, profile_id: Optional[str]) -> Dict[str, Any]:
        if not profile_id:
            return {}

        for profile in self.list_profiles():
            if profile.get("id") == profile_id:
                return profile

        return {}

    def create_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._load_raw()
        profile = self._profile_with_meta(payload)
        
        channel_name = profile.get("channel_name", "").strip()
        if not channel_name:
            profile["channel_name"] = f"Channel Profile {len(raw.get('profiles', [])) + 1}"
        else:
            existing_names = [p.get("channel_name", "").strip().lower() for p in raw.get("profiles", [])]
            if channel_name.lower() in existing_names:
                raise ValueError(f"A profile with the name '{channel_name}' already exists")
        
        raw_profiles = raw.get("profiles") or []
        raw_profiles.append(profile)
        raw["profiles"] = raw_profiles
        self._save_raw(raw)
        return profile

    def update_profile(self, profile_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._load_raw()
        profiles = raw.get("profiles") or []
        updated_profile: Optional[Dict[str, Any]] = None

        for index, existing in enumerate(profiles):
            if existing.get("id") != profile_id:
                continue
            merged = {**existing, **(payload or {})}
            updated_profile = self._profile_with_meta(
                merged,
                profile_id=profile_id,
                created_at=existing.get("created_at"),
            )
            if not updated_profile.get("channel_name"):
                updated_profile["channel_name"] = existing.get("channel_name") or "Untitled Channel Profile"
            else:
                channel_name = updated_profile.get("channel_name", "").strip()
                existing_names = [p.get("channel_name", "").strip().lower() for p in profiles if p.get("id") != profile_id]
                if channel_name.lower() in existing_names:
                    raise ValueError(f"A profile with the name '{channel_name}' already exists")
            profiles[index] = updated_profile
            break

        if updated_profile is None:
            raise KeyError(f"Channel profile not found: {profile_id}")

        raw["profiles"] = profiles
        self._save_raw(raw)
        return updated_profile

    def delete_profile(self, profile_id: str) -> bool:
        raw = self._load_raw()
        profiles = raw.get("profiles") or []
        filtered = [p for p in profiles if p.get("id") != profile_id]
        if len(filtered) == len(profiles):
            return False
        raw["profiles"] = filtered
        self._save_raw(raw)
        return True


channel_profile_store = ChannelProfileStore(
    profile_path=getattr(settings, "channel_profile_path", "data/channel_profile.json")
)


def _dedupe_in_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in items:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value.strip())
    return out


def build_channel_context_text(profile: Dict[str, Any]) -> str:
    reusable_items = profile.get("reusable_items") or []
    reusable_lines = []
    for item in reusable_items:
        if isinstance(item, dict):
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if key and value:
                reusable_lines.append(f"  - {key}: {value}")

    useful_links = profile.get("useful_links") or []
    useful_links_lines = []
    for item in useful_links:
        if isinstance(item, dict):
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if key and value:
                useful_links_lines.append(f"  - {key}: {value}")

    social_links = profile.get("social_links") or []
    social_links_text = ", ".join(social_links) if social_links else "Not set"

    default_hashtags = profile.get("default_hashtags") or []
    hashtags_text = " ".join(default_hashtags) if default_hashtags else "Not set"

    return (
        "Channel profile context:\n"
        f"- Channel name: {profile.get('channel_name') or 'Not set'}\n"
        f"- Channel link: {profile.get('channel_link') or 'Not set'}\n"
        f"- Script intro line: {profile.get('script_intro_line') or 'Not set'}\n"
        f"- Description intro line: {profile.get('intro_line') or 'Not set'}\n"
        f"- Description footer: {profile.get('description_footer') or 'Not set'}\n"
        f"- Brand notes: {profile.get('brand_notes') or 'Not set'}\n"
        f"- Social links: {social_links_text}\n"
        f"- Useful links:\n{chr(10).join(useful_links_lines) if useful_links_lines else '  Not set'}\n"
        f"- Default hashtags: {hashtags_text}\n"
        f"- Reusable text/link items:\n{chr(10).join(reusable_lines) if reusable_lines else '  Not set'}"
    )
