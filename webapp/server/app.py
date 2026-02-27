import hashlib
import hmac
import json
import mimetypes
import os
import re
import time
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import httpx
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import DESCENDING, MongoClient
from starlette.background import BackgroundTask


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "8537979650:AAFkSIbRnx7ha7muxZ1MDK5QMIxV5MAC4ww").strip()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:wGVAMNxMWZgocdRVBduRDnRlJePweOay@metro.proxy.rlwy.net:36399").strip()
MONGODB_DB = os.getenv("MONGODB_DB", "kino_bot").strip() or "kino_bot"
BOT_USERNAME = os.getenv("BOT_USERNAME", "@MirTopKinoBot").strip()
WEBAPP_ALLOWED_ORIGINS_RAW = [item.strip() for item in os.getenv("WEBAPP_ALLOWED_ORIGINS", "*").split(",") if item.strip()]
WEBAPP_AUTH_MAX_AGE_SECONDS = int(os.getenv("WEBAPP_AUTH_MAX_AGE_SECONDS", "604800"))
WEBAPP_ENABLE_DEV_AUTH = os.getenv("WEBAPP_ENABLE_DEV_AUTH", "0").strip() in {"1", "true", "yes"}
WEBAPP_DEV_USER_ID = int(os.getenv("WEBAPP_DEV_USER_ID", "0") or 0)
WEBAPP_MEDIA_TOKEN_TTL_SECONDS = int(os.getenv("WEBAPP_MEDIA_TOKEN_TTL_SECONDS", "604800"))
WEBAPP_SUBSCRIPTION_CACHE_SECONDS = int(os.getenv("WEBAPP_SUBSCRIPTION_CACHE_SECONDS", "180"))
ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required for webapp backend.")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is required for webapp backend.")


def _parse_admin_ids(value: str) -> set[int]:
    rows: set[int] = set()
    for part in str(value or "").replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            rows.add(int(text))
        except ValueError:
            continue
    return rows


def _normalize_origins(items: list[str]) -> list[str]:
    if not items:
        return ["*"]
    if "*" in items:
        return ["*"]
    normalized: list[str] = []
    for item in items:
        if item.startswith(("http://", "https://")):
            normalized.append(item.rstrip("/"))
        else:
            normalized.append(f"https://{item.rstrip('/')}")
    return normalized


WEBAPP_ALLOWED_ORIGINS = _normalize_origins(WEBAPP_ALLOWED_ORIGINS_RAW)
ADMIN_IDS = _parse_admin_ids(ADMIN_IDS_RAW)


client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
db = client[MONGODB_DB]

users_col = db["users"]
required_channels_col = db["required_channels"]
join_requests_col = db["join_requests"]
movies_col = db["movies"]
serials_col = db["serials"]
shorts_col = db["shorts"]
serial_episodes_col = db["serial_episodes"]
favorites_col = db["favorites"]
reactions_col = db["web_reactions"]
comments_col = db["web_comments"]
history_col = db["web_history"]
downloads_col = db["web_downloads"]
watch_progress_col = db["web_watch_progress"]
admins_col = db["admins"]
comment_reactions_col = db["web_comment_reactions"]

reactions_col.create_index([("user_tg_id", 1), ("content_type", 1), ("content_ref", 1)], unique=True)
comments_col.create_index([("content_type", 1), ("content_ref", 1), ("created_at", -1)])
comments_col.create_index([("parent_comment_id", 1), ("created_at", -1)])
history_col.create_index([("user_tg_id", 1), ("viewed_at", -1)])
downloads_col.create_index([("user_tg_id", 1), ("created_at", -1)])
watch_progress_col.create_index(
    [("user_tg_id", 1), ("content_type", 1), ("content_ref", 1), ("episode_number", 1)],
    unique=True,
)
watch_progress_col.create_index([("user_tg_id", 1), ("updated_at", -1)])
comment_reactions_col.create_index([("comment_id", 1), ("reaction", 1)])
comment_reactions_col.create_index([("user_tg_id", 1), ("comment_id", 1)], unique=True)
shorts_col.create_index([("created_at", -1)])


def _seed_admins_from_env() -> None:
    if not ADMIN_IDS:
        return
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for admin_id in ADMIN_IDS:
        admins_col.update_one(
            {"tg_id": int(admin_id)},
            {"$setOnInsert": {"tg_id": int(admin_id), "added_at": now_iso}},
            upsert=True,
        )


_seed_admins_from_env()

BASE_DIR = Path(__file__).resolve().parent
CLIENT_DIST_DIR = BASE_DIR.parent / "client" / "dist"
CLIENT_ASSETS_DIR = CLIENT_DIST_DIR / "assets"


app = FastAPI(title="Kino Bot WebApp API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=WEBAPP_ALLOWED_ORIGINS if WEBAPP_ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if CLIENT_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(CLIENT_ASSETS_DIR)), name="client-assets")


MEMBER_STATUSES = {"creator", "administrator", "member", "restricted"}
TELEGRAM_API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
TELEGRAM_FILE_BASE = f"https://api.telegram.org/file/bot{BOT_TOKEN}"
HTTP_TIMEOUT = httpx.Timeout(20.0)
_BOT_USERNAME_CACHE: str | None = BOT_USERNAME.lstrip("@") or None
CONTENT_PROJECTION = {
    "code": 1,
    "title": 1,
    "description": 1,
    "year": 1,
    "quality": 1,
    "genres": 1,
    "downloads": 1,
    "media_type": 1,
    "file_id": 1,
    "preview_media_type": 1,
    "preview_file_id": 1,
    "preview_photo_file_id": 1,
    "trailer_url": 1,
    "is_active": 1,
    "created_at": 1,
}
VALID_CONTENT_TYPES = {"movie", "serial", "short"}

FILE_PATH_CACHE_TTL_SECONDS = 900
_FILE_PATH_CACHE: dict[str, tuple[str, float]] = {}
SERIAL_PREVIEW_CACHE_TTL_SECONDS = 900
_SERIAL_PREVIEW_CACHE: dict[str, tuple[str, str, float]] = {}
_SUBSCRIPTION_CACHE: dict[int, tuple[list[dict[str, Any]], float]] = {}


class AdminEpisodeIn(BaseModel):
    episode_number: int
    media_type: str = "video"
    file_id: str


class FavoriteToggleIn(BaseModel):
    content_type: str
    content_ref: str


class ReactionIn(BaseModel):
    content_type: str
    content_ref: str
    reaction: str


class CommentIn(BaseModel):
    content_type: str
    content_ref: str
    text: str
    parent_comment_id: str | None = None


class DownloadTrackIn(BaseModel):
    content_type: str
    content_ref: str


class WatchProgressIn(BaseModel):
    content_type: str
    content_ref: str
    episode_number: int | None = None
    position_seconds: int
    duration_seconds: int = 0


class AdminToggleIn(BaseModel):
    content_type: str
    content_ref: str
    is_active: bool


class CommentReactionIn(BaseModel):
    comment_id: str
    reaction: str


class AdminCreateIn(BaseModel):
    content_type: str
    title: str
    code: str = ""
    description: str = ""
    year: int | None = None
    quality: str = ""
    genres: list[str] = []
    media_type: str = "video"
    file_id: str = ""
    preview_media_type: str = "photo"
    preview_file_id: str = ""
    preview_photo_file_id: str = ""
    trailer_url: str = ""
    is_active: bool = True
    episodes: list[AdminEpisodeIn] = []


def _normalize_lookup(value: str) -> str:
    cleaned = re.sub(r"[^\w\s]+", " ", value.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _make_join_url(channel_ref: str, join_link: str | None) -> str | None:
    if join_link:
        return join_link
    if channel_ref.startswith("@"):
        return f"https://t.me/{channel_ref[1:]}"
    return None


def _is_subscribed_status(status: str) -> bool:
    return status in MEMBER_STATUSES


def _doc_id(doc: dict[str, Any]) -> str:
    return str(doc.get("_id"))


def _is_active(doc: dict[str, Any] | None) -> bool:
    if not doc:
        return False
    return bool(doc.get("is_active", True))


def _is_video_media_type(media_type: str) -> bool:
    value = (media_type or "").strip().lower()
    if not value:
        return True
    if value.startswith("video/"):
        return True
    if value.startswith("image/"):
        return False
    if value in {
        "video",
        "document",
        "file",
        "animation",
        "mp4",
        "mkv",
        "mov",
        "avi",
        "webm",
        "m4v",
        "wmv",
        "mpeg",
    }:
        return True
    if value in {"photo", "image", "sticker", "gif"}:
        return False
    return True


def _extract_preview_photo_file_id(content_type: str, doc: dict[str, Any]) -> str:
    preview_photo = str(doc.get("preview_photo_file_id") or "")
    if preview_photo:
        return preview_photo
    preview_media_type = str(doc.get("preview_media_type") or "")
    preview_file_id = str(doc.get("preview_file_id") or "")
    if preview_media_type == "photo" and preview_file_id:
        return preview_file_id
    if content_type == "movie":
        media_type = str(doc.get("media_type") or "")
        file_id = str(doc.get("file_id") or "")
        if media_type == "photo" and file_id:
            return file_id
    return ""


def _serialize_content(doc: dict[str, Any], content_type: str, bot_username: str | None) -> dict[str, Any]:
    content_id = _doc_id(doc)
    if content_type == "movie":
        payload = f"m_{content_id}"
    elif content_type == "serial":
        payload = f"s_{content_id}"
    else:
        payload = f"sh_{content_id}"
    deep_link = f"https://t.me/{bot_username}?start={payload}" if bot_username else ""
    preview_file_id = _extract_preview_photo_file_id(content_type, doc)
    media_type = str(doc.get("media_type") or "")
    file_id = str(doc.get("file_id") or "")
    if content_type == "serial" and not file_id:
        serial_preview_file_id, serial_preview_media_type = _serial_preview_stream(content_id)
        if serial_preview_file_id:
            file_id = serial_preview_file_id
            if not media_type:
                media_type = serial_preview_media_type
    code = str(doc.get("code") or "")
    quality = str(doc.get("quality") or "")
    genres = [str(g) for g in doc.get("genres", []) if str(g).strip()]
    if content_type == "short":
        if not code:
            code = f"SH-{content_id[:6].upper()}"
        if not quality:
            quality = "Short"
        if not genres:
            genres = ["Shorts"]
    return {
        "id": content_id,
        "content_type": content_type,
        "code": code,
        "title": str(doc.get("title") or ""),
        "description": str(doc.get("description") or ""),
        "year": doc.get("year"),
        "quality": quality,
        "genres": genres,
        "downloads": int(doc.get("downloads") or 0),
        "preview_file_id": preview_file_id,
        "media_type": media_type,
        "file_id": file_id,
        "is_video": _is_video_media_type(media_type),
        "is_active": bool(doc.get("is_active", True)),
        "trailer_url": str(doc.get("trailer_url") or ""),
        "deep_link": deep_link,
        "created_at": str(doc.get("created_at") or ""),
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _now_ts() -> float:
    return time.time()


def _iso_to_epoch(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))
    except Exception:
        return 0.0


def _safe_object_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except Exception:
        return None


def _guess_media_type(file_path: str, fallback: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(str(file_path or "").strip())
    if guessed:
        return guessed
    return fallback


def _serial_preview_stream(serial_id: str) -> tuple[str, str]:
    serial_ref = str(serial_id or "").strip()
    if not serial_ref:
        return "", ""
    cached = _SERIAL_PREVIEW_CACHE.get(serial_ref)
    now_ts = _now_ts()
    if cached and cached[2] > now_ts:
        return cached[0], cached[1]

    oid = _safe_object_id(serial_ref)
    if not oid:
        return "", ""
    doc = serial_episodes_col.find_one(
        {"serial_id": oid, "file_id": {"$exists": True, "$ne": ""}},
        {"file_id": 1, "media_type": 1},
        sort=[("episode_number", 1)],
    ) or {}
    file_id = str(doc.get("file_id") or "")
    media_type = str(doc.get("media_type") or "video")
    _SERIAL_PREVIEW_CACHE[serial_ref] = (
        file_id,
        media_type,
        now_ts + SERIAL_PREVIEW_CACHE_TTL_SECONDS,
    )
    return file_id, media_type


async def _resolve_file_path(file_id: str) -> str:
    safe_file_id = str(file_id or "").strip()
    if not safe_file_id:
        return ""
    cached = _FILE_PATH_CACHE.get(safe_file_id)
    now_ts = _now_ts()
    if cached and cached[1] > now_ts:
        return cached[0]

    result = await _telegram_api("getFile", {"file_id": safe_file_id})
    file_path = str(result.get("file_path") or "").strip()
    if file_path:
        _FILE_PATH_CACHE[safe_file_id] = (
            file_path,
            now_ts + FILE_PATH_CACHE_TTL_SECONDS,
        )
    return file_path


def _get_content_doc(content_type: str, content_ref: str, include_inactive: bool = False) -> dict[str, Any] | None:
    oid = _safe_object_id(content_ref)
    if not oid:
        return None
    if content_type == "movie":
        doc = movies_col.find_one({"_id": oid})
        if doc and (include_inactive or _is_active(doc)):
            return doc
        return None
    if content_type == "serial":
        doc = serials_col.find_one({"_id": oid})
        if doc and (include_inactive or _is_active(doc)):
            return doc
        return None
    if content_type == "short":
        doc = shorts_col.find_one({"_id": oid})
        if doc and (include_inactive or _is_active(doc)):
            return doc
        return None
    return None


def _content_key(content_type: str, content_ref: str) -> str:
    return f"{content_type}:{content_ref}"


def _content_match_clauses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = {"movie": set(), "serial": set(), "short": set()}
    for item in items:
        content_type = str(item.get("content_type") or "").strip()
        content_ref = str(item.get("id") or item.get("content_ref") or "").strip()
        if content_type in grouped and content_ref:
            grouped[content_type].add(content_ref)
    clauses: list[dict[str, Any]] = []
    for content_type, refs in grouped.items():
        if refs:
            clauses.append({"content_type": content_type, "content_ref": {"$in": sorted(refs)}})
    return clauses


def _count_by_content(collection: Any, match: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": {"content_type": "$content_type", "content_ref": "$content_ref"},
                "count": {"$sum": 1},
            }
        },
    ]
    for row in collection.aggregate(pipeline):
        row_id = row.get("_id") or {}
        content_type = str(row_id.get("content_type") or "")
        content_ref = str(row_id.get("content_ref") or "")
        if not content_type or not content_ref:
            continue
        result[_content_key(content_type, content_ref)] = int(row.get("count") or 0)
    return result


def _engagement_maps(items: list[dict[str, Any]], user_tg_id: int) -> dict[str, dict[str, Any]]:
    clauses = _content_match_clauses(items)
    if not clauses:
        return {
            "likes": {},
            "dislikes": {},
            "comments": {},
            "downloads": {},
            "user_reaction": {},
        }

    likes = _count_by_content(reactions_col, {"reaction": "like", "$or": clauses})
    dislikes = _count_by_content(reactions_col, {"reaction": "dislike", "$or": clauses})
    comments = _count_by_content(comments_col, {"$or": clauses})
    downloads = _count_by_content(downloads_col, {"$or": clauses})

    user_reaction: dict[str, str] = {}
    for row in reactions_col.find(
        {
            "user_tg_id": user_tg_id,
            "$or": clauses,
        },
        {"content_type": 1, "content_ref": 1, "reaction": 1},
    ):
        content_type = str(row.get("content_type") or "")
        content_ref = str(row.get("content_ref") or "")
        if not content_type or not content_ref:
            continue
        user_reaction[_content_key(content_type, content_ref)] = str(row.get("reaction") or "")

    return {
        "likes": likes,
        "dislikes": dislikes,
        "comments": comments,
        "downloads": downloads,
        "user_reaction": user_reaction,
    }


def _apply_engagement_maps(items: list[dict[str, Any]], maps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    likes_map = maps.get("likes") or {}
    dislikes_map = maps.get("dislikes") or {}
    comments_map = maps.get("comments") or {}
    downloads_map = maps.get("downloads") or {}
    user_reaction_map = maps.get("user_reaction") or {}
    enriched: list[dict[str, Any]] = []
    for item in items:
        content_type = str(item.get("content_type") or "").strip()
        content_ref = str(item.get("id") or item.get("content_ref") or "").strip()
        if not content_type or not content_ref:
            enriched.append(item)
            continue
        key = _content_key(content_type, content_ref)
        row = dict(item)
        row.update(
            {
                "likes": int(likes_map.get(key, 0)),
                "dislikes": int(dislikes_map.get(key, 0)),
                "comments": int(comments_map.get(key, 0)),
                "downloads_tracked": int(downloads_map.get(key, 0)),
                "user_reaction": str(user_reaction_map.get(key, "")),
            }
        )
        enriched.append(row)
    return enriched


def _engagement_summary(content_type: str, content_ref: str, user_tg_id: int) -> dict[str, Any]:
    key = _content_key(content_type, content_ref)
    maps = _engagement_maps([{"content_type": content_type, "id": content_ref}], user_tg_id)
    return {
        "likes": int(maps["likes"].get(key, 0)),
        "dislikes": int(maps["dislikes"].get(key, 0)),
        "comments": int(maps["comments"].get(key, 0)),
        "downloads_tracked": int(maps["downloads"].get(key, 0)),
        "user_reaction": str(maps["user_reaction"].get(key, "")),
    }


def _attach_engagement(items: list[dict[str, Any]], user_tg_id: int) -> list[dict[str, Any]]:
    maps = _engagement_maps(items, user_tg_id)
    enriched: list[dict[str, Any]] = []
    for item in items:
        content_type = str(item.get("content_type") or "").strip()
        content_ref = str(item.get("id") or item.get("content_ref") or "").strip()
        if not content_type or not content_ref:
            enriched.append(item)
            continue
        key = _content_key(content_type, content_ref)
        row = dict(item)
        row.update(
            {
                "likes": int(maps["likes"].get(key, 0)),
                "dislikes": int(maps["dislikes"].get(key, 0)),
                "comments": int(maps["comments"].get(key, 0)),
                "downloads_tracked": int(maps["downloads"].get(key, 0)),
                "user_reaction": str(maps["user_reaction"].get(key, "")),
            }
        )
        enriched.append(row)
    return enriched


def _serialize_comment(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc.get("_id")),
        "content_type": str(doc.get("content_type") or ""),
        "content_ref": str(doc.get("content_ref") or ""),
        "parent_comment_id": str(doc.get("parent_comment_id") or ""),
        "user_tg_id": int(doc.get("user_tg_id") or 0),
        "full_name": str(doc.get("full_name") or ""),
        "username": str(doc.get("username") or ""),
        "text": str(doc.get("text") or ""),
        "created_at": str(doc.get("created_at") or ""),
    }


def _comment_reaction_maps(comment_ids: list[str], user_tg_id: int) -> dict[str, dict[str, Any]]:
    safe_ids = sorted({str(value).strip() for value in comment_ids if str(value).strip()})
    if not safe_ids:
        return {"likes": {}, "dislikes": {}, "user_reaction": {}}

    likes: dict[str, int] = {}
    dislikes: dict[str, int] = {}
    for row in comment_reactions_col.find(
        {"comment_id": {"$in": safe_ids}, "reaction": {"$in": ["like", "dislike"]}},
        {"comment_id": 1, "reaction": 1},
    ):
        comment_id = str(row.get("comment_id") or "")
        reaction = str(row.get("reaction") or "")
        if not comment_id:
            continue
        if reaction == "like":
            likes[comment_id] = likes.get(comment_id, 0) + 1
        elif reaction == "dislike":
            dislikes[comment_id] = dislikes.get(comment_id, 0) + 1

    user_reaction: dict[str, str] = {}
    for row in comment_reactions_col.find(
        {"user_tg_id": user_tg_id, "comment_id": {"$in": safe_ids}},
        {"comment_id": 1, "reaction": 1},
    ):
        comment_id = str(row.get("comment_id") or "")
        if comment_id:
            user_reaction[comment_id] = str(row.get("reaction") or "")

    return {"likes": likes, "dislikes": dislikes, "user_reaction": user_reaction}


def _comment_reaction_summary(comment_id: str, user_tg_id: int) -> dict[str, Any]:
    comment_id_safe = str(comment_id or "").strip()
    maps = _comment_reaction_maps([comment_id_safe], user_tg_id)
    return {
        "likes": int((maps.get("likes") or {}).get(comment_id_safe, 0)),
        "dislikes": int((maps.get("dislikes") or {}).get(comment_id_safe, 0)),
        "user_reaction": str((maps.get("user_reaction") or {}).get(comment_id_safe, "")),
    }


def _comment_score_for_top(comment: dict[str, Any], now_ts: float) -> float:
    likes = int(comment.get("likes") or 0)
    dislikes = int(comment.get("dislikes") or 0)
    reply_count = int(comment.get("reply_count") or 0)
    created_epoch = _iso_to_epoch(str(comment.get("created_at") or ""))
    recency_boost = max(0.0, (72 * 3600) - max(0.0, now_ts - created_epoch)) / 3600
    return (likes * 2.5) - (dislikes * 1.4) + (reply_count * 2.8) + recency_boost


def _list_comment_threads(
    content_type: str,
    content_ref: str,
    user_tg_id: int,
    limit: int,
    sort: str,
    is_admin: bool = False,
) -> list[dict[str, Any]]:
    limit_safe = max(1, min(limit, 120))
    top_filter = {
        "content_type": content_type,
        "content_ref": content_ref,
        "$or": [{"parent_comment_id": {"$exists": False}}, {"parent_comment_id": None}, {"parent_comment_id": ""}],
    }
    scan_limit = max(limit_safe * 4, 80)
    direction = DESCENDING if sort != "old" else 1
    top_docs = list(comments_col.find(top_filter).sort("created_at", direction).limit(scan_limit))
    if not top_docs:
        return []

    top_ids = [str(doc.get("_id")) for doc in top_docs if doc.get("_id")]
    replies_docs = list(
        comments_col.find({"content_type": content_type, "content_ref": content_ref, "parent_comment_id": {"$in": top_ids}})
        .sort("created_at", 1)
        .limit(max(scan_limit * 6, 180))
    )
    replies_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in replies_docs:
        parent_id = str(row.get("parent_comment_id") or "")
        if not parent_id:
            continue
        replies_by_parent.setdefault(parent_id, []).append(row)

    all_ids = top_ids + [str(row.get("_id")) for row in replies_docs if row.get("_id")]
    reaction_maps = _comment_reaction_maps(all_ids, user_tg_id)

    def _build_comment_payload(row: dict[str, Any], reply_count: int = 0) -> dict[str, Any]:
        payload = _serialize_comment(row)
        comment_id = payload["id"]
        owner_id = int(payload.get("user_tg_id") or 0)
        payload.update(
            {
                "likes": int((reaction_maps.get("likes") or {}).get(comment_id, 0)),
                "dislikes": int((reaction_maps.get("dislikes") or {}).get(comment_id, 0)),
                "user_reaction": str((reaction_maps.get("user_reaction") or {}).get(comment_id, "")),
                "reply_count": int(reply_count),
                "is_owner": owner_id == user_tg_id,
                "can_delete": owner_id == user_tg_id or is_admin,
                "replies": [],
            }
        )
        return payload

    items: list[dict[str, Any]] = []
    for row in top_docs:
        comment_id = str(row.get("_id") or "")
        reply_rows = replies_by_parent.get(comment_id, [])
        payload = _build_comment_payload(row, reply_count=len(reply_rows))
        payload["replies"] = [_build_comment_payload(reply_row) for reply_row in reply_rows]
        items.append(payload)

    if sort == "top":
        now_ts = _now_ts()
        items.sort(
            key=lambda x: (
                _comment_score_for_top(x, now_ts),
                str(x.get("created_at") or ""),
            ),
            reverse=True,
        )
    elif sort == "old":
        items.sort(key=lambda x: str(x.get("created_at") or ""))
    else:
        items.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return items[:limit_safe]


def _track_history_view(user_tg_id: int, content_type: str, content_ref: str) -> None:
    if content_type not in VALID_CONTENT_TYPES or not content_ref:
        return
    history_col.insert_one(
        {
            "user_tg_id": user_tg_id,
            "content_type": content_type,
            "content_ref": content_ref,
            "viewed_at": _now_iso(),
        }
    )


def _list_recent_history(user_tg_id: int, bot_username: str | None, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in history_col.find({"user_tg_id": user_tg_id}).sort("viewed_at", DESCENDING).limit(max(1, limit * 4)):
        content_type = str(event.get("content_type") or "")
        content_ref = str(event.get("content_ref") or "")
        key = f"{content_type}:{content_ref}"
        if key in seen:
            continue
        seen.add(key)
        if content_type not in VALID_CONTENT_TYPES:
            continue
        doc = _get_content_doc(content_type, content_ref)
        if not doc:
            continue
        rows.append(_serialize_content(doc, content_type, bot_username))
        if len(rows) >= max(1, limit):
            break
    return _attach_engagement(rows, user_tg_id)


def _serialize_episode(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc.get("_id") or ""),
        "episode_number": int(doc.get("episode_number") or 0),
        "media_type": str(doc.get("media_type") or ""),
        "file_id": str(doc.get("file_id") or ""),
        "is_video": _is_video_media_type(str(doc.get("media_type") or "")),
    }


def _list_serial_episodes(serial_ref: str) -> list[dict[str, Any]]:
    serial_oid = _safe_object_id(serial_ref)
    if not serial_oid:
        return []
    rows: list[dict[str, Any]] = []
    for doc in serial_episodes_col.find(
        {"serial_id": serial_oid},
        {"episode_number": 1, "media_type": 1, "file_id": 1},
    ).sort("episode_number", 1):
        rows.append(_serialize_episode(doc))
    return rows


def _progress_key(content_type: str, content_ref: str, episode_number: int | None) -> dict[str, Any]:
    return {
        "content_type": content_type,
        "content_ref": content_ref,
        "episode_number": int(episode_number) if episode_number is not None else None,
    }


def _get_watch_progress(user_tg_id: int, content_type: str, content_ref: str, episode_number: int | None) -> dict[str, Any]:
    criteria = {"user_tg_id": user_tg_id}
    criteria.update(_progress_key(content_type, content_ref, episode_number))
    row = watch_progress_col.find_one(criteria) or {}
    return {
        "position_seconds": int(row.get("position_seconds") or 0),
        "duration_seconds": int(row.get("duration_seconds") or 0),
        "updated_at": str(row.get("updated_at") or ""),
    }


def _list_continue_watching(user_tg_id: int, bot_username: str | None, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for progress in watch_progress_col.find({"user_tg_id": user_tg_id}).sort("updated_at", DESCENDING).limit(max(1, limit * 3)):
        content_type = str(progress.get("content_type") or "")
        content_ref = str(progress.get("content_ref") or "")
        if content_type not in VALID_CONTENT_TYPES or not content_ref:
            continue
        doc = _get_content_doc(content_type, content_ref)
        if not doc:
            continue
        item = _serialize_content(doc, content_type, bot_username)
        item["watch_progress"] = {
            "position_seconds": int(progress.get("position_seconds") or 0),
            "duration_seconds": int(progress.get("duration_seconds") or 0),
            "episode_number": progress.get("episode_number"),
            "updated_at": str(progress.get("updated_at") or ""),
        }
        rows.append(item)
        if len(rows) >= max(1, limit):
            break
    return _attach_engagement(rows, user_tg_id)


def _query_filter(query: str) -> dict[str, Any]:
    value = query.strip()
    if not value:
        return {}
    regex = {"$regex": re.escape(value), "$options": "i"}
    return {
        "$or": [
            {"code": regex},
            {"title": regex},
            {"description": regex},
        ]
    }


def _with_active_filter(base_filter: dict[str, Any]) -> dict[str, Any]:
    if not base_filter:
        return {"is_active": {"$ne": False}}
    return {"$and": [base_filter, {"is_active": {"$ne": False}}]}


def _similarity_score(query_norm: str, target_norm: str) -> float:
    if not query_norm or not target_norm:
        return 0.0
    ratio = SequenceMatcher(None, query_norm, target_norm).ratio() * 100
    query_tokens = [token for token in query_norm.split() if len(token) >= 2]
    target_tokens = [token for token in target_norm.split() if token]
    token_hits = 0
    for token in query_tokens:
        if any(token in item or item.startswith(token) for item in target_tokens):
            token_hits += 1
    return ratio + (min(token_hits, 4) * 6)


def _fuzzy_content_fill(
    items: list[dict[str, Any]],
    query: str,
    content_type: str,
    limit: int,
    bot_username: str | None,
) -> list[dict[str, Any]]:
    value = query.strip()
    if not value:
        return items
    query_norm = _normalize_lookup(value)
    if len(query_norm) < 3:
        return items

    limit_safe = max(1, min(limit, 200))
    if len(items) >= limit_safe:
        return items[:limit_safe]

    existing_keys = {
        _content_key(str(row.get("content_type") or ""), str(row.get("id") or row.get("content_ref") or ""))
        for row in items
    }
    scored_rows: list[dict[str, Any]] = []
    scan_limit = max(limit_safe * 8, 180)
    collections: list[tuple[str, Any]] = []
    if content_type in {"all", "movie"}:
        collections.append(("movie", movies_col))
    if content_type in {"all", "serial"}:
        collections.append(("serial", serials_col))

    for current_type, collection in collections:
        docs = collection.find({"is_active": {"$ne": False}}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit)
        for doc in docs:
            item = _serialize_content(doc, current_type, bot_username)
            key = _content_key(current_type, str(item.get("id") or ""))
            if key in existing_keys:
                continue
            title_norm = _normalize_lookup(str(item.get("title") or ""))
            code_norm = _normalize_lookup(str(item.get("code") or ""))
            score = max(_similarity_score(query_norm, title_norm), _similarity_score(query_norm, code_norm))
            if query_norm in title_norm and title_norm:
                score += 20
            if score < 50:
                continue
            row = dict(item)
            row["fuzzy_score"] = float(score)
            scored_rows.append(row)

    scored_rows.sort(key=lambda x: (float(x.get("fuzzy_score") or 0), str(x.get("created_at") or "")), reverse=True)
    for row in scored_rows:
        row.pop("fuzzy_score", None)
        key = _content_key(str(row.get("content_type") or ""), str(row.get("id") or ""))
        if key in existing_keys:
            continue
        items.append(row)
        existing_keys.add(key)
        if len(items) >= limit_safe:
            break

    return items[:limit_safe]


def _list_content(
    query: str,
    content_type: str,
    limit: int,
    bot_username: str | None,
    user_tg_id: int,
) -> list[dict[str, Any]]:
    limit_safe = max(1, min(limit, 200))
    mongo_filter = _with_active_filter(_query_filter(query))
    rows: list[dict[str, Any]] = []

    if content_type in {"all", "movie"}:
        for doc in movies_col.find(mongo_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
            rows.append(_serialize_content(doc, "movie", bot_username))

    if content_type in {"all", "serial"}:
        for doc in serials_col.find(mongo_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
            rows.append(_serialize_content(doc, "serial", bot_username))

    if content_type in {"all", "short"}:
        for doc in shorts_col.find(mongo_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
            rows.append(_serialize_content(doc, "short", bot_username))

    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    rows = _fuzzy_content_fill(rows, query=query, content_type=content_type, limit=limit_safe, bot_username=bot_username)
    return _attach_engagement(rows[:limit_safe], user_tg_id)


def _list_favorites(user_tg_id: int, bot_username: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fav in favorites_col.find({"user_tg_id": user_tg_id}).sort("created_at", DESCENDING):
        content_type = str(fav.get("content_type") or "")
        content_ref = str(fav.get("content_ref") or "")
        doc = _get_content_doc(content_type, content_ref)
        if doc:
            rows.append(_serialize_content(doc, content_type, bot_username))
    return _attach_engagement(rows, user_tg_id)


def _list_trending(user_tg_id: int, bot_username: str | None, limit: int = 18) -> list[dict[str, Any]]:
    limit_safe = max(1, min(limit, 60))
    scan_limit = max(120, limit_safe * 10)
    rows: list[dict[str, Any]] = []

    active_filter = {"is_active": {"$ne": False}}
    for doc in movies_col.find(active_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
        rows.append(_serialize_content(doc, "movie", bot_username))
    for doc in serials_col.find(active_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
        rows.append(_serialize_content(doc, "serial", bot_username))

    enriched = _attach_engagement(rows, user_tg_id)
    for item in enriched:
        base_downloads = int(item.get("downloads") or 0)
        likes = int(item.get("likes") or 0)
        dislikes = int(item.get("dislikes") or 0)
        comments = int(item.get("comments") or 0)
        tracked_downloads = int(item.get("downloads_tracked") or 0)
        score = (base_downloads * 2) + (likes * 8) + (comments * 5) + (tracked_downloads * 3) - (dislikes * 2)
        item["trend_score"] = max(score, 0)

    enriched.sort(key=lambda x: (int(x.get("trend_score") or 0), str(x.get("created_at") or "")), reverse=True)
    return enriched[:limit_safe]


def _list_shorts(
    user_tg_id: int,
    bot_username: str | None,
    limit: int = 24,
    include_inactive: bool = False,
) -> list[dict[str, Any]]:
    limit_safe = max(1, min(limit, 80))
    query = {} if include_inactive else {"is_active": {"$ne": False}}
    rows: list[dict[str, Any]] = []
    for doc in shorts_col.find(query, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
        rows.append(_serialize_content(doc, "short", bot_username))
    return _attach_engagement(rows, user_tg_id)


def _user_preference_profile(user_tg_id: int, max_events: int = 80) -> dict[str, Any]:
    genre_weights: dict[str, int] = {}
    positive_keys: set[str] = set()
    negative_keys: set[str] = set()
    doc_cache: dict[str, dict[str, Any] | None] = {}

    def _cached_doc(content_type: str, content_ref: str) -> dict[str, Any] | None:
        key = _content_key(content_type, content_ref)
        if key in doc_cache:
            return doc_cache[key]
        doc_cache[key] = _get_content_doc(content_type, content_ref)
        return doc_cache[key]

    for row in reactions_col.find(
        {"user_tg_id": user_tg_id, "reaction": {"$in": ["like", "dislike"]}},
        {"content_type": 1, "content_ref": 1, "reaction": 1},
    ).sort("updated_at", DESCENDING).limit(max_events):
        content_type = str(row.get("content_type") or "")
        content_ref = str(row.get("content_ref") or "")
        reaction = str(row.get("reaction") or "")
        if content_type not in {"movie", "serial"} or not content_ref:
            continue
        key = _content_key(content_type, content_ref)
        if reaction == "like":
            positive_keys.add(key)
            weight = 5
        else:
            negative_keys.add(key)
            weight = -7
        doc = _cached_doc(content_type, content_ref)
        if not doc:
            continue
        for value in doc.get("genres", []):
            genre = _normalize_lookup(str(value))
            if not genre:
                continue
            genre_weights[genre] = genre_weights.get(genre, 0) + weight

    for row in watch_progress_col.find({"user_tg_id": user_tg_id}).sort("updated_at", DESCENDING).limit(max_events):
        content_type = str(row.get("content_type") or "")
        content_ref = str(row.get("content_ref") or "")
        if content_type not in {"movie", "serial"} or not content_ref:
            continue
        position_seconds = int(row.get("position_seconds") or 0)
        duration_seconds = int(row.get("duration_seconds") or 0)
        if position_seconds < 60:
            continue
        watch_weight = 2 if duration_seconds > 0 and position_seconds >= int(duration_seconds * 0.5) else 1
        doc = _cached_doc(content_type, content_ref)
        if not doc:
            continue
        for value in doc.get("genres", []):
            genre = _normalize_lookup(str(value))
            if not genre:
                continue
            genre_weights[genre] = genre_weights.get(genre, 0) + watch_weight

    for row in history_col.find({"user_tg_id": user_tg_id}).sort("viewed_at", DESCENDING).limit(max_events):
        content_type = str(row.get("content_type") or "")
        content_ref = str(row.get("content_ref") or "")
        if content_type not in {"movie", "serial"} or not content_ref:
            continue
        doc = _cached_doc(content_type, content_ref)
        if not doc:
            continue
        for value in doc.get("genres", []):
            genre = _normalize_lookup(str(value))
            if not genre:
                continue
            genre_weights[genre] = genre_weights.get(genre, 0) + 1

    return {
        "genre_weights": genre_weights,
        "positive_keys": positive_keys,
        "negative_keys": negative_keys,
    }


def _list_recommendations(
    content_type: str,
    content_ref: str,
    user_tg_id: int,
    bot_username: str | None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    if content_type not in {"movie", "serial"}:
        return []
    base_doc = _get_content_doc(content_type, content_ref)
    if not base_doc:
        return []

    limit_safe = max(1, min(limit, 30))
    base_genres = {_normalize_lookup(str(item)) for item in base_doc.get("genres", []) if str(item).strip()}
    base_title_norm = _normalize_lookup(str(base_doc.get("title") or ""))
    base_tokens = {token for token in base_title_norm.split() if token}
    base_year = base_doc.get("year") if isinstance(base_doc.get("year"), int) else None
    profile = _user_preference_profile(user_tg_id)
    genre_weights = profile["genre_weights"]
    positive_keys = profile["positive_keys"]
    negative_keys = profile["negative_keys"]

    collection = movies_col if content_type == "movie" else serials_col
    candidates: list[dict[str, Any]] = []
    candidate_genres: dict[str, set[str]] = {}
    candidate_years: dict[str, int | None] = {}
    candidate_title_tokens: dict[str, set[str]] = {}
    for doc in collection.find({"is_active": {"$ne": False}}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(280):
        doc_id = str(doc.get("_id") or "")
        if not doc_id or doc_id == content_ref:
            continue
        item = _serialize_content(doc, content_type, bot_username)
        key = _content_key(content_type, doc_id)
        candidates.append(item)
        candidate_genres[key] = {_normalize_lookup(str(value)) for value in doc.get("genres", []) if str(value).strip()}
        candidate_years[key] = doc.get("year") if isinstance(doc.get("year"), int) else None
        candidate_title_tokens[key] = {token for token in _normalize_lookup(str(doc.get("title") or "")).split() if token}

    if not candidates:
        return []

    engagement = _engagement_maps(candidates, user_tg_id)
    scored: list[dict[str, Any]] = []
    for item in candidates:
        key = _content_key(content_type, str(item.get("id") or ""))
        item_genres = candidate_genres.get(key, set())
        overlap = len(base_genres & item_genres)
        token_overlap = len(base_tokens & candidate_title_tokens.get(key, set()))

        score = overlap * 8
        score += min(token_overlap, 5) * 2
        score += sum(int(genre_weights.get(genre, 0)) for genre in item_genres)

        doc_year = candidate_years.get(key)
        if base_year is not None and doc_year is not None:
            diff = abs(base_year - doc_year)
            if diff <= 1:
                score += 3
            elif diff <= 3:
                score += 1

        likes = int((engagement.get("likes") or {}).get(key, 0))
        dislikes = int((engagement.get("dislikes") or {}).get(key, 0))
        comments = int((engagement.get("comments") or {}).get(key, 0))
        downloads = int(item.get("downloads") or 0)
        tracked_downloads = int((engagement.get("downloads") or {}).get(key, 0))
        user_reaction = str((engagement.get("user_reaction") or {}).get(key, ""))

        score += min(likes // 5, 8)
        score += min(comments // 4, 6)
        score += min((downloads + tracked_downloads) // 40, 6)
        score -= min(dislikes // 4, 5)
        if key in positive_keys or user_reaction == "like":
            score += 8
        if key in negative_keys or user_reaction == "dislike":
            score -= 20

        row = dict(item)
        row["recommend_score"] = int(score)
        scored.append(row)

    scored.sort(key=lambda x: (int(x.get("recommend_score") or 0), str(x.get("created_at") or "")), reverse=True)
    selected: list[dict[str, Any]] = []
    for row in scored:
        if int(row.get("recommend_score") or 0) <= -10 and len(selected) >= limit_safe:
            break
        row.pop("recommend_score", None)
        selected.append(row)
        if len(selected) >= limit_safe:
            break

    return _apply_engagement_maps(selected, engagement)


def _dedupe_content_rows(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    limit_safe = max(1, limit)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = _content_key(str(item.get("content_type") or ""), str(item.get("id") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(item)
        if len(rows) >= limit_safe:
            break
    return rows


def _list_reacted_content(
    user_tg_id: int,
    reaction: str,
    bot_username: str | None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if reaction not in {"like", "dislike"}:
        return []
    limit_safe = max(1, min(limit, 80))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in reactions_col.find({"user_tg_id": user_tg_id, "reaction": reaction}).sort("updated_at", DESCENDING).limit(limit_safe * 3):
        content_type = str(row.get("content_type") or "")
        content_ref = str(row.get("content_ref") or "")
        key = _content_key(content_type, content_ref)
        if key in seen:
            continue
        seen.add(key)
        doc = _get_content_doc(content_type, content_ref)
        if not doc:
            continue
        rows.append(_serialize_content(doc, content_type, bot_username))
        if len(rows) >= limit_safe:
            break
    return _attach_engagement(rows, user_tg_id)


def _list_for_you(user_tg_id: int, bot_username: str | None, limit: int = 20) -> list[dict[str, Any]]:
    limit_safe = max(1, min(limit, 60))
    profile = _user_preference_profile(user_tg_id)
    genre_weights = profile.get("genre_weights") or {}
    positive_keys = profile.get("positive_keys") or set()
    negative_keys = profile.get("negative_keys") or set()

    candidates: list[dict[str, Any]] = []
    scan_limit = max(180, limit_safe * 8)
    for doc in movies_col.find({"is_active": {"$ne": False}}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
        candidates.append(_serialize_content(doc, "movie", bot_username))
    for doc in serials_col.find({"is_active": {"$ne": False}}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
        candidates.append(_serialize_content(doc, "serial", bot_username))
    if not candidates:
        return []

    enriched = _attach_engagement(candidates, user_tg_id)
    scored: list[dict[str, Any]] = []
    for item in enriched:
        key = _content_key(str(item.get("content_type") or ""), str(item.get("id") or ""))
        item_genres = {_normalize_lookup(str(value)) for value in item.get("genres", []) if str(value).strip()}
        score = sum(int(genre_weights.get(genre, 0)) for genre in item_genres)
        likes = int(item.get("likes") or 0)
        dislikes = int(item.get("dislikes") or 0)
        comments = int(item.get("comments") or 0)
        score += min(likes // 3, 16) + min(comments // 3, 12) - min(dislikes // 2, 15)
        if key in positive_keys:
            score += 14
        if key in negative_keys:
            score -= 24
        row = dict(item)
        row["for_you_score"] = score
        scored.append(row)
    scored.sort(key=lambda x: (int(x.get("for_you_score") or 0), str(x.get("created_at") or "")), reverse=True)
    for row in scored:
        row.pop("for_you_score", None)
    return scored[:limit_safe]


def _recommendation_feed(
    user_tg_id: int,
    bot_username: str | None,
    content_type: str = "",
    content_ref: str = "",
    limit: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    limit_safe = max(1, min(limit, 30))
    similar = (
        _list_recommendations(content_type, content_ref, user_tg_id, bot_username, limit=limit_safe)
        if content_type in {"movie", "serial"} and content_ref
        else []
    )
    for_you = _list_for_you(user_tg_id, bot_username, limit=limit_safe)
    trend = _list_trending(user_tg_id, bot_username, limit=limit_safe)
    continue_rows = _list_continue_watching(user_tg_id, bot_username, limit=limit_safe)

    # Keep sections distinct enough for UI rows.
    for_you_keys = {_content_key(str(item.get("content_type") or ""), str(item.get("id") or "")) for item in for_you}
    trend = _dedupe_content_rows(
        [
            row
            for row in trend
            if _content_key(str(row.get("content_type") or ""), str(row.get("id") or ""))
            not in for_you_keys
        ],
        limit=limit_safe,
    )
    return {
        "similar": similar[:limit_safe],
        "for_you": for_you[:limit_safe],
        "trend": trend[:limit_safe],
        "continue_watching": continue_rows[:limit_safe],
    }


def _serialize_notification(item: dict[str, Any], kind: str, title: str, created_at: str, text: str = "") -> dict[str, Any]:
    content_type = str(item.get("content_type") or "")
    content_ref = str(item.get("id") or item.get("content_ref") or "")
    return {
        "id": f"{kind}:{content_type}:{content_ref}:{created_at}",
        "type": kind,
        "title": title,
        "text": text,
        "created_at": created_at,
        "item": item,
    }


def _notification_seen_at(user_tg_id: int) -> str:
    row = users_col.find_one({"tg_id": user_tg_id}, {"notifications_seen_at": 1}) or {}
    return str(row.get("notifications_seen_at") or "")


def _mark_notifications_read(user_tg_id: int) -> str:
    now_iso = _now_iso()
    users_col.update_one({"tg_id": user_tg_id}, {"$set": {"notifications_seen_at": now_iso}})
    return now_iso


def _list_notifications(user_tg_id: int, bot_username: str | None, limit: int = 20) -> dict[str, Any]:
    limit_safe = max(1, min(limit, 40))
    scan_limit = max(40, limit_safe * 6)
    notifications: list[dict[str, Any]] = []

    for doc in movies_col.find({"is_active": {"$ne": False}}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
        item = _serialize_content(doc, "movie", bot_username)
        created_at = str(item.get("created_at") or "")
        if not created_at:
            continue
        notifications.append(
            _serialize_notification(
                item,
                kind="new_content",
                title="Yangi kino qo'shildi",
                text=str(item.get("title") or ""),
                created_at=created_at,
            )
        )

    for doc in serials_col.find({"is_active": {"$ne": False}}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
        item = _serialize_content(doc, "serial", bot_username)
        created_at = str(item.get("created_at") or "")
        if not created_at:
            continue
        notifications.append(
            _serialize_notification(
                item,
                kind="new_content",
                title="Yangi serial qo'shildi",
                text=str(item.get("title") or ""),
                created_at=created_at,
            )
        )

    for doc in shorts_col.find({"is_active": {"$ne": False}}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
        item = _serialize_content(doc, "short", bot_username)
        created_at = str(item.get("created_at") or "")
        if not created_at:
            continue
        notifications.append(
            _serialize_notification(
                item,
                kind="new_short",
                title="Yangi short qo'shildi",
                text=str(item.get("title") or ""),
                created_at=created_at,
            )
        )

    for item in _list_continue_watching(user_tg_id, bot_username, limit=4):
        progress = item.get("watch_progress") or {}
        updated_at = str(progress.get("updated_at") or "")
        if not updated_at:
            continue
        notifications.append(
            _serialize_notification(
                item,
                kind="continue_watching",
                title="Tomoshani davom ettiring",
                text=str(item.get("title") or ""),
                created_at=updated_at,
            )
        )

    notifications.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    unique_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in notifications:
        item = row.get("item") or {}
        key = f"{row.get('type')}:{item.get('content_type')}:{item.get('id')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_rows.append(row)
        if len(unique_rows) >= limit_safe:
            break

    seen_at = _notification_seen_at(user_tg_id)
    if not seen_at:
        unread_count = len(unique_rows)
    else:
        unread_count = sum(1 for row in unique_rows if str(row.get("created_at") or "") > seen_at)

    return {
        "items": unique_rows,
        "unread_count": unread_count,
        "seen_at": seen_at,
    }


def _verify_webapp_init_data(init_data: str) -> dict[str, Any]:
    if not init_data:
        raise HTTPException(status_code=401, detail="Missing Telegram init data.")

    pairs = dict(parse_qsl(init_data, keep_blank_values=True))
    hash_received = pairs.pop("hash", "")
    if not hash_received:
        raise HTTPException(status_code=401, detail="Invalid Telegram init data.")

    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    secret_key = hmac.new(b"WebAppData", BOT_TOKEN.encode("utf-8"), hashlib.sha256).digest()
    calculated_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(calculated_hash, hash_received):
        raise HTTPException(status_code=401, detail="Telegram init data signature mismatch.")

    try:
        auth_date = int(pairs.get("auth_date", "0"))
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid auth_date.")
    if auth_date <= 0:
        raise HTTPException(status_code=401, detail="Missing auth_date.")
    if WEBAPP_AUTH_MAX_AGE_SECONDS > 0 and (int(time.time()) - auth_date) > WEBAPP_AUTH_MAX_AGE_SECONDS:
        raise HTTPException(status_code=401, detail="Telegram session expired. Reopen from bot.")

    user_raw = pairs.get("user")
    if not user_raw:
        raise HTTPException(status_code=401, detail="Missing Telegram user payload.")
    try:
        user = json.loads(user_raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=401, detail="Invalid user payload.")
    if not isinstance(user, dict) or "id" not in user:
        raise HTTPException(status_code=401, detail="Invalid Telegram user.")
    return user


def _dev_user_payload() -> dict[str, Any]:
    return {
        "id": WEBAPP_DEV_USER_ID,
        "first_name": "Dev",
        "last_name": "User",
        "username": "dev_user",
    }


def _issue_media_token(user_tg_id: int) -> str:
    user_id = int(user_tg_id)
    ttl = max(300, int(WEBAPP_MEDIA_TOKEN_TTL_SECONDS))
    expires_at = int(time.time()) + ttl
    payload = f"{user_id}:{expires_at}"
    signature = hmac.new(
        BOT_TOKEN.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{user_id}.{expires_at}.{signature}"


def _verify_media_token(token: str) -> int:
    raw = str(token or "").strip()
    if not raw:
        return 0
    parts = raw.split(".", 2)
    if len(parts) != 3:
        return 0
    user_raw, expires_raw, signature = parts
    try:
        user_id = int(user_raw)
        expires_at = int(expires_raw)
    except ValueError:
        return 0
    if user_id <= 0 or expires_at <= int(time.time()):
        return 0
    payload = f"{user_id}:{expires_at}"
    expected = hmac.new(
        BOT_TOKEN.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return 0
    return user_id


def _resolve_user(init_data: str) -> dict[str, Any]:
    if init_data:
        return _verify_webapp_init_data(init_data)
    if WEBAPP_ENABLE_DEV_AUTH and WEBAPP_DEV_USER_ID > 0:
        return _dev_user_payload()
    raise HTTPException(status_code=401, detail="Telegram auth required.")


def _resolve_media_user(token: str, init_data: str) -> dict[str, Any]:
    token_user_id = _verify_media_token(token)
    if token_user_id > 0:
        return {"id": token_user_id}
    return _resolve_user(init_data)


def _upsert_user(user: dict[str, Any]) -> None:
    tg_id = int(user["id"])
    full_name = f"{str(user.get('first_name') or '').strip()} {str(user.get('last_name') or '').strip()}".strip()
    users_col.update_one(
        {"tg_id": tg_id},
        {
            "$set": {"full_name": full_name or str(user.get("username") or "")},
            "$setOnInsert": {"joined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        },
        upsert=True,
    )


async def _telegram_api(method: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{TELEGRAM_API_BASE}/{method}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as session:
        response = await session.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Telegram API error: {response.text}")
    payload = response.json()
    if not payload.get("ok"):
        return {}
    return payload.get("result", {})


async def _close_http_stream(response: httpx.Response, session: httpx.AsyncClient) -> None:
    try:
        await response.aclose()
    finally:
        await session.aclose()


@lru_cache(maxsize=1)
def _bot_username_env_cached() -> str:
    return BOT_USERNAME


async def _get_bot_username() -> str | None:
    env_name = _bot_username_env_cached()
    if env_name:
        return env_name.lstrip("@")
    global _BOT_USERNAME_CACHE
    if _BOT_USERNAME_CACHE:
        return _BOT_USERNAME_CACHE
    result = await _telegram_api("getMe", {})
    username = str(result.get("username") or "").strip()
    _BOT_USERNAME_CACHE = username or None
    return _BOT_USERNAME_CACHE


async def _get_missing_channels(user_tg_id: int) -> list[dict[str, Any]]:
    now_ts = _now_ts()
    cached = _SUBSCRIPTION_CACHE.get(int(user_tg_id))
    if cached and cached[1] > now_ts:
        return [dict(row) for row in cached[0]]

    channels = list(required_channels_col.find({"is_active": True}).sort("created_at", DESCENDING))
    if not channels:
        _SUBSCRIPTION_CACHE[int(user_tg_id)] = ([], now_ts + max(30, WEBAPP_SUBSCRIPTION_CACHE_SECONDS))
        return []
    pending_refs = {
        str(row.get("channel_ref") or "").strip()
        for row in join_requests_col.find({"user_tg_id": user_tg_id}, {"channel_ref": 1})
    }
    missing: list[dict[str, Any]] = []
    for row in channels:
        channel_ref = str(row.get("channel_ref") or "").strip()
        if not channel_ref:
            continue
        if channel_ref in pending_refs:
            continue
        joined = False
        try:
            member = await _telegram_api("getChatMember", {"chat_id": channel_ref, "user_id": user_tg_id})
            status = str(member.get("status") or "").strip().lower()
            joined = _is_subscribed_status(status)
        except HTTPException:
            if cached and cached[1] > now_ts:
                return [dict(row) for row in cached[0]]
            joined = False
        if joined:
            continue
        title = str(row.get("title") or channel_ref)
        join_link = str(row.get("join_link") or "").strip() or None
        missing.append(
            {
                "channel_ref": channel_ref,
                "title": title,
                "join_url": _make_join_url(channel_ref, join_link),
            }
        )
    _SUBSCRIPTION_CACHE[int(user_tg_id)] = (
        [dict(row) for row in missing],
        now_ts + max(30, WEBAPP_SUBSCRIPTION_CACHE_SECONDS),
    )
    return missing


async def _current_user(
    x_telegram_init_data: str | None = Header(default=None),
) -> dict[str, Any]:
    user = _resolve_user((x_telegram_init_data or "").strip())
    _upsert_user(user)
    return user


async def _require_subscribed(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    user_id = int(user["id"])
    missing = await _get_missing_channels(user_id)
    if missing:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "subscription_required",
                "missing_channels": missing,
            },
        )
    return user


def _is_admin_user(user_tg_id: int) -> bool:
    if int(user_tg_id) in ADMIN_IDS:
        return True
    return admins_col.find_one({"tg_id": user_tg_id}, {"_id": 1}) is not None


async def _require_admin(user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    user_id = int(user["id"])
    if not _is_admin_user(user_id):
        raise HTTPException(status_code=403, detail="Admin required.")
    return user


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/bootstrap")
async def bootstrap(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    user_id = int(user["id"])
    is_admin = _is_admin_user(user_id)
    full_name = f"{str(user.get('first_name') or '').strip()} {str(user.get('last_name') or '').strip()}".strip()
    missing_channels = await _get_missing_channels(user_id)
    bot_username = await _get_bot_username()
    blocked = len(missing_channels) > 0
    content = [] if blocked else _list_content(
        query="",
        content_type="all",
        limit=120,
        bot_username=bot_username,
        user_tg_id=user_id,
    )
    favorites = [] if blocked else _list_favorites(user_id, bot_username)
    history = [] if blocked else _list_recent_history(user_id, bot_username, limit=12)
    trending = [] if blocked else _list_trending(user_id, bot_username, limit=18)
    shorts = [] if blocked else _list_shorts(user_id, bot_username, limit=18)
    notifications = {"items": [], "unread_count": 0, "seen_at": ""} if blocked else _list_notifications(user_id, bot_username, limit=12)
    recommendations_feed = {"similar": [], "for_you": [], "trend": [], "continue_watching": []}
    if not blocked:
        recommendations_feed = _recommendation_feed(user_id, bot_username, limit=12)
    return {
        "user": {
            "id": user_id,
            "first_name": str(user.get("first_name") or ""),
            "last_name": str(user.get("last_name") or ""),
            "username": str(user.get("username") or ""),
            "full_name": full_name or str(user.get("username") or "") or "User",
            "is_admin": is_admin,
        },
        "media_token": _issue_media_token(user_id),
        "blocked": blocked,
        "missing_channels": missing_channels,
        "content": content,
        "favorites": favorites,
        "history": history,
        "trending": trending,
        "shorts": shorts,
        "continue_watching": [] if blocked else _list_continue_watching(user_id, bot_username, limit=12),
        "recommendations_feed": recommendations_feed,
        "notifications": notifications,
        "stats": {
            "movies": movies_col.count_documents({}),
            "serials": serials_col.count_documents({}),
            "shorts": shorts_col.count_documents({}),
            "favorites": favorites_col.count_documents({"user_tg_id": user_id}),
            "comments": comments_col.count_documents({"user_tg_id": user_id}),
            "downloads": downloads_col.count_documents({"user_tg_id": user_id}),
            "likes_given": reactions_col.count_documents({"user_tg_id": user_id, "reaction": "like"}),
            "dislikes_given": reactions_col.count_documents({"user_tg_id": user_id, "reaction": "dislike"}),
            "history_views": history_col.count_documents({"user_tg_id": user_id}),
            "watched": history_col.count_documents({"user_tg_id": user_id}),
            "continue_items": watch_progress_col.count_documents({"user_tg_id": user_id}),
            "notifications_unread": int(notifications.get("unread_count") or 0),
        },
    }


@app.get("/api/content")
async def content(
    q: str = Query(default=""),
    content_type: str = Query(default="all"),
    limit: int = Query(default=80, ge=1, le=200),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    user_id = int(user["id"])
    content_type_safe = content_type if content_type in {"all", "movie", "serial", "short"} else "all"
    bot_username = await _get_bot_username()
    items = _list_content(
        query=q,
        content_type=content_type_safe,
        limit=limit,
        bot_username=bot_username,
        user_tg_id=user_id,
    )
    return {"items": items}


@app.get("/api/content/{content_type}/{content_ref}")
async def content_detail(
    content_type: str,
    content_ref: str,
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    content_type_safe = content_type if content_type in VALID_CONTENT_TYPES else ""
    if not content_type_safe:
        raise HTTPException(status_code=400, detail="Invalid content_type.")
    doc = _get_content_doc(content_type_safe, content_ref)
    if not doc:
        raise HTTPException(status_code=404, detail="Content not found.")

    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    item = _serialize_content(doc, content_type_safe, bot_username)
    item.update(_engagement_summary(content_type_safe, content_ref, user_id))
    _track_history_view(user_id, content_type_safe, content_ref)
    recommendations = _list_recommendations(content_type_safe, content_ref, user_id, bot_username, limit=10)
    feed = _recommendation_feed(
        user_tg_id=user_id,
        bot_username=bot_username,
        content_type=content_type_safe,
        content_ref=content_ref,
        limit=10,
    )
    return {"item": item, "recommendations": recommendations, "feed": feed}


@app.get("/api/recommendations")
async def recommendations(
    content_type: str = Query(...),
    content_ref: str = Query(...),
    limit: int = Query(default=12, ge=1, le=30),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    content_type_safe = content_type if content_type in {"movie", "serial"} else ""
    if not content_type_safe:
        raise HTTPException(status_code=400, detail="Invalid content_type.")
    if not _get_content_doc(content_type_safe, content_ref):
        raise HTTPException(status_code=404, detail="Content not found.")

    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    items = _list_recommendations(content_type_safe, content_ref, user_id, bot_username, limit=limit)
    feed = _recommendation_feed(
        user_tg_id=user_id,
        bot_username=bot_username,
        content_type=content_type_safe,
        content_ref=content_ref,
        limit=limit,
    )
    return {"items": items, "feed": feed}


@app.get("/api/recommendations/feed")
async def recommendations_feed(
    limit: int = Query(default=12, ge=1, le=30),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    return _recommendation_feed(user_tg_id=user_id, bot_username=bot_username, limit=limit)


@app.get("/api/watch")
async def watch_info(
    content_type: str = Query(...),
    content_ref: str = Query(...),
    episode: int | None = Query(default=None),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    content_type_safe = content_type if content_type in VALID_CONTENT_TYPES else ""
    if not content_type_safe:
        raise HTTPException(status_code=400, detail="Invalid content_type.")

    doc = _get_content_doc(content_type_safe, content_ref)
    if not doc:
        raise HTTPException(status_code=404, detail="Content not found.")

    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    item = _serialize_content(doc, content_type_safe, bot_username)
    item.update(_engagement_summary(content_type_safe, content_ref, user_id))
    _track_history_view(user_id, content_type_safe, content_ref)

    if content_type_safe in {"movie", "short"}:
        file_id = str(doc.get("file_id") or "")
        media_type = str(doc.get("media_type") or "")
        if not file_id:
            raise HTTPException(status_code=404, detail="Video file not found.")
        progress = _get_watch_progress(user_id, content_type_safe, content_ref, None)
        return {
            "item": item,
            "current_episode": None,
            "episodes": [],
            "playback": progress,
            "stream_file_id": file_id,
            "media_type": media_type,
            "is_video": _is_video_media_type(media_type),
        }

    episodes = _list_serial_episodes(content_ref)
    if not episodes:
        raise HTTPException(status_code=404, detail="Serial episodes not found.")
    chosen = None
    if episode is not None:
        for row in episodes:
            if int(row.get("episode_number") or 0) == int(episode):
                chosen = row
                break
    if not chosen:
        chosen = episodes[0]

    progress = _get_watch_progress(user_id, content_type_safe, content_ref, int(chosen["episode_number"]))
    return {
        "item": item,
        "current_episode": int(chosen["episode_number"]),
        "episodes": episodes,
        "playback": progress,
        "stream_file_id": str(chosen["file_id"]),
        "media_type": str(chosen.get("media_type") or ""),
        "is_video": bool(chosen.get("is_video")),
    }


@app.get("/api/favorites")
async def favorites(user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    return {"items": _list_favorites(user_id, bot_username)}


@app.post("/api/favorites/toggle")
async def favorite_toggle(payload: FavoriteToggleIn, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    if content_type not in VALID_CONTENT_TYPES or not content_ref:
        raise HTTPException(status_code=400, detail="Invalid payload.")

    user_id = int(user["id"])
    criteria = {"user_tg_id": user_id, "content_type": content_type, "content_ref": content_ref}
    existing = favorites_col.find_one(criteria, {"_id": 1})
    if existing:
        favorites_col.delete_one(criteria)
        is_favorite = False
    else:
        favorites_col.update_one(
            criteria,
            {
                "$setOnInsert": {
                    "user_tg_id": user_id,
                    "content_type": content_type,
                    "content_ref": content_ref,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            },
            upsert=True,
        )
        is_favorite = True
    return {"ok": True, "is_favorite": is_favorite}


@app.post("/api/reactions/set")
async def set_reaction(payload: ReactionIn, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    reaction = payload.reaction.strip().lower()
    if content_type not in VALID_CONTENT_TYPES or not content_ref:
        raise HTTPException(status_code=400, detail="Invalid payload.")
    if reaction not in {"like", "dislike", "none"}:
        raise HTTPException(status_code=400, detail="Invalid reaction.")
    if not _get_content_doc(content_type, content_ref):
        raise HTTPException(status_code=404, detail="Content not found.")

    user_id = int(user["id"])
    criteria = {"user_tg_id": user_id, "content_type": content_type, "content_ref": content_ref}
    if reaction == "none":
        reactions_col.delete_one(criteria)
        current = ""
    else:
        reactions_col.update_one(
            criteria,
            {
                "$set": {
                    "reaction": reaction,
                    "updated_at": _now_iso(),
                },
                "$setOnInsert": {
                    "created_at": _now_iso(),
                    "user_tg_id": user_id,
                    "content_type": content_type,
                    "content_ref": content_ref,
                },
            },
            upsert=True,
        )
        current = reaction

    summary = _engagement_summary(content_type, content_ref, user_id)
    summary["user_reaction"] = current
    return {"ok": True, "summary": summary}


@app.get("/api/comments")
async def list_comments(
    content_type: str = Query(...),
    content_ref: str = Query(...),
    limit: int = Query(default=40, ge=1, le=120),
    sort: str = Query(default="new"),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    if content_type not in VALID_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid content_type.")
    sort_safe = sort if sort in {"new", "old", "top"} else "new"
    user_id = int(user["id"])
    items = _list_comment_threads(
        content_type=content_type,
        content_ref=content_ref,
        user_tg_id=user_id,
        limit=limit,
        sort=sort_safe,
        is_admin=_is_admin_user(user_id),
    )
    return {"items": items}


@app.delete("/api/comments/{comment_id}")
async def delete_comment(comment_id: str, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    oid = _safe_object_id(comment_id)
    if not oid:
        raise HTTPException(status_code=400, detail="Invalid comment id.")

    user_id = int(user["id"])
    row = comments_col.find_one({"_id": oid})
    if not row:
        raise HTTPException(status_code=404, detail="Comment not found.")

    is_owner = int(row.get("user_tg_id") or 0) == user_id
    if not is_owner and not _is_admin_user(user_id):
        raise HTTPException(status_code=403, detail="Not allowed.")

    comment_id = str(row.get("_id") or "")
    reply_ids = [str(item.get("_id")) for item in comments_col.find({"parent_comment_id": comment_id}, {"_id": 1})]
    comments_col.delete_many({"$or": [{"_id": oid}, {"parent_comment_id": comment_id}]})
    comment_reactions_col.delete_many({"comment_id": {"$in": [comment_id, *reply_ids]}})
    content_type = str(row.get("content_type") or "")
    content_ref = str(row.get("content_ref") or "")
    summary = _engagement_summary(content_type, content_ref, user_id) if content_type and content_ref else {}
    return {"ok": True, "summary": summary}


@app.post("/api/comments/add")
async def add_comment(payload: CommentIn, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    text = re.sub(r"\s+", " ", payload.text.strip())
    parent_comment_id = str(payload.parent_comment_id or "").strip()
    if content_type not in VALID_CONTENT_TYPES or not content_ref:
        raise HTTPException(status_code=400, detail="Invalid payload.")
    if len(text) < 2:
        raise HTTPException(status_code=400, detail="Comment too short.")
    if len(text) > 600:
        raise HTTPException(status_code=400, detail="Comment too long.")
    if not _get_content_doc(content_type, content_ref):
        raise HTTPException(status_code=404, detail="Content not found.")

    parent_doc: dict[str, Any] | None = None
    if parent_comment_id:
        parent_oid = _safe_object_id(parent_comment_id)
        if not parent_oid:
            raise HTTPException(status_code=400, detail="Invalid parent comment id.")
        parent_doc = comments_col.find_one({"_id": parent_oid, "content_type": content_type, "content_ref": content_ref})
        if not parent_doc:
            raise HTTPException(status_code=404, detail="Parent comment not found.")
        # Keep one-level threads by anchoring replies to the top-level comment id.
        parent_comment_id = str(parent_doc.get("parent_comment_id") or parent_doc.get("_id") or "")

    user_id = int(user["id"])
    full_name = f"{str(user.get('first_name') or '').strip()} {str(user.get('last_name') or '').strip()}".strip()
    doc = {
        "content_type": content_type,
        "content_ref": content_ref,
        "user_tg_id": user_id,
        "full_name": full_name or "User",
        "username": str(user.get("username") or ""),
        "text": text,
        "created_at": _now_iso(),
    }
    if parent_comment_id:
        doc["parent_comment_id"] = parent_comment_id
    result = comments_col.insert_one(doc)
    doc["_id"] = result.inserted_id
    summary = _engagement_summary(content_type, content_ref, user_id)
    item = _serialize_comment(doc)
    item.update(
        {
            "likes": 0,
            "dislikes": 0,
            "user_reaction": "",
            "reply_count": 0,
            "is_owner": True,
            "can_delete": True,
            "replies": [],
        }
    )
    return {"ok": True, "item": item, "summary": summary}


@app.post("/api/comments/reactions/set")
async def set_comment_reaction(
    payload: CommentReactionIn,
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    comment_id = str(payload.comment_id or "").strip()
    reaction = str(payload.reaction or "").strip().lower()
    if reaction not in {"like", "dislike", "none"}:
        raise HTTPException(status_code=400, detail="Invalid reaction.")

    oid = _safe_object_id(comment_id)
    if not oid:
        raise HTTPException(status_code=400, detail="Invalid comment id.")
    row = comments_col.find_one({"_id": oid}, {"_id": 1})
    if not row:
        raise HTTPException(status_code=404, detail="Comment not found.")

    user_id = int(user["id"])
    criteria = {"user_tg_id": user_id, "comment_id": comment_id}
    if reaction == "none":
        comment_reactions_col.delete_one(criteria)
    else:
        comment_reactions_col.update_one(
            criteria,
            {
                "$set": {
                    "reaction": reaction,
                    "updated_at": _now_iso(),
                    "user_tg_id": user_id,
                    "comment_id": comment_id,
                },
                "$setOnInsert": {"created_at": _now_iso()},
            },
            upsert=True,
        )
    summary = _comment_reaction_summary(comment_id, user_id)
    return {"ok": True, "summary": summary}


@app.post("/api/downloads/track")
async def track_download(payload: DownloadTrackIn, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    if content_type not in VALID_CONTENT_TYPES or not content_ref:
        raise HTTPException(status_code=400, detail="Invalid payload.")
    if not _get_content_doc(content_type, content_ref):
        raise HTTPException(status_code=404, detail="Content not found.")

    user_id = int(user["id"])
    downloads_col.insert_one(
        {
            "user_tg_id": user_id,
            "content_type": content_type,
            "content_ref": content_ref,
            "created_at": _now_iso(),
        }
    )
    summary = _engagement_summary(content_type, content_ref, user_id)
    return {"ok": True, "summary": summary}


@app.post("/api/watch/progress")
async def track_watch_progress(payload: WatchProgressIn, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    if content_type not in VALID_CONTENT_TYPES or not content_ref:
        raise HTTPException(status_code=400, detail="Invalid payload.")
    if not _get_content_doc(content_type, content_ref):
        raise HTTPException(status_code=404, detail="Content not found.")

    episode_number = payload.episode_number if payload.episode_number is None else int(payload.episode_number)
    position = max(0, int(payload.position_seconds))
    duration = max(0, int(payload.duration_seconds or 0))
    user_id = int(user["id"])

    criteria = {"user_tg_id": user_id}
    criteria.update(_progress_key(content_type, content_ref, episode_number))
    watch_progress_col.update_one(
        criteria,
        {
            "$set": {
                "position_seconds": position,
                "duration_seconds": duration,
                "updated_at": _now_iso(),
                "user_tg_id": user_id,
                "content_type": content_type,
                "content_ref": content_ref,
                "episode_number": episode_number,
            },
            "$setOnInsert": {"created_at": _now_iso()},
        },
        upsert=True,
    )
    return {"ok": True}


@app.get("/api/history")
async def history(
    limit: int = Query(default=20, ge=1, le=100),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    return {"items": _list_recent_history(user_id, bot_username, limit=limit)}


@app.get("/api/continue")
async def continue_watching(
    limit: int = Query(default=20, ge=1, le=100),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    return {"items": _list_continue_watching(user_id, bot_username, limit=limit)}


@app.get("/api/notifications")
async def notifications(
    limit: int = Query(default=20, ge=1, le=40),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    return _list_notifications(user_id, bot_username, limit=limit)


@app.post("/api/notifications/read")
async def notifications_read(user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    user_id = int(user["id"])
    seen_at = _mark_notifications_read(user_id)
    return {"ok": True, "seen_at": seen_at}


@app.get("/api/profile")
async def profile(user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    full_name = f"{str(user.get('first_name') or '').strip()} {str(user.get('last_name') or '').strip()}".strip()
    notifications = _list_notifications(user_id, bot_username, limit=12)
    saved = _list_favorites(user_id, bot_username)
    continue_rows = _list_continue_watching(user_id, bot_username, limit=18)
    likes = _list_reacted_content(user_id, "like", bot_username, limit=18)
    return {
        "user": {
            "id": user_id,
            "first_name": str(user.get("first_name") or ""),
            "last_name": str(user.get("last_name") or ""),
            "username": str(user.get("username") or ""),
            "full_name": full_name or str(user.get("username") or "") or "User",
        },
        "stats": {
            "movies": movies_col.count_documents({}),
            "serials": serials_col.count_documents({}),
            "shorts": shorts_col.count_documents({}),
            "favorites": favorites_col.count_documents({"user_tg_id": user_id}),
            "comments": comments_col.count_documents({"user_tg_id": user_id}),
            "downloads": downloads_col.count_documents({"user_tg_id": user_id}),
            "likes_given": reactions_col.count_documents({"user_tg_id": user_id, "reaction": "like"}),
            "dislikes_given": reactions_col.count_documents({"user_tg_id": user_id, "reaction": "dislike"}),
            "history_views": history_col.count_documents({"user_tg_id": user_id}),
            "watched": history_col.count_documents({"user_tg_id": user_id}),
            "continue_items": watch_progress_col.count_documents({"user_tg_id": user_id}),
            "notifications_unread": int(notifications.get("unread_count") or 0),
        },
        "history": _list_recent_history(user_id, bot_username, limit=18),
        "saved": saved,
        "likes": likes,
        "continue_watching": continue_rows,
        "notifications": notifications,
    }


@app.get("/api/admin/overview")
async def admin_overview(user: dict[str, Any] = Depends(_require_admin)) -> dict[str, Any]:
    _ = user
    return {
        "stats": {
            "users": users_col.count_documents({}),
            "movies": movies_col.count_documents({}),
            "serials": serials_col.count_documents({}),
            "shorts": shorts_col.count_documents({}),
            "movies_active": movies_col.count_documents({"is_active": {"$ne": False}}),
            "serials_active": serials_col.count_documents({"is_active": {"$ne": False}}),
            "shorts_active": shorts_col.count_documents({"is_active": {"$ne": False}}),
            "episodes": serial_episodes_col.count_documents({}),
            "comments": comments_col.count_documents({}),
            "downloads": downloads_col.count_documents({}),
            "favorites": favorites_col.count_documents({}),
            "reactions": reactions_col.count_documents({}),
        }
    }


@app.get("/api/admin/content")
async def admin_content(
    content_type: str = Query(default="all"),
    q: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=400),
    user: dict[str, Any] = Depends(_require_admin),
) -> dict[str, Any]:
    _ = user
    content_type_safe = content_type if content_type in {"all", "movie", "serial", "short"} else "all"
    limit_safe = max(1, min(limit, 400))
    mongo_filter = _query_filter(q)
    bot_username = await _get_bot_username()
    rows: list[dict[str, Any]] = []

    if content_type_safe in {"all", "movie"}:
        for doc in movies_col.find(mongo_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
            rows.append(_serialize_content(doc, "movie", bot_username))
    if content_type_safe in {"all", "serial"}:
        for doc in serials_col.find(mongo_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
            rows.append(_serialize_content(doc, "serial", bot_username))
    if content_type_safe in {"all", "short"}:
        for doc in shorts_col.find(mongo_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
            rows.append(_serialize_content(doc, "short", bot_username))

    serial_ids = [_safe_object_id(str(item.get("id") or "")) for item in rows if item.get("content_type") == "serial"]
    serial_ids_safe = [value for value in serial_ids if value is not None]
    episodes_map: dict[str, int] = {}
    if serial_ids_safe:
        pipeline = [
            {"$match": {"serial_id": {"$in": serial_ids_safe}}},
            {"$group": {"_id": "$serial_id", "count": {"$sum": 1}}},
        ]
        for row in serial_episodes_col.aggregate(pipeline):
            row_id = row.get("_id")
            if row_id is not None:
                episodes_map[str(row_id)] = int(row.get("count") or 0)
    for item in rows:
        if item.get("content_type") == "serial":
            item["episodes_count"] = int(episodes_map.get(str(item.get("id") or ""), 0))

    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"items": rows[:limit_safe]}


@app.post("/api/admin/content/toggle")
async def admin_toggle_content(payload: AdminToggleIn, user: dict[str, Any] = Depends(_require_admin)) -> dict[str, Any]:
    _ = user
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    if content_type not in VALID_CONTENT_TYPES or not content_ref:
        raise HTTPException(status_code=400, detail="Invalid payload.")

    oid = _safe_object_id(content_ref)
    if not oid:
        raise HTTPException(status_code=400, detail="Invalid content_ref.")

    criteria = {"_id": oid}
    update = {"$set": {"is_active": bool(payload.is_active)}}
    if content_type == "movie":
        result = movies_col.update_one(criteria, update)
    elif content_type == "serial":
        result = serials_col.update_one(criteria, update)
    else:
        result = shorts_col.update_one(criteria, update)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Content not found.")
    return {"ok": True}


@app.post("/api/admin/content/create")
async def admin_create_content(
    payload: AdminCreateIn,
    user: dict[str, Any] = Depends(_require_admin),
) -> dict[str, Any]:
    _ = user
    content_type = str(payload.content_type or "").strip().lower()
    if content_type not in VALID_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid content_type.")

    title = re.sub(r"\s+", " ", str(payload.title or "").strip())
    if len(title) < 2:
        raise HTTPException(status_code=400, detail="Title too short.")
    description = re.sub(r"\s+", " ", str(payload.description or "").strip())
    if len(description) > 3000:
        raise HTTPException(status_code=400, detail="Description too long.")

    year = payload.year if isinstance(payload.year, int) else None
    if year is not None and (year < 1900 or year > 2100):
        raise HTTPException(status_code=400, detail="Invalid year.")

    genres = []
    seen_genres: set[str] = set()
    for value in payload.genres or []:
        genre = re.sub(r"\s+", " ", str(value or "").strip())
        if not genre:
            continue
        key = genre.lower()
        if key in seen_genres:
            continue
        seen_genres.add(key)
        genres.append(genre)
    genres = genres[:12]

    code = str(payload.code or "").strip()
    if not code:
        code = f"{content_type[:1].upper()}{int(time.time())}"

    media_type = str(payload.media_type or "video").strip().lower() or "video"
    preview_media_type = str(payload.preview_media_type or "photo").strip().lower() or "photo"
    file_id = str(payload.file_id or "").strip()
    preview_file_id = str(payload.preview_file_id or "").strip()
    preview_photo_file_id = str(payload.preview_photo_file_id or "").strip() or preview_file_id
    trailer_url = str(payload.trailer_url or "").strip()
    quality = re.sub(r"\s+", " ", str(payload.quality or "").strip())

    base_doc: dict[str, Any] = {
        "code": code,
        "title": title,
        "description": description,
        "year": year,
        "quality": quality,
        "genres": genres,
        "downloads": 0,
        "media_type": media_type,
        "file_id": file_id,
        "preview_media_type": preview_media_type,
        "preview_file_id": preview_file_id,
        "preview_photo_file_id": preview_photo_file_id,
        "trailer_url": trailer_url,
        "is_active": bool(payload.is_active),
        "created_at": _now_iso(),
    }

    if content_type == "short":
        if not file_id:
            raise HTTPException(status_code=400, detail="Short file_id is required.")
        if not quality:
            base_doc["quality"] = "Short"
        if not genres:
            base_doc["genres"] = ["Shorts"]
        inserted = shorts_col.insert_one(base_doc)
        doc = shorts_col.find_one({"_id": inserted.inserted_id}) or {}
        item = _serialize_content(doc, "short", await _get_bot_username())
        return {"ok": True, "item": item, "episodes_created": 0}

    if content_type == "movie":
        inserted = movies_col.insert_one(base_doc)
        doc = movies_col.find_one({"_id": inserted.inserted_id}) or {}
        item = _serialize_content(doc, "movie", await _get_bot_username())
        return {"ok": True, "item": item, "episodes_created": 0}

    inserted = serials_col.insert_one(base_doc)
    serial_id = inserted.inserted_id
    episodes_created = 0
    episodes_safe = sorted(
        [
            episode
            for episode in (payload.episodes or [])
            if isinstance(episode.episode_number, int) and episode.episode_number > 0 and str(episode.file_id or "").strip()
        ],
        key=lambda x: int(x.episode_number),
    )
    seen_episode_numbers: set[int] = set()
    for episode in episodes_safe:
        episode_number = int(episode.episode_number)
        if episode_number in seen_episode_numbers:
            continue
        seen_episode_numbers.add(episode_number)
        serial_episodes_col.insert_one(
            {
                "serial_id": serial_id,
                "episode_number": episode_number,
                "media_type": str(episode.media_type or "video").strip().lower() or "video",
                "file_id": str(episode.file_id or "").strip(),
                "created_at": _now_iso(),
            }
        )
        episodes_created += 1

    if not file_id and episodes_safe:
        first_episode = episodes_safe[0]
        serials_col.update_one(
            {"_id": serial_id},
            {
                "$set": {
                    "file_id": str(first_episode.file_id or "").strip(),
                    "media_type": str(first_episode.media_type or "video").strip().lower() or "video",
                }
            },
        )
    _SERIAL_PREVIEW_CACHE.pop(str(serial_id), None)

    doc = serials_col.find_one({"_id": serial_id}) or {}
    item = _serialize_content(doc, "serial", await _get_bot_username())
    return {"ok": True, "item": item, "episodes_created": episodes_created}


@app.get("/api/media/file")
async def media_file(
    file_id: str = Query(...),
    init_data: str = Query(default=""),
    token: str = Query(default=""),
) -> Response:
    user = _resolve_media_user(token=token.strip(), init_data=init_data.strip())
    _ = user

    file_path = await _resolve_file_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found.")

    url = f"{TELEGRAM_FILE_BASE}/{file_path}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as session:
        resp = await session.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Cannot download file from Telegram.")
    media_type = resp.headers.get("content-type") or ""
    if not media_type or media_type == "application/octet-stream":
        media_type = _guess_media_type(file_path, fallback="application/octet-stream")
    return Response(
        content=resp.content,
        media_type=media_type,
        headers={
            "cache-control": "public, max-age=300",
            "content-disposition": "inline",
        },
    )


@app.get("/api/media/stream")
async def media_stream(
    request: Request,
    file_id: str = Query(...),
    init_data: str = Query(default=""),
    token: str = Query(default=""),
) -> Response:
    user = _resolve_media_user(token=token.strip(), init_data=init_data.strip())
    _ = user

    file_path = await _resolve_file_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found.")

    upstream_headers: dict[str, str] = {}
    range_header = request.headers.get("range")
    if range_header:
        upstream_headers["Range"] = range_header

    url = f"{TELEGRAM_FILE_BASE}/{file_path}"
    session = httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True)
    request_obj = session.build_request("GET", url, headers=upstream_headers)
    resp = await session.send(request_obj, stream=True)

    if resp.status_code not in {200, 206}:
        await _close_http_stream(resp, session)
        raise HTTPException(status_code=502, detail=f"Cannot stream file from Telegram ({resp.status_code}).")

    response_headers: dict[str, str] = {}
    for key in ("content-length", "content-range", "accept-ranges", "cache-control", "etag", "last-modified"):
        value = resp.headers.get(key)
        if value:
            response_headers[key] = value
    if "accept-ranges" not in response_headers:
        response_headers["accept-ranges"] = "bytes"
    if "cache-control" not in response_headers:
        response_headers["cache-control"] = "public, max-age=300"
    response_headers["content-disposition"] = "inline"

    media_type = resp.headers.get("content-type") or ""
    if not media_type or media_type == "application/octet-stream":
        media_type = _guess_media_type(file_path, fallback="application/octet-stream")
    return StreamingResponse(
        resp.aiter_bytes(chunk_size=1024 * 256),
        status_code=resp.status_code,
        headers=response_headers,
        media_type=media_type,
        background=BackgroundTask(_close_http_stream, resp, session),
    )


@app.get("/", include_in_schema=False)
async def web_index() -> Response:
    index_file = CLIENT_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return Response(
        "Web client build topilmadi. `webapp/client` ichida npm run build qiling.",
        media_type="text/plain",
        status_code=200,
    )


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> Response:
    if full_path.startswith("api/") or full_path in {"health"}:
        raise HTTPException(status_code=404, detail="Not found.")

    index_file = CLIENT_DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web client build not found.")

    requested = (CLIENT_DIST_DIR / full_path).resolve()
    base_resolved = CLIENT_DIST_DIR.resolve()
    try:
        requested.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found.")

    if requested.exists() and requested.is_file():
        return FileResponse(requested)
    return FileResponse(index_file)
