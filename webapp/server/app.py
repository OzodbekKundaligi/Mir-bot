import hashlib
import hmac
import json
import os
import re
import time
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl

import httpx
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import DESCENDING, MongoClient


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "8537979650:AAFkSIbRnx7ha7muxZ1MDK5QMIxV5MAC4ww").strip()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:wGVAMNxMWZgocdRVBduRDnRlJePweOay@metro.proxy.rlwy.net:36399").strip()
MONGODB_DB = os.getenv("MONGODB_DB", "kino_bot").strip() or "kino_bot"
BOT_USERNAME = os.getenv("BOT_USERNAME", "@MirTopKinoBot").strip()
WEBAPP_ALLOWED_ORIGINS = [item.strip() for item in os.getenv("WEBAPP_ALLOWED_ORIGINS", "mir-bot-production.up.railway.app").split(",") if item.strip()]
WEBAPP_AUTH_MAX_AGE_SECONDS = int(os.getenv("WEBAPP_AUTH_MAX_AGE_SECONDS", "604800"))
WEBAPP_ENABLE_DEV_AUTH = os.getenv("WEBAPP_ENABLE_DEV_AUTH", "0").strip() in {"1", "true", "yes"}
WEBAPP_DEV_USER_ID = int(os.getenv("WEBAPP_DEV_USER_ID", "0") or 0)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required for webapp backend.")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is required for webapp backend.")


client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
db = client[MONGODB_DB]

users_col = db["users"]
required_channels_col = db["required_channels"]
join_requests_col = db["join_requests"]
movies_col = db["movies"]
serials_col = db["serials"]
favorites_col = db["favorites"]
reactions_col = db["web_reactions"]
comments_col = db["web_comments"]
history_col = db["web_history"]
downloads_col = db["web_downloads"]

reactions_col.create_index([("user_tg_id", 1), ("content_type", 1), ("content_ref", 1)], unique=True)
comments_col.create_index([("content_type", 1), ("content_ref", 1), ("created_at", -1)])
history_col.create_index([("user_tg_id", 1), ("viewed_at", -1)])
downloads_col.create_index([("user_tg_id", 1), ("created_at", -1)])


app = FastAPI(title="Kino Bot WebApp API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=WEBAPP_ALLOWED_ORIGINS if WEBAPP_ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    "created_at": 1,
}


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


class DownloadTrackIn(BaseModel):
    content_type: str
    content_ref: str


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
    payload = f"m_{content_id}" if content_type == "movie" else f"s_{content_id}"
    deep_link = f"https://t.me/{bot_username}?start={payload}" if bot_username else ""
    preview_file_id = _extract_preview_photo_file_id(content_type, doc)
    return {
        "id": content_id,
        "content_type": content_type,
        "code": str(doc.get("code") or ""),
        "title": str(doc.get("title") or ""),
        "description": str(doc.get("description") or ""),
        "year": doc.get("year"),
        "quality": str(doc.get("quality") or ""),
        "genres": [str(g) for g in doc.get("genres", []) if str(g).strip()],
        "downloads": int(doc.get("downloads") or 0),
        "preview_file_id": preview_file_id,
        "deep_link": deep_link,
        "created_at": str(doc.get("created_at") or ""),
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_object_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except Exception:
        return None


def _get_content_doc(content_type: str, content_ref: str) -> dict[str, Any] | None:
    oid = _safe_object_id(content_ref)
    if not oid:
        return None
    if content_type == "movie":
        return movies_col.find_one({"_id": oid})
    if content_type == "serial":
        return serials_col.find_one({"_id": oid})
    return None


def _content_key(content_type: str, content_ref: str) -> str:
    return f"{content_type}:{content_ref}"


def _content_match_clauses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = {"movie": set(), "serial": set()}
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
        "user_tg_id": int(doc.get("user_tg_id") or 0),
        "full_name": str(doc.get("full_name") or ""),
        "username": str(doc.get("username") or ""),
        "text": str(doc.get("text") or ""),
        "created_at": str(doc.get("created_at") or ""),
    }


def _track_history_view(user_tg_id: int, content_type: str, content_ref: str) -> None:
    if content_type not in {"movie", "serial"} or not content_ref:
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
        doc = _get_content_doc(content_type, content_ref)
        if not doc:
            continue
        rows.append(_serialize_content(doc, content_type, bot_username))
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


def _list_content(
    query: str,
    content_type: str,
    limit: int,
    bot_username: str | None,
    user_tg_id: int,
) -> list[dict[str, Any]]:
    limit_safe = max(1, min(limit, 200))
    mongo_filter = _query_filter(query)
    rows: list[dict[str, Any]] = []

    if content_type in {"all", "movie"}:
        for doc in movies_col.find(mongo_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
            rows.append(_serialize_content(doc, "movie", bot_username))

    if content_type in {"all", "serial"}:
        for doc in serials_col.find(mongo_filter, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(limit_safe):
            rows.append(_serialize_content(doc, "serial", bot_username))

    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
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

    for doc in movies_col.find({}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
        rows.append(_serialize_content(doc, "movie", bot_username))
    for doc in serials_col.find({}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(scan_limit):
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

    collection = movies_col if content_type == "movie" else serials_col
    scored: list[dict[str, Any]] = []

    for doc in collection.find({}, CONTENT_PROJECTION).sort("created_at", DESCENDING).limit(260):
        doc_id = str(doc.get("_id") or "")
        if not doc_id or doc_id == content_ref:
            continue

        item = _serialize_content(doc, content_type, bot_username)
        item_genres = {_normalize_lookup(str(value)) for value in doc.get("genres", []) if str(value).strip()}
        overlap = len(base_genres & item_genres)

        title_norm = _normalize_lookup(str(doc.get("title") or ""))
        title_tokens = {token for token in title_norm.split() if token}
        token_overlap = len(base_tokens & title_tokens)

        score = overlap * 7
        score += min(token_overlap, 4) * 2

        doc_year = doc.get("year") if isinstance(doc.get("year"), int) else None
        if base_year is not None and doc_year is not None and abs(base_year - doc_year) <= 1:
            score += 2

        score += min(int(doc.get("downloads") or 0) // 50, 4)
        item["recommend_score"] = score
        scored.append(item)

    scored.sort(key=lambda x: (int(x.get("recommend_score") or 0), str(x.get("created_at") or "")), reverse=True)

    selected: list[dict[str, Any]] = []
    for row in scored:
        if int(row.get("recommend_score") or 0) <= 0 and len(selected) >= limit_safe:
            break
        row.pop("recommend_score", None)
        selected.append(row)
        if len(selected) >= limit_safe:
            break

    return _attach_engagement(selected, user_tg_id)


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


def _resolve_user(init_data: str) -> dict[str, Any]:
    if init_data:
        return _verify_webapp_init_data(init_data)
    if WEBAPP_ENABLE_DEV_AUTH and WEBAPP_DEV_USER_ID > 0:
        return _dev_user_payload()
    raise HTTPException(status_code=401, detail="Telegram auth required.")


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
    channels = list(required_channels_col.find({"is_active": True}).sort("created_at", DESCENDING))
    if not channels:
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


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/bootstrap")
async def bootstrap(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    user_id = int(user["id"])
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
    return {
        "user": {
            "id": user_id,
            "first_name": str(user.get("first_name") or ""),
            "last_name": str(user.get("last_name") or ""),
            "username": str(user.get("username") or ""),
        },
        "blocked": blocked,
        "missing_channels": missing_channels,
        "content": content,
        "favorites": favorites,
        "history": history,
        "trending": trending,
        "stats": {
            "movies": movies_col.count_documents({}),
            "serials": serials_col.count_documents({}),
            "favorites": favorites_col.count_documents({"user_tg_id": user_id}),
            "comments": comments_col.count_documents({"user_tg_id": user_id}),
            "downloads": downloads_col.count_documents({"user_tg_id": user_id}),
            "likes_given": reactions_col.count_documents({"user_tg_id": user_id, "reaction": "like"}),
            "dislikes_given": reactions_col.count_documents({"user_tg_id": user_id, "reaction": "dislike"}),
            "history_views": history_col.count_documents({"user_tg_id": user_id}),
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
    content_type_safe = content_type if content_type in {"all", "movie", "serial"} else "all"
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
    content_type_safe = content_type if content_type in {"movie", "serial"} else ""
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
    return {"item": item, "recommendations": recommendations}


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
    return {"items": items}


@app.get("/api/favorites")
async def favorites(user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    return {"items": _list_favorites(user_id, bot_username)}


@app.post("/api/favorites/toggle")
async def favorite_toggle(payload: FavoriteToggleIn, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    if content_type not in {"movie", "serial"} or not content_ref:
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
    if content_type not in {"movie", "serial"} or not content_ref:
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
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    _ = user
    if content_type not in {"movie", "serial"}:
        raise HTTPException(status_code=400, detail="Invalid content_type.")
    docs = comments_col.find({"content_type": content_type, "content_ref": content_ref}).sort("created_at", DESCENDING).limit(limit)
    return {"items": [_serialize_comment(doc) for doc in docs]}


@app.post("/api/comments/add")
async def add_comment(payload: CommentIn, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    text = re.sub(r"\s+", " ", payload.text.strip())
    if content_type not in {"movie", "serial"} or not content_ref:
        raise HTTPException(status_code=400, detail="Invalid payload.")
    if len(text) < 2:
        raise HTTPException(status_code=400, detail="Comment too short.")
    if len(text) > 600:
        raise HTTPException(status_code=400, detail="Comment too long.")
    if not _get_content_doc(content_type, content_ref):
        raise HTTPException(status_code=404, detail="Content not found.")

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
    result = comments_col.insert_one(doc)
    doc["_id"] = result.inserted_id
    summary = _engagement_summary(content_type, content_ref, user_id)
    return {"ok": True, "item": _serialize_comment(doc), "summary": summary}


@app.post("/api/downloads/track")
async def track_download(payload: DownloadTrackIn, user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    content_type = payload.content_type.strip()
    content_ref = payload.content_ref.strip()
    if content_type not in {"movie", "serial"} or not content_ref:
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


@app.get("/api/history")
async def history(
    limit: int = Query(default=20, ge=1, le=100),
    user: dict[str, Any] = Depends(_require_subscribed),
) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    return {"items": _list_recent_history(user_id, bot_username, limit=limit)}


@app.get("/api/profile")
async def profile(user: dict[str, Any] = Depends(_require_subscribed)) -> dict[str, Any]:
    user_id = int(user["id"])
    bot_username = await _get_bot_username()
    return {
        "user": {
            "id": user_id,
            "first_name": str(user.get("first_name") or ""),
            "last_name": str(user.get("last_name") or ""),
            "username": str(user.get("username") or ""),
        },
        "stats": {
            "movies": movies_col.count_documents({}),
            "serials": serials_col.count_documents({}),
            "favorites": favorites_col.count_documents({"user_tg_id": user_id}),
            "comments": comments_col.count_documents({"user_tg_id": user_id}),
            "downloads": downloads_col.count_documents({"user_tg_id": user_id}),
            "likes_given": reactions_col.count_documents({"user_tg_id": user_id, "reaction": "like"}),
            "dislikes_given": reactions_col.count_documents({"user_tg_id": user_id, "reaction": "dislike"}),
            "history_views": history_col.count_documents({"user_tg_id": user_id}),
        },
        "history": _list_recent_history(user_id, bot_username, limit=18),
    }


@app.get("/api/media/file")
async def media_file(
    file_id: str = Query(...),
    init_data: str = Query(default=""),
) -> Response:
    user = _resolve_user(init_data.strip())
    user_id = int(user["id"])
    missing = await _get_missing_channels(user_id)
    if missing:
        raise HTTPException(status_code=403, detail="Subscription required.")

    result = await _telegram_api("getFile", {"file_id": file_id})
    file_path = str(result.get("file_path") or "").strip()
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found.")

    url = f"{TELEGRAM_FILE_BASE}/{file_path}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as session:
        resp = await session.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Cannot download file from Telegram.")
    media_type = resp.headers.get("content-type") or "application/octet-stream"
    return Response(content=resp.content, media_type=media_type)
