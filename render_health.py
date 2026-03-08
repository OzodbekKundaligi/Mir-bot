import asyncio
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import re
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import main as bot_main


HOST = os.getenv("WEB_HOST", "0.0.0.0").strip() or "0.0.0.0"
PORT = int((os.getenv("PORT") or os.getenv("WEB_PORT") or "8000").strip())
BASE_DIR = Path(__file__).resolve().parent
MINIAPP_DIR = BASE_DIR / "miniapp"
UPLOADS_DIR = BASE_DIR / "miniapp_uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MAX_JSON_BODY = 6 * 1024 * 1024
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEV_USER_ID = int((os.getenv("MINI_APP_DEV_USER_ID") or "0").strip() or 0)
FILE_PATH_CACHE: dict[str, str] = {}


def env_public_base_url() -> str:
    for env_name in ("PUBLIC_URL", "RENDER_EXTERNAL_URL", "RAILWAY_PUBLIC_DOMAIN"):
        value = os.getenv(env_name, "").strip()
        if not value:
            continue
        if value.startswith(("http://", "https://")):
            return value.rstrip("/")
        return f"https://{value}".rstrip("/")
    return ""


def handler_public_base_url(handler: BaseHTTPRequestHandler) -> str:
    forced = env_public_base_url()
    if forced:
        return forced
    host = (handler.headers.get("X-Forwarded-Host") or handler.headers.get("Host") or "").strip()
    proto = (handler.headers.get("X-Forwarded-Proto") or "http").strip() or "http"
    if not host:
        return ""
    return f"{proto}://{host}".rstrip("/")


def absolute_url(handler: BaseHTTPRequestHandler, path: str) -> str:
    base = handler_public_base_url(handler)
    if not base:
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


def full_name_from_user(user: dict[str, Any]) -> str:
    first_name = str(user.get("first_name") or "").strip()
    last_name = str(user.get("last_name") or "").strip()
    username = str(user.get("username") or "").strip()
    parts = [part for part in (first_name, last_name) if part]
    if parts:
        return " ".join(parts)
    if username:
        return username
    return str(user.get("id") or "User")


def validate_telegram_init_data(init_data: str) -> dict[str, Any] | None:
    raw = (init_data or "").strip()
    if not raw:
        return None

    pairs = urllib.parse.parse_qsl(raw, keep_blank_values=True)
    values = dict(pairs)
    provided_hash = values.pop("hash", "")
    if not provided_hash:
        return None

    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
    secret_key = hmac.new(b"WebAppData", bot_main.BOT_TOKEN.encode("utf-8"), hashlib.sha256).digest()
    calculated_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(calculated_hash, provided_hash):
        return None

    user_raw = values.get("user", "")
    if not user_raw:
        return None
    try:
        user = json.loads(user_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(user, dict):
        return None
    user_id = user.get("id")
    if not isinstance(user_id, int):
        return None
    return user


def authenticated_user_from_headers(handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
    init_data = (handler.headers.get("X-Telegram-Init-Data") or "").strip()
    user = validate_telegram_init_data(init_data)
    if user:
        bot_main.db.add_user(user["id"], full_name_from_user(user))
        return user

    if DEV_USER_ID:
        fallback = bot_main.db.get_user(DEV_USER_ID) or {}
        fallback_name = str(fallback.get("full_name") or f"Dev {DEV_USER_ID}")
        bot_main.db.add_user(DEV_USER_ID, fallback_name)
        return {
            "id": DEV_USER_ID,
            "first_name": fallback_name,
            "last_name": "",
            "username": "",
            "is_dev": True,
        }
    return None


def resolve_telegram_file_url(file_id: str) -> str | None:
    cached = FILE_PATH_CACHE.get(file_id)
    if cached:
        return cached
    request_url = f"https://api.telegram.org/bot{bot_main.BOT_TOKEN}/getFile?file_id={urllib.parse.quote(file_id)}"
    with urllib.request.urlopen(request_url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        return None
    result = payload.get("result") or {}
    file_path = str(result.get("file_path") or "").strip()
    if not file_path:
        return None
    final_url = f"https://api.telegram.org/file/bot{bot_main.BOT_TOKEN}/{file_path}"
    FILE_PATH_CACHE[file_id] = final_url
    return final_url


async def run_payment_request_action(request_id: str, action: str, admin_id: int) -> dict[str, Any]:
    request = bot_main.db.get_payment_request(request_id)
    if not request:
        raise ValueError("Payment request not found")
    current_status = str(request.get("status") or "")
    if current_status != "pending":
        return request

    user_id = int(request.get("user_tg_id") or 0)
    bot = bot_main.Bot(token=bot_main.BOT_TOKEN)
    try:
        if action == "approve":
            if not bot_main.db.resolve_payment_request(request_id, "approved", reviewed_by=admin_id):
                return bot_main.db.get_payment_request(request_id) or request
            bot_main.db.set_pro_state(user_id, True, admin_id=admin_id, note="To'lov tasdiqlandi")
            if user_id and bot_main.db.get_notification_settings(user_id).get("pro_updates"):
                try:
                    await bot.send_message(chat_id=user_id, text="✅ PRO yoqildi.")
                except (bot_main.TelegramBadRequest, bot_main.TelegramForbiddenError):
                    pass
        elif action == "reject":
            if not bot_main.db.resolve_payment_request(request_id, "rejected", reviewed_by=admin_id):
                return bot_main.db.get_payment_request(request_id) or request
            if user_id and bot_main.db.get_notification_settings(user_id).get("pro_updates"):
                try:
                    await bot.send_message(chat_id=user_id, text="❌ PRO rad etildi.")
                except (bot_main.TelegramBadRequest, bot_main.TelegramForbiddenError):
                    pass
        else:
            raise ValueError("Invalid payment action")

        await bot_main.close_review_messages(bot, bot_main.db.get_payment_request_review_messages(request_id))
        bot_main.db.clear_payment_request_review_messages(request_id)
        return bot_main.db.get_payment_request(request_id) or request
    finally:
        await bot.session.close()


async def run_ad_review_action(ad_id: str, action: str, admin_id: int, channel_id: str = "") -> dict[str, Any]:
    ad = bot_main.db.get_ad(ad_id)
    if not ad:
        raise ValueError("Ad not found")
    current_status = str(ad.get("status") or "")
    if current_status != "pending":
        return ad

    bot = bot_main.Bot(token=bot_main.BOT_TOKEN)
    try:
        if action == "reject":
            if not bot_main.db.resolve_ad(ad_id, "rejected", reviewed_by=admin_id):
                return bot_main.db.get_ad(ad_id) or ad
            user_id = int(ad.get("user_tg_id") or 0)
            if user_id and bot_main.db.get_notification_settings(user_id).get("ads_updates"):
                try:
                    await bot.send_message(chat_id=user_id, text="❌ E'lon rad etildi.")
                except (bot_main.TelegramBadRequest, bot_main.TelegramForbiddenError):
                    pass
        elif action == "approve":
            if not channel_id:
                raise ValueError("Channel is required")
            channel = bot_main.db.get_ad_channel(channel_id)
            if not channel:
                raise ValueError("Ad channel not found")
            channel_ref = str(channel.get("channel_ref") or "").strip()
            if not channel_ref:
                raise ValueError("Ad channel is invalid")
            await bot_main.post_ad_to_channel(bot, channel_ref, ad)
            if not bot_main.db.resolve_ad(ad_id, "posted", reviewed_by=admin_id, channel_ref=channel_ref):
                return bot_main.db.get_ad(ad_id) or ad
            user_id = int(ad.get("user_tg_id") or 0)
            if user_id and bot_main.db.get_notification_settings(user_id).get("ads_updates"):
                try:
                    await bot.send_message(chat_id=user_id, text=f"✅ E'lon joylandi: {channel.get('title') or channel_ref}")
                except (bot_main.TelegramBadRequest, bot_main.TelegramForbiddenError):
                    pass
        else:
            raise ValueError("Invalid ad action")

        await bot_main.close_review_messages(bot, bot_main.db.get_ad_review_messages(ad_id))
        bot_main.db.clear_ad_review_messages(ad_id)
        return bot_main.db.get_ad(ad_id) or ad
    finally:
        await bot.session.close()


async def resolve_ad_channel_create(channel_input: str) -> dict[str, Any]:
    channel_ref = bot_main.normalize_channel_ref_input(channel_input)
    if not channel_ref:
        raise ValueError("Channel username or ID is invalid")
    title = channel_ref
    bot = bot_main.Bot(token=bot_main.BOT_TOKEN)
    try:
        try:
            chat = await bot.get_chat(channel_ref)
            title = str(getattr(chat, "title", "") or getattr(chat, "username", "") or channel_ref)
        except bot_main.TelegramBadRequest:
            pass
    finally:
        await bot.session.close()
    created = bot_main.db.add_ad_channel(channel_ref, title=title)
    result = {
        "created": created,
        "channel": None,
    }
    if created:
        channels = bot_main.db.list_ad_channels()
        for row in channels:
            if str(row.get("channel_ref") or "") == channel_ref:
                result["channel"] = row
                break
    return result


def serialize_content_item(item: dict[str, Any], user_id: int, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_type = str(item.get("content_type") or "").strip()
    content_ref = str(item.get("id") or item.get("content_ref") or "").strip()
    reaction = bot_main.db.get_reaction_summary(content_type, content_ref)
    preview = bot_main.resolve_inline_media_preview(item)
    if not preview and content_type == "serial" and content_ref:
        preview = bot_main.db.get_serial_inline_media_preview(content_ref)

    preview_kind = ""
    preview_url = ""
    if preview:
        preview_kind, preview_file_id = preview
        if preview_file_id.startswith(("http://", "https://")):
            preview_url = preview_file_id
        else:
            preview_url = f"/api/telegram-file?file_id={urllib.parse.quote(preview_file_id)}"

    episodes_count = None
    if content_type == "serial" and content_ref:
        episodes_count = len(bot_main.db.list_serial_episodes(content_ref))

    return {
        "id": content_ref,
        "content_type": content_type,
        "code": str(item.get("code") or ""),
        "title": str(item.get("title") or ""),
        "description": str(item.get("description") or ""),
        "year": item.get("year"),
        "quality": str(item.get("quality") or ""),
        "genres": [str(genre) for genre in (item.get("genres") or []) if str(genre).strip()],
        "views": int(item.get("views") or 0),
        "downloads": int(item.get("downloads") or 0),
        "created_at": str(item.get("created_at") or ""),
        "episodes_count": episodes_count,
        "is_favorite": bot_main.db.is_favorite(user_id, content_type, content_ref),
        "user_reaction": bot_main.db.get_user_reaction(user_id, content_type, content_ref),
        "likes": int(reaction.get("likes") or 0),
        "dislikes": int(reaction.get("dislikes") or 0),
        "rating": float(reaction.get("rating") or 0.0),
        "preview_kind": preview_kind,
        "preview_url": preview_url,
        "open_payload": {
            "action": "open_content",
            "content_type": content_type,
            "content_ref": content_ref,
        },
        "share_text": f"{item.get('title') or ''} ({item.get('code') or ''})".strip(),
        "public_preview_url": absolute_url(handler, preview_url) if preview_url.startswith("/") else preview_url,
    }


def load_content(content_type: str, content_ref: str) -> dict[str, Any] | None:
    if content_type == "movie":
        row = bot_main.db.get_movie_by_id(content_ref) or bot_main.db.get_movie(content_ref)
    elif content_type == "serial":
        row = bot_main.db.get_serial(content_ref) or bot_main.db.get_serial_by_code(content_ref)
    else:
        return None
    if not row:
        return None
    normalized = dict(row)
    normalized["content_type"] = content_type
    return normalized


def build_recent_movies(limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bot_main.db.list_movies(limit=limit):
        row = bot_main.db.get_movie_by_id(str(item.get("id") or ""))
        if not row:
            continue
        row["content_type"] = "movie"
        rows.append(row)
    return rows


def build_recent_serials(limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bot_main.db.list_serials(limit=limit):
        row = bot_main.db.get_serial(str(item.get("id") or ""))
        if not row:
            continue
        row["content_type"] = "serial"
        rows.append(row)
    return rows


def serialize_ad(ad: dict[str, Any], handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    photo = str(ad.get("photo_file_id") or "").strip()
    if photo and photo.startswith("/"):
        photo = absolute_url(handler, photo)
    return {
        "id": str(ad.get("id") or ""),
        "user_tg_id": int(ad.get("user_tg_id") or 0),
        "title": str(ad.get("title") or ""),
        "description": str(ad.get("description") or ""),
        "photo_url": photo,
        "button_text": str(ad.get("button_text") or ""),
        "button_url": str(ad.get("button_url") or ""),
        "status": str(ad.get("status") or ""),
        "created_at": str(ad.get("created_at") or ""),
        "review_note": str(ad.get("review_note") or ""),
        "channel_ref": str(ad.get("channel_ref") or ""),
    }


def serialize_user_summary(row: dict[str, Any]) -> dict[str, Any]:
    user_id = int(row.get("tg_id") or 0)
    pro_info = bot_main.db.get_pro_status(user_id) if user_id else {}
    return {
        "id": user_id,
        "full_name": str(row.get("full_name") or user_id or "User"),
        "joined_at": str(row.get("joined_at") or ""),
        "is_admin": bot_main.db.is_admin(user_id) if user_id else False,
        "is_seed_admin": user_id in set(getattr(bot_main, "ADMIN_IDS", []) or []),
        "is_pro": bool(pro_info.get("is_pro")),
        "pro_status": str(pro_info.get("pro_status") or row.get("pro_status") or ""),
        "pro_until": str(pro_info.get("pro_until") or row.get("pro_until") or ""),
        "pro_note": str(row.get("pro_note") or ""),
    }


def bootstrap_payload(user: dict[str, Any], handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    user_id = int(user["id"])
    settings = bot_main.db.get_bot_settings()
    pro_info = bot_main.db.get_pro_status(user_id)
    notifications = bot_main.db.get_notification_settings(user_id)
    notice = bot_main.db.get_site_notice()
    if notice.get("link", "").startswith("/"):
        notice["link"] = absolute_url(handler, notice["link"])

    recent_movies = [serialize_content_item(row, user_id, handler) for row in build_recent_movies(10)]
    recent_serials = [serialize_content_item(row, user_id, handler) for row in build_recent_serials(10)]

    top_viewed: list[dict[str, Any]] = []
    for row in bot_main.db.list_top_viewed_content(limit=10):
        content_type = str(row.get("content_type") or "")
        content_ref = str(row.get("id") or "")
        full_row = load_content(content_type, content_ref)
        if not full_row:
            continue
        top_viewed.append(serialize_content_item(full_row, user_id, handler))

    favorites: list[dict[str, Any]] = []
    for row in bot_main.db.list_favorites(user_id, limit=20):
        content_type = str(row.get("content_type") or "")
        content_ref = str(row.get("content_ref") or "")
        full_row = load_content(content_type, content_ref)
        if not full_row:
            continue
        favorites.append(serialize_content_item(full_row, user_id, handler))

    payload: dict[str, Any] = {
        "ok": True,
        "user": {
            "id": user_id,
            "full_name": full_name_from_user(user),
            "username": str(user.get("username") or ""),
            "is_admin": bot_main.db.is_admin(user_id),
            "is_pro": bool(pro_info.get("is_pro")),
            "pro_status": str(pro_info.get("pro_status") or ""),
            "pro_until": str(pro_info.get("pro_until") or ""),
        },
        "settings": {
            "pro_price_text": settings["pro_price_text"],
            "pro_duration_days": settings["pro_duration_days"],
            "content_mode": settings["content_mode"],
            "content_mode_label": bot_main.content_mode_label(settings["content_mode"]),
        },
        "notifications": {
            "new_content": bool(notifications.get("new_content")),
            "pro_updates": bool(notifications.get("pro_updates")),
            "ads_updates": bool(notifications.get("ads_updates")),
        },
        "payment": {
            "code": bot_main.format_pro_payment_code(user_id),
            "links": [
                link
                for link in (bot_main.PRO_PAYMENT_LINK_1, bot_main.PRO_PAYMENT_LINK_2)
                if str(link or "").strip()
            ],
        },
        "notice": notice,
        "sections": {
            "recent_movies": recent_movies,
            "recent_serials": recent_serials,
            "top_viewed": top_viewed,
            "favorites": favorites,
            "open_requests": bot_main.db.list_open_request_topics(limit=6),
        },
        "ads": {
            "can_create": bot_main.db.is_pro_active(user_id),
            "mine": [serialize_ad(row, handler) for row in bot_main.db.list_user_ads(user_id, limit=20)],
        },
        "links": {
            "mini_app": absolute_url(handler, "/app/"),
        },
    }

    if bot_main.db.is_admin(user_id):
        payload["admin"] = {
            "total_users": len(bot_main.db.list_user_ids()),
            "total_pro_users": len(bot_main.db.list_active_pro_user_ids()),
            "total_movies": len(bot_main.db.list_movies(limit=None)),
            "total_serials": len(bot_main.db.list_serials(limit=None)),
            "pending_payment_count": bot_main.db.payment_requests.count_documents({"status": "pending"}),
            "pending_ads_count": bot_main.db.ads.count_documents({"status": "pending"}),
            "pending_payments": [
                {
                    "id": str(row.get("id") or ""),
                    "user_tg_id": int(row.get("user_tg_id") or 0),
                    "payment_code": str(row.get("payment_code") or ""),
                    "comment": str(row.get("comment") or ""),
                    "created_at": str(row.get("created_at") or ""),
                    "status": str(row.get("status") or ""),
                }
                for row in bot_main.db.list_pending_payment_requests(limit=8)
            ],
            "pending_ads": [serialize_ad(row, handler) for row in bot_main.db.list_pending_ads(limit=8)],
            "ad_channels": [
                {
                    "id": str(row.get("id") or ""),
                    "title": str(row.get("title") or row.get("channel_ref") or ""),
                    "channel_ref": str(row.get("channel_ref") or ""),
                }
                for row in bot_main.db.list_ad_channels()
            ],
            "recent_users": [serialize_user_summary(row) for row in bot_main.db.search_users("", limit=8)],
        }

    return payload


class AppHandler(BaseHTTPRequestHandler):
    server_version = "KinoBotMiniApp/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _write_bytes(self, status_code: int, body: bytes, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._write_bytes(status_code, body, "application/json; charset=utf-8")

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = int(self.headers.get("Content-Length") or "0")
        if raw_length <= 0 or raw_length > MAX_JSON_BODY:
            raise ValueError("Invalid body size")
        payload = self.rfile.read(raw_length)
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("JSON parse error") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        return data

    def _require_user(self) -> dict[str, Any] | None:
        user = authenticated_user_from_headers(self)
        if not user:
            self._write_json(401, {"ok": False, "detail": "Unauthorized"})
            return None
        return user

    def _serve_static(self, relative_path: str) -> None:
        target = (MINIAPP_DIR / relative_path).resolve()
        if MINIAPP_DIR not in target.parents and target != MINIAPP_DIR:
            self._write_json(404, {"ok": False, "detail": "Not found"})
            return
        if not target.exists() or not target.is_file():
            self._write_json(404, {"ok": False, "detail": "Not found"})
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._write_bytes(200, target.read_bytes(), content_type)

    def _serve_upload(self, relative_name: str) -> None:
        safe_name = Path(relative_name).name
        target = (UPLOADS_DIR / safe_name).resolve()
        if not target.exists() or not target.is_file():
            self._write_json(404, {"ok": False, "detail": "Not found"})
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with target.open("rb") as fh:
            self.wfile.write(fh.read())

    def _handle_telegram_file_proxy(self, file_id: str) -> None:
        if not file_id:
            self._write_json(400, {"ok": False, "detail": "file_id required"})
            return
        try:
            if file_id.startswith(("http://", "https://")):
                with urllib.request.urlopen(file_id, timeout=20) as response:
                    body = response.read()
                    content_type = response.headers.get_content_type() or "application/octet-stream"
            else:
                source_url = resolve_telegram_file_url(file_id)
                if not source_url:
                    self._write_json(404, {"ok": False, "detail": "Media not found"})
                    return
                with urllib.request.urlopen(source_url, timeout=30) as response:
                    body = response.read()
                    content_type = response.headers.get_content_type() or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self._write_json(502, {"ok": False, "detail": f"Media proxy error: {exc}"})

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        query = urllib.parse.parse_qs(parsed.query)

        if path in {"", "/"}:
            self._write_json(200, {"ok": True, "service": "kino-bot", "mini_app": "/app/"})
            return
        if path == "/health":
            self._write_json(200, {"ok": True})
            return
        if path == "/app":
            self._redirect("/app/")
            return
        if path == "/app/":
            self._serve_static("index.html")
            return
        if path.startswith("/app/"):
            self._serve_static(path.replace("/app/", "", 1))
            return
        if path.startswith("/uploads/"):
            self._serve_upload(path.replace("/uploads/", "", 1))
            return
        if path == "/api/bootstrap":
            user = self._require_user()
            if not user:
                return
            self._write_json(200, bootstrap_payload(user, self))
            return
        if path == "/api/search":
            user = self._require_user()
            if not user:
                return
            q = str(query.get("q", [""])[0] or "").strip()
            content_filter = str(query.get("type", ["all"])[0] or "all").strip().lower()
            results = []
            for row in bot_main.db.search_content(q, limit=24):
                if content_filter in {"movie", "serial"} and str(row.get("content_type") or "") != content_filter:
                    continue
                results.append(serialize_content_item(row, int(user["id"]), self))
            self._write_json(200, {"ok": True, "items": results})
            return
        if path.startswith("/api/content/"):
            user = self._require_user()
            if not user:
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) != 4:
                self._write_json(404, {"ok": False, "detail": "Not found"})
                return
            _, _, content_type, content_ref = parts
            item = load_content(content_type, content_ref)
            if not item:
                self._write_json(404, {"ok": False, "detail": "Content not found"})
                return
            payload = {"ok": True, "item": serialize_content_item(item, int(user["id"]), self)}
            if content_type == "serial":
                payload["item"]["episodes"] = [
                    {"episode_number": int(row.get("episode_number") or 0)}
                    for row in bot_main.db.list_serial_episodes(content_ref)
                ]
            self._write_json(200, payload)
            return
        if path == "/api/ads/mine":
            user = self._require_user()
            if not user:
                return
            ads = [serialize_ad(row, self) for row in bot_main.db.list_user_ads(int(user["id"]), limit=30)]
            self._write_json(200, {"ok": True, "items": ads})
            return
        if path == "/api/admin/users/search":
            user = self._require_user()
            if not user:
                return
            if not bot_main.db.is_admin(int(user["id"])):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            q = str(query.get("q", [""])[0] or "").strip()
            items = [serialize_user_summary(row) for row in bot_main.db.search_users(q, limit=20)]
            self._write_json(200, {"ok": True, "items": items})
            return
        if path == "/api/telegram-file":
            file_id = str(query.get("file_id", [""])[0] or "")
            self._handle_telegram_file_proxy(file_id)
            return

        self._write_json(404, {"ok": False, "detail": "Not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"

        if path == "/api/favorites/toggle":
            user = self._require_user()
            if not user:
                return
            try:
                payload = self._read_json_body()
                content_type = str(payload.get("contentType") or "").strip()
                content_ref = str(payload.get("contentRef") or "").strip()
                if content_type not in {"movie", "serial"} or not content_ref:
                    raise ValueError("Invalid content")
                user_id = int(user["id"])
                if bot_main.db.is_favorite(user_id, content_type, content_ref):
                    active = not bot_main.db.remove_favorite(user_id, content_type, content_ref)
                else:
                    active = bool(bot_main.db.add_favorite(user_id, content_type, content_ref))
                self._write_json(200, {"ok": True, "active": active})
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/reactions":
            user = self._require_user()
            if not user:
                return
            try:
                payload = self._read_json_body()
                content_type = str(payload.get("contentType") or "").strip()
                content_ref = str(payload.get("contentRef") or "").strip()
                reaction = str(payload.get("reaction") or "").strip()
                summary = bot_main.db.set_reaction(int(user["id"]), content_type, content_ref, reaction)
                current = bot_main.db.get_user_reaction(int(user["id"]), content_type, content_ref)
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "likes": int(summary.get("likes") or 0),
                        "dislikes": int(summary.get("dislikes") or 0),
                        "rating": float(summary.get("rating") or 0.0),
                        "user_reaction": current,
                    },
                )
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/notifications/toggle":
            user = self._require_user()
            if not user:
                return
            try:
                payload = self._read_json_body()
                key = str(payload.get("key") or "").strip()
                updated = bot_main.db.toggle_notification_setting(int(user["id"]), key)
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "settings": {
                            "new_content": bool(updated.get("new_content")),
                            "pro_updates": bool(updated.get("pro_updates")),
                            "ads_updates": bool(updated.get("ads_updates")),
                        },
                    },
                )
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/upload-image":
            user = self._require_user()
            if not user:
                return
            try:
                payload = self._read_json_body()
                data_url = str(payload.get("dataUrl") or "").strip()
                if not data_url.startswith("data:image/"):
                    raise ValueError("Image data required")
                match = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", data_url, flags=re.DOTALL)
                if not match:
                    raise ValueError("Invalid image data")
                mime_type = match.group(1)
                raw = base64.b64decode(match.group(2), validate=True)
                if not raw or len(raw) > MAX_UPLOAD_BYTES:
                    raise ValueError("Image is too large")
                ext = mimetypes.guess_extension(mime_type) or ".jpg"
                filename = f"{uuid.uuid4().hex}{ext}"
                target = UPLOADS_DIR / filename
                target.write_bytes(raw)
                public_url = absolute_url(self, f"/uploads/{filename}")
                self._write_json(200, {"ok": True, "url": public_url, "path": f"/uploads/{filename}"})
                return
            except Exception as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/ads":
            user = self._require_user()
            if not user:
                return
            user_id = int(user["id"])
            if not bot_main.db.is_pro_active(user_id):
                self._write_json(403, {"ok": False, "detail": "PRO required"})
                return
            try:
                payload = self._read_json_body()
                title = str(payload.get("title") or "").strip()
                description = str(payload.get("description") or "").strip()
                photo_url = str(payload.get("photoUrl") or "").strip()
                button_text = str(payload.get("buttonText") or "").strip()
                button_url = str(payload.get("buttonUrl") or "").strip()
                if len(title) < 3:
                    raise ValueError("Title is too short")
                if len(description) < 5:
                    raise ValueError("Description is too short")
                normalized_button_url = bot_main.normalize_button_url(button_url) if button_url else None
                if button_url and not normalized_button_url:
                    raise ValueError("Button URL is invalid")
                ad_id = bot_main.db.create_ad(
                    user_id,
                    title,
                    description,
                    photo_file_id=photo_url,
                    button_text=button_text,
                    button_url=normalized_button_url or "",
                )
                self._write_json(201, {"ok": True, "id": ad_id})
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/content-mode":
            user = self._require_user()
            if not user:
                return
            if not bot_main.db.is_admin(int(user["id"])):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                mode = bot_main.normalize_content_mode(payload.get("mode"))
                bot_main.db.set_content_mode(mode)
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "content_mode": mode,
                        "content_mode_label": bot_main.content_mode_label(mode),
                    },
                )
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/notice":
            user = self._require_user()
            if not user:
                return
            if not bot_main.db.is_admin(int(user["id"])):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                text = str(payload.get("text") or "").strip()
                link = str(payload.get("link") or "").strip()
                if link and not link.startswith(("http://", "https://", "/app")):
                    raise ValueError("Notice link is invalid")
                bot_main.db.set_site_notice(text, admin_id=int(user["id"]), link=link)
                self._write_json(200, {"ok": True, "notice": bot_main.db.get_site_notice()})
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/pro-settings":
            user = self._require_user()
            if not user:
                return
            if not bot_main.db.is_admin(int(user["id"])):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                price_text = str(payload.get("priceText") or "").strip()
                duration_days_raw = payload.get("durationDays")
                if len(price_text) < 3:
                    raise ValueError("PRO price text is too short")
                try:
                    duration_days = int(duration_days_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Duration must be a number") from exc
                if duration_days < 1 or duration_days > 3650:
                    raise ValueError("Duration must be between 1 and 3650 days")
                bot_main.db.set_pro_price_text(price_text)
                bot_main.db.set_pro_duration_days(duration_days)
                settings = bot_main.db.get_bot_settings()
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "settings": {
                            "pro_price_text": settings["pro_price_text"],
                            "pro_duration_days": settings["pro_duration_days"],
                            "content_mode": settings["content_mode"],
                            "content_mode_label": bot_main.content_mode_label(settings["content_mode"]),
                        },
                    },
                )
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/ad-channels/create":
            user = self._require_user()
            if not user:
                return
            if not bot_main.db.is_admin(int(user["id"])):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                channel_input = str(payload.get("channelRef") or "").strip()
                result = asyncio.run(resolve_ad_channel_create(channel_input))
                channel = result.get("channel") or {}
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "created": bool(result.get("created")),
                        "channel": {
                            "id": str(channel.get("id") or ""),
                            "title": str(channel.get("title") or channel.get("channel_ref") or ""),
                            "channel_ref": str(channel.get("channel_ref") or ""),
                        } if channel else None,
                    },
                )
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/ad-channels/delete":
            user = self._require_user()
            if not user:
                return
            if not bot_main.db.is_admin(int(user["id"])):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                channel_id = str(payload.get("channelId") or "").strip()
                if not channel_id:
                    raise ValueError("Channel ID is required")
                removed = bot_main.db.remove_ad_channel(channel_id)
                self._write_json(200, {"ok": True, "removed": removed})
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/users/pro":
            user = self._require_user()
            if not user:
                return
            admin_id = int(user["id"])
            if not bot_main.db.is_admin(admin_id):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                target_user_id = int(payload.get("userId") or 0)
                enabled = bool(payload.get("enabled"))
                if target_user_id <= 0:
                    raise ValueError("User ID is invalid")
                note = "Mini App admin"
                bot_main.db.set_pro_state(target_user_id, enabled, admin_id=admin_id, note=note)
                item = serialize_user_summary(bot_main.db.get_user(target_user_id) or {"tg_id": target_user_id})
                self._write_json(200, {"ok": True, "item": item})
                return
            except (TypeError, ValueError) as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/users/admin":
            user = self._require_user()
            if not user:
                return
            admin_id = int(user["id"])
            if not bot_main.db.is_admin(admin_id):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                target_user_id = int(payload.get("userId") or 0)
                enabled = bool(payload.get("enabled"))
                if target_user_id <= 0:
                    raise ValueError("User ID is invalid")
                if target_user_id == admin_id and not enabled:
                    raise ValueError("You cannot remove your own admin access")
                if not enabled and target_user_id in set(getattr(bot_main, "ADMIN_IDS", []) or []):
                    raise ValueError("Seed admins are managed from environment")
                changed = bot_main.db.add_admin(target_user_id) if enabled else bot_main.db.remove_admin(target_user_id)
                item = serialize_user_summary(bot_main.db.get_user(target_user_id) or {"tg_id": target_user_id})
                self._write_json(200, {"ok": True, "changed": changed, "item": item})
                return
            except (TypeError, ValueError) as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/payments/review":
            user = self._require_user()
            if not user:
                return
            admin_id = int(user["id"])
            if not bot_main.db.is_admin(admin_id):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                request_id = str(payload.get("requestId") or "").strip()
                action = str(payload.get("action") or "").strip().lower()
                if action not in {"approve", "reject"}:
                    raise ValueError("Invalid payment action")
                result = asyncio.run(run_payment_request_action(request_id, action, admin_id))
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "item": {
                            "id": str(result.get("id") or ""),
                            "user_tg_id": int(result.get("user_tg_id") or 0),
                            "payment_code": str(result.get("payment_code") or ""),
                            "comment": str(result.get("comment") or ""),
                            "created_at": str(result.get("created_at") or ""),
                            "status": str(result.get("status") or ""),
                        },
                    },
                )
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        if path == "/api/admin/ads/review":
            user = self._require_user()
            if not user:
                return
            admin_id = int(user["id"])
            if not bot_main.db.is_admin(admin_id):
                self._write_json(403, {"ok": False, "detail": "Admin required"})
                return
            try:
                payload = self._read_json_body()
                ad_id = str(payload.get("adId") or "").strip()
                action = str(payload.get("action") or "").strip().lower()
                channel_id = str(payload.get("channelId") or "").strip()
                if action not in {"approve", "reject"}:
                    raise ValueError("Invalid ad action")
                result = asyncio.run(run_ad_review_action(ad_id, action, admin_id, channel_id))
                self._write_json(200, {"ok": True, "item": serialize_ad(result, self)})
                return
            except ValueError as exc:
                self._write_json(400, {"ok": False, "detail": str(exc)})
                return

        self._write_json(404, {"ok": False, "detail": "Not found"})


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"[health] listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()
