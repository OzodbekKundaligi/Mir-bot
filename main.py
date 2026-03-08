import asyncio
import base64
import hashlib
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Iterable
from urllib.parse import urlparse

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatMemberStatus, ContentType
from aiogram.exceptions import ClientDecodeError, TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    ChatJoinRequest,
    CallbackQuery,
    InlineQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InlineQueryResultCachedPhoto,
    InlineQueryResultCachedVideo,
    InputTextMessageContent,
    KeyboardButton,
    MessageOriginChannel,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Movie:
    code: str
    title: str
    description: str
    media_type: str
    file_id: str
    year: int | None = None
    quality: str = ""
    genres: list[str] | None = None
    preview_media_type: str = ""
    preview_file_id: str = ""


class Database:
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        self.db = self.client[db_name]

        self.admins = self.db["admins"]
        self.users = self.db["users"]
        self.required_channels = self.db["required_channels"]
        self.join_requests = self.db["join_requests"]
        self.movies = self.db["movies"]
        self.serials = self.db["serials"]
        self.serial_episodes = self.db["serial_episodes"]
        self.requests_log = self.db["requests_log"]
        self.content_requests = self.db["content_requests"]
        self.favorites = self.db["favorites"]
        self.notification_log = self.db["notification_log"]
        self.settings = self.db["settings"]
        self.payment_requests = self.db["payment_requests"]
        self.reactions = self.db["reactions"]
        self.notification_settings = self.db["notification_settings"]
        self.ad_channels = self.db["ad_channels"]
        self.ads = self.db["ads"]
        self.pro_history = self.db["pro_history"]

        self.init_indexes()

    @staticmethod
    def _doc_without_object_id(doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if not doc:
            return None
        result = dict(doc)
        if "_id" in result:
            result["id"] = str(result["_id"])
            del result["_id"]
        return result

    @staticmethod
    def _to_object_id(value: str) -> ObjectId | None:
        try:
            return ObjectId(value)
        except (InvalidId, TypeError):
            return None

    @staticmethod
    def _normalize_lookup(value: str) -> str:
        cleaned = re.sub(r"[^\w\s]+", " ", value.lower(), flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _normalize_quality(value: str | None) -> str:
        return re.sub(r"\s+", "", (value or "").strip().lower())

    @staticmethod
    def _parse_iso_dt(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    @classmethod
    def _title_matches_query(cls, title: str, description: str, query: str) -> bool:
        query_norm = cls._normalize_lookup(query)
        if not query_norm:
            return True
        haystack = cls._normalize_lookup(f"{title} {description}")
        return query_norm in haystack

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def init_indexes(self) -> None:
        self.admins.create_index("tg_id", unique=True)
        self.users.create_index("tg_id", unique=True)
        self.required_channels.create_index("channel_ref", unique=True)
        self.required_channels.create_index([("is_active", ASCENDING), ("created_at", DESCENDING)])
        self.join_requests.create_index([("user_tg_id", ASCENDING), ("channel_ref", ASCENDING)], unique=True)
        self.join_requests.create_index([("created_at", DESCENDING)])
        self.movies.create_index("code", unique=True)
        self.movies.create_index([("created_at", DESCENDING)])
        self.movies.create_index([("year", DESCENDING), ("quality_norm", ASCENDING)])
        self.movies.create_index("genres")
        self.serials.create_index("code", unique=True)
        self.serials.create_index([("created_at", DESCENDING)])
        self.serials.create_index([("year", DESCENDING), ("quality_norm", ASCENDING)])
        self.serials.create_index("genres")
        self.serial_episodes.create_index(
            [("serial_id", ASCENDING), ("episode_number", ASCENDING)],
            unique=True,
        )
        self.requests_log.create_index([("created_at", DESCENDING)])
        self.content_requests.create_index(
            [("user_tg_id", ASCENDING), ("request_type", ASCENDING), ("normalized_query", ASCENDING)],
            unique=True,
        )
        self.content_requests.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
        self.content_requests.create_index([("request_type", ASCENDING), ("normalized_query", ASCENDING)])
        self.favorites.create_index(
            [("user_tg_id", ASCENDING), ("content_type", ASCENDING), ("content_ref", ASCENDING)],
            unique=True,
        )
        self.favorites.create_index([("user_tg_id", ASCENDING), ("created_at", DESCENDING)])
        self.notification_log.create_index([("created_at", DESCENDING)])
        self.payment_requests.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
        self.payment_requests.create_index([("user_tg_id", ASCENDING), ("created_at", DESCENDING)])
        self.reactions.create_index(
            [("user_tg_id", ASCENDING), ("content_type", ASCENDING), ("content_ref", ASCENDING)],
            unique=True,
        )
        self.reactions.create_index([("content_type", ASCENDING), ("content_ref", ASCENDING)])
        self.notification_settings.create_index("user_tg_id", unique=True)
        self.ad_channels.create_index("channel_ref", unique=True)
        self.ad_channels.create_index([("created_at", DESCENDING)])
        self.ads.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
        self.ads.create_index([("user_tg_id", ASCENDING), ("created_at", DESCENDING)])
        self.pro_history.create_index([("user_tg_id", ASCENDING), ("created_at", DESCENDING)])

    def seed_admins(self, admin_ids: Iterable[int]) -> None:
        now = utc_now_iso()
        for admin_id in admin_ids:
            self.admins.update_one(
                {"tg_id": admin_id},
                {"$setOnInsert": {"tg_id": admin_id, "added_at": now}},
                upsert=True,
            )

    def is_admin(self, tg_id: int) -> bool:
        return self.admins.find_one({"tg_id": tg_id}, {"_id": 1}) is not None

    def add_admin(self, tg_id: int) -> bool:
        now = utc_now_iso()
        try:
            self.admins.insert_one({"tg_id": tg_id, "added_at": now})
            return True
        except DuplicateKeyError:
            return False

    def list_admin_ids(self) -> list[int]:
        result: list[int] = []
        for doc in self.admins.find({}, {"tg_id": 1}):
            if not doc:
                continue
            tg_id = doc.get("tg_id")
            if isinstance(tg_id, int):
                result.append(tg_id)
        return result

    def get_bot_settings(self) -> dict[str, Any]:
        doc = self.settings.find_one({"_id": "config"}) or {}
        return {
            "pro_price_text": str(doc.get("pro_price_text") or PRO_PRICE_TEXT_DEFAULT),
            "pro_duration_days": max(1, int(doc.get("pro_duration_days") or PRO_DURATION_DAYS_DEFAULT)),
        }

    def set_pro_price_text(self, price_text: str) -> None:
        self.settings.update_one(
            {"_id": "config"},
            {
                "$set": {
                    "pro_price_text": price_text.strip()[:120],
                    "updated_at": utc_now_iso(),
                }
            },
            upsert=True,
        )

    def set_pro_duration_days(self, days: int) -> None:
        self.settings.update_one(
            {"_id": "config"},
            {
                "$set": {
                    "pro_duration_days": max(1, int(days)),
                    "updated_at": utc_now_iso(),
                }
            },
            upsert=True,
        )

    def add_user(self, tg_id: int, full_name: str) -> None:
        now = utc_now_iso()
        self.users.update_one(
            {"tg_id": tg_id},
            {
                "$set": {"full_name": full_name},
                "$setOnInsert": {
                    "joined_at": now,
                    "is_pro": False,
                    "pro_until": None,
                    "pro_status": "free",
                    "pro_note": "",
                },
            },
            upsert=True,
        )
        self.notification_settings.update_one(
            {"user_tg_id": tg_id},
            {
                "$setOnInsert": {
                    "user_tg_id": tg_id,
                    "new_content": True,
                    "pro_updates": True,
                    "ads_updates": True,
                    "created_at": now,
                }
            },
            upsert=True,
        )

    def get_user(self, tg_id: int) -> dict[str, Any] | None:
        doc = self.users.find_one(
            {"tg_id": tg_id},
            {
                "tg_id": 1,
                "full_name": 1,
                "is_pro": 1,
                "pro_until": 1,
                "pro_status": 1,
                "pro_note": 1,
                "pro_given_at": 1,
                "pro_given_by": 1,
            },
        )
        return self._doc_without_object_id(doc)

    def is_pro_active(self, tg_id: int) -> bool:
        if self.is_admin(tg_id):
            return True
        user = self.get_user(tg_id)
        if not user:
            return False
        pro_until = str(user.get("pro_until") or "").strip()
        return bool(user.get("is_pro")) and bool(pro_until) and pro_until > utc_now_iso()

    def get_pro_status(self, tg_id: int) -> dict[str, Any]:
        user = self.get_user(tg_id) or {"tg_id": tg_id}
        settings = self.get_bot_settings()
        if self.is_admin(tg_id):
            return {
                "is_pro": True,
                "pro_until": "Cheksiz (Admin)",
                "pro_status": "active",
                "pro_note": "Admin uchun cheksiz PRO",
                "pro_price_text": settings["pro_price_text"],
                "pro_duration_days": settings["pro_duration_days"],
            }
        is_active = self.is_pro_active(tg_id)
        current_status = "active" if is_active else ("expired" if user.get("is_pro") else str(user.get("pro_status") or "free"))
        return {
            "is_pro": is_active,
            "pro_until": str(user.get("pro_until") or ""),
            "pro_status": current_status,
            "pro_note": str(user.get("pro_note") or ""),
            "pro_price_text": settings["pro_price_text"],
            "pro_duration_days": settings["pro_duration_days"],
        }

    def set_pro_state(
        self,
        tg_id: int,
        enabled: bool,
        *,
        admin_id: int | None = None,
        note: str = "",
        duration_days: int | None = None,
    ) -> None:
        now = utc_now_iso()
        if enabled:
            settings = self.get_bot_settings()
            days = max(1, int(duration_days or settings["pro_duration_days"]))
            pro_until = (datetime.now(UTC) + timedelta(days=days)).isoformat()
            update = {
                "is_pro": True,
                "pro_until": pro_until,
                "pro_status": "active",
                "pro_note": note.strip()[:200],
                "pro_given_at": now,
                "pro_given_by": admin_id,
            }
            history_action = "enabled"
        else:
            update = {
                "is_pro": False,
                "pro_until": None,
                "pro_status": "disabled",
                "pro_note": note.strip()[:200],
                "pro_given_at": now,
                "pro_given_by": admin_id,
            }
            history_action = "disabled"
        self.users.update_one(
            {"tg_id": tg_id},
            {
                "$set": update,
                "$setOnInsert": {"joined_at": now, "full_name": ""},
            },
            upsert=True,
        )
        self.pro_history.insert_one(
            {
                "user_tg_id": tg_id,
                "action": history_action,
                "admin_id": admin_id,
                "note": note.strip()[:200],
                "created_at": now,
                "pro_until": update.get("pro_until"),
            }
        )

    def list_active_pro_user_ids(self) -> list[int]:
        now_iso = utc_now_iso()
        user_ids: list[int] = self.list_admin_ids()
        for doc in self.users.find(
            {"is_pro": True, "pro_until": {"$gt": now_iso}},
            {"tg_id": 1},
        ):
            if not doc:
                continue
            tg_id = doc.get("tg_id")
            if isinstance(tg_id, int) and tg_id not in user_ids:
                user_ids.append(tg_id)
        return user_ids

    def count_active_pro_users(self) -> int:
        return len(self.list_active_pro_user_ids())

    def get_notification_settings(self, user_tg_id: int) -> dict[str, Any]:
        now = utc_now_iso()
        self.notification_settings.update_one(
            {"user_tg_id": user_tg_id},
            {
                "$setOnInsert": {
                    "user_tg_id": user_tg_id,
                    "new_content": True,
                    "pro_updates": True,
                    "ads_updates": True,
                    "created_at": now,
                }
            },
            upsert=True,
        )
        doc = self.notification_settings.find_one({"user_tg_id": user_tg_id})
        return self._doc_without_object_id(doc) or {
            "user_tg_id": user_tg_id,
            "new_content": True,
            "pro_updates": True,
            "ads_updates": True,
        }

    def toggle_notification_setting(self, user_tg_id: int, key: str) -> dict[str, Any]:
        if key not in {"new_content", "pro_updates", "ads_updates"}:
            return self.get_notification_settings(user_tg_id)
        current = self.get_notification_settings(user_tg_id)
        next_value = not bool(current.get(key))
        self.notification_settings.update_one(
            {"user_tg_id": user_tg_id},
            {"$set": {key: next_value, "updated_at": utc_now_iso()}},
            upsert=True,
        )
        return self.get_notification_settings(user_tg_id)

    def add_required_channel(
        self,
        channel_ref: str,
        title: str | None = None,
        join_link: str | None = None,
    ) -> bool:
        now = utc_now_iso()
        doc = {
            "channel_ref": channel_ref,
            "title": title,
            "join_link": join_link,
            "is_active": True,
            "created_at": now,
        }
        try:
            self.required_channels.insert_one(doc)
            return True
        except DuplicateKeyError:
            return False

    def remove_required_channel(self, channel_ref: str) -> bool:
        result = self.required_channels.delete_one({"channel_ref": channel_ref})
        return result.deleted_count > 0

    def get_required_channels(self) -> list[dict[str, Any]]:
        cursor = self.required_channels.find(
            {"is_active": True},
            {"channel_ref": 1, "title": 1, "join_link": 1},
        ).sort("created_at", DESCENDING)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def mark_join_request(self, user_tg_id: int, channel_ref: str) -> None:
        channel_ref = channel_ref.strip()
        if not channel_ref:
            return
        now = utc_now_iso()
        self.join_requests.update_one(
            {"user_tg_id": user_tg_id, "channel_ref": channel_ref},
            {
                "$set": {"updated_at": now},
                "$setOnInsert": {
                    "user_tg_id": user_tg_id,
                    "channel_ref": channel_ref,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    def get_join_request_refs(self, user_tg_id: int) -> set[str]:
        cursor = self.join_requests.find(
            {"user_tg_id": user_tg_id},
            {"channel_ref": 1},
        )
        return {
            str(doc["channel_ref"]).strip()
            for doc in cursor
            if doc and str(doc.get("channel_ref", "")).strip()
        }

    def add_movie(self, movie: Movie) -> bool:
        now = utc_now_iso()
        genres = sorted({g.strip().lower() for g in (movie.genres or []) if g and g.strip()})
        doc = {
            "code": movie.code,
            "title": movie.title,
            "description": movie.description,
            "media_type": movie.media_type,
            "file_id": movie.file_id,
            "year": movie.year,
            "quality": movie.quality.strip(),
            "quality_norm": self._normalize_quality(movie.quality),
            "genres": genres,
            "preview_media_type": movie.preview_media_type.strip(),
            "preview_file_id": movie.preview_file_id.strip(),
            "downloads": 0,
            "views": 0,
            "title_norm": self._normalize_lookup(movie.title),
            "created_at": now,
        }
        try:
            self.movies.insert_one(doc)
            return True
        except DuplicateKeyError:
            return False

    def delete_movie(self, code: str) -> bool:
        result = self.movies.delete_one({"code": code})
        return result.deleted_count > 0

    def get_movie(self, code: str) -> dict[str, Any] | None:
        doc = self.movies.find_one(
            {"code": code},
            {
                "code": 1,
                "title": 1,
                "description": 1,
                "media_type": 1,
                "file_id": 1,
                "year": 1,
                "quality": 1,
                "genres": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "downloads": 1,
                "views": 1,
            },
        )
        return self._doc_without_object_id(doc)

    def get_movie_by_id(self, movie_id: str) -> dict[str, Any] | None:
        movie_object_id = self._to_object_id(movie_id)
        if not movie_object_id:
            return None
        doc = self.movies.find_one(
            {"_id": movie_object_id},
            {
                "code": 1,
                "title": 1,
                "description": 1,
                "media_type": 1,
                "file_id": 1,
                "year": 1,
                "quality": 1,
                "genres": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "downloads": 1,
                "views": 1,
            },
        )
        return self._doc_without_object_id(doc)

    def update_movie(
        self,
        code: str,
        title: str,
        description: str,
        media_type: str,
        file_id: str,
        year: int | None = None,
        quality: str = "",
        genres: list[str] | None = None,
        preview_media_type: str = "",
        preview_file_id: str = "",
    ) -> bool:
        cleaned_quality = quality.strip()
        cleaned_genres = sorted({g.strip().lower() for g in (genres or []) if g and g.strip()})
        result = self.movies.update_one(
            {"code": code},
            {
                "$set": {
                    "title": title,
                    "description": description,
                    "media_type": media_type,
                    "file_id": file_id,
                    "year": year,
                    "quality": cleaned_quality,
                    "quality_norm": self._normalize_quality(cleaned_quality),
                    "genres": cleaned_genres,
                    "preview_media_type": preview_media_type.strip(),
                    "preview_file_id": preview_file_id.strip(),
                    "title_norm": self._normalize_lookup(title),
                    "updated_at": utc_now_iso(),
                }
            },
        )
        return result.matched_count > 0

    def list_movies(self, limit: int | None = 50) -> list[dict[str, Any]]:
        cursor = self.movies.find(
            {},
            {"code": 1, "title": 1, "year": 1, "quality": 1, "created_at": 1},
        ).sort("created_at", DESCENDING)
        if limit and limit > 0:
            cursor = cursor.limit(limit)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def list_serials(self, limit: int | None = None) -> list[dict[str, Any]]:
        cursor = self.serials.find(
            {},
            {"code": 1, "title": 1, "year": 1, "quality": 1, "created_at": 1},
        ).sort("created_at", DESCENDING)
        if limit and limit > 0:
            cursor = cursor.limit(limit)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def increment_movie_downloads(self, movie_id: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        movie_object_id = self._to_object_id(movie_id)
        if not movie_object_id:
            return
        self.movies.update_one(
            {"_id": movie_object_id},
            {
                "$inc": {"downloads": amount},
                "$set": {"updated_at": utc_now_iso()},
            },
        )

    def increment_movie_views(self, movie_id: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        movie_object_id = self._to_object_id(movie_id)
        if not movie_object_id:
            return
        self.movies.update_one(
            {"_id": movie_object_id},
            {
                "$inc": {"views": amount},
                "$set": {"updated_at": utc_now_iso()},
            },
        )

    def increment_serial_downloads(self, serial_id: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        serial_object_id = self._to_object_id(serial_id)
        if not serial_object_id:
            return
        self.serials.update_one(
            {"_id": serial_object_id},
            {
                "$inc": {"downloads": amount},
                "$set": {"updated_at": utc_now_iso()},
            },
        )

    def increment_serial_views(self, serial_id: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        serial_object_id = self._to_object_id(serial_id)
        if not serial_object_id:
            return
        self.serials.update_one(
            {"_id": serial_object_id},
            {
                "$inc": {"views": amount},
                "$set": {"updated_at": utc_now_iso()},
            },
        )

    def add_serial(
        self,
        code: str,
        title: str,
        description: str,
        year: int | None = None,
        quality: str = "",
        genres: list[str] | None = None,
    ) -> str | None:
        now = utc_now_iso()
        cleaned_quality = quality.strip()
        cleaned_genres = sorted({g.strip().lower() for g in (genres or []) if g and g.strip()})
        doc = {
            "code": code,
            "title": title,
            "description": description,
            "year": year,
            "quality": cleaned_quality,
            "quality_norm": self._normalize_quality(cleaned_quality),
            "genres": cleaned_genres,
            "title_norm": self._normalize_lookup(title),
            "preview_media_type": "",
            "preview_file_id": "",
            "preview_photo_file_id": "",
            "downloads": 0,
            "views": 0,
            "created_at": now,
        }
        try:
            result = self.serials.insert_one(doc)
            return str(result.inserted_id)
        except DuplicateKeyError:
            return None

    def delete_serial(self, serial_id: str) -> bool:
        serial_object_id = self._to_object_id(serial_id)
        if not serial_object_id:
            return False
        self.serial_episodes.delete_many({"serial_id": serial_id})
        result = self.serials.delete_one({"_id": serial_object_id})
        return result.deleted_count > 0

    def add_serial_episode(
        self,
        serial_id: str,
        episode_number: int,
        media_type: str,
        file_id: str,
    ) -> bool:
        now = utc_now_iso()
        doc = {
            "serial_id": serial_id,
            "episode_number": episode_number,
            "media_type": media_type,
            "file_id": file_id,
            "created_at": now,
        }
        try:
            self.serial_episodes.insert_one(doc)
            return True
        except DuplicateKeyError:
            return False

    def get_serial(self, serial_id: str) -> dict[str, Any] | None:
        serial_object_id = self._to_object_id(serial_id)
        if not serial_object_id:
            return None
        doc = self.serials.find_one(
            {"_id": serial_object_id},
            {
                "code": 1,
                "title": 1,
                "description": 1,
                "year": 1,
                "quality": 1,
                "genres": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "preview_photo_file_id": 1,
                "downloads": 1,
                "views": 1,
            },
        )
        return self._doc_without_object_id(doc)

    def get_serial_by_code(self, code: str) -> dict[str, Any] | None:
        doc = self.serials.find_one(
            {"code": code},
            {
                "code": 1,
                "title": 1,
                "description": 1,
                "year": 1,
                "quality": 1,
                "genres": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "preview_photo_file_id": 1,
                "downloads": 1,
                "views": 1,
            },
        )
        return self._doc_without_object_id(doc)

    def update_serial_preview(
        self,
        serial_id: str,
        media_type: str,
        file_id: str,
        preview_photo_file_id: str = "",
    ) -> bool:
        serial_object_id = self._to_object_id(serial_id)
        if not serial_object_id:
            return False
        result = self.serials.update_one(
            {"_id": serial_object_id},
            {
                "$set": {
                    "preview_media_type": media_type.strip(),
                    "preview_file_id": file_id.strip(),
                    "preview_photo_file_id": preview_photo_file_id.strip(),
                    "updated_at": utc_now_iso(),
                }
            },
        )
        return result.matched_count > 0

    def list_serial_episodes(self, serial_id: str) -> list[dict[str, Any]]:
        cursor = self.serial_episodes.find(
            {"serial_id": serial_id},
            {"episode_number": 1},
        ).sort("episode_number", ASCENDING)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def get_next_serial_episode_number(self, serial_id: str) -> int:
        doc = self.serial_episodes.find_one(
            {"serial_id": serial_id},
            {"episode_number": 1},
            sort=[("episode_number", DESCENDING)],
        )
        if not doc:
            return 1
        return int(doc.get("episode_number", 0)) + 1

    def get_serial_episode(self, serial_id: str, episode_number: int) -> dict[str, Any] | None:
        doc = self.serial_episodes.find_one(
            {"serial_id": serial_id, "episode_number": episode_number},
            {"media_type": 1, "file_id": 1},
        )
        return self._doc_without_object_id(doc)

    def get_serial_inline_media_preview(self, serial_id: str) -> tuple[str, str] | None:
        # Fallback for inline cards: use first episode media when explicit serial preview is missing.
        doc = self.serial_episodes.find_one(
            {"serial_id": serial_id},
            {"media_type": 1, "file_id": 1},
            sort=[("episode_number", ASCENDING)],
        )
        if not doc:
            return None
        media_type = str(doc.get("media_type") or "")
        file_id = str(doc.get("file_id") or "")
        if not file_id:
            return None
        if media_type == "photo":
            return "photo", file_id
        if media_type == "video":
            return "video", file_id
        return None

    def code_exists(self, code: str) -> bool:
        if self.movies.find_one({"code": code}, {"_id": 1}):
            return True
        return self.serials.find_one({"code": code}, {"_id": 1}) is not None

    def get_all_codes(self) -> set[str]:
        codes: set[str] = set()
        for doc in self.movies.find({}, {"code": 1}):
            code = str(doc.get("code", "")).strip() if doc else ""
            if code:
                codes.add(code)
        for doc in self.serials.find({}, {"code": 1}):
            code = str(doc.get("code", "")).strip() if doc else ""
            if code:
                codes.add(code)
        return codes

    def list_user_ids(self) -> list[int]:
        user_ids: list[int] = []
        for doc in self.users.find({}, {"tg_id": 1}):
            if not doc:
                continue
            tg_id = doc.get("tg_id")
            if isinstance(tg_id, int):
                user_ids.append(tg_id)
        return user_ids

    def get_user_reaction(self, user_tg_id: int, content_type: str, content_ref: str) -> str:
        doc = self.reactions.find_one(
            {
                "user_tg_id": user_tg_id,
                "content_type": content_type,
                "content_ref": content_ref,
            },
            {"reaction": 1},
        )
        return str((doc or {}).get("reaction") or "")

    def get_reaction_summary(self, content_type: str, content_ref: str) -> dict[str, Any]:
        likes = self.reactions.count_documents(
            {"content_type": content_type, "content_ref": content_ref, "reaction": "like"}
        )
        dislikes = self.reactions.count_documents(
            {"content_type": content_type, "content_ref": content_ref, "reaction": "dislike"}
        )
        total = likes + dislikes
        rating = round((likes / total) * 5, 1) if total else 0.0
        return {
            "likes": int(likes),
            "dislikes": int(dislikes),
            "total": int(total),
            "rating": rating,
        }

    def set_reaction(
        self,
        user_tg_id: int,
        content_type: str,
        content_ref: str,
        reaction: str,
    ) -> dict[str, Any]:
        if content_type not in {"movie", "serial"} or reaction not in {"like", "dislike"}:
            return self.get_reaction_summary(content_type, content_ref)
        criteria = {
            "user_tg_id": user_tg_id,
            "content_type": content_type,
            "content_ref": content_ref,
        }
        existing = self.reactions.find_one(criteria, {"reaction": 1})
        now = utc_now_iso()
        if existing and str(existing.get("reaction") or "") == reaction:
            self.reactions.delete_one(criteria)
        elif existing:
            self.reactions.update_one(criteria, {"$set": {"reaction": reaction, "updated_at": now}})
        else:
            self.reactions.insert_one({**criteria, "reaction": reaction, "created_at": now, "updated_at": now})
        return self.get_reaction_summary(content_type, content_ref)

    def _reaction_map(self) -> dict[tuple[str, str], dict[str, int]]:
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "content_type": "$content_type",
                        "content_ref": "$content_ref",
                    },
                    "likes": {
                        "$sum": {
                            "$cond": [{"$eq": ["$reaction", "like"]}, 1, 0]
                        }
                    },
                    "dislikes": {
                        "$sum": {
                            "$cond": [{"$eq": ["$reaction", "dislike"]}, 1, 0]
                        }
                    },
                }
            }
        ]
        result: dict[tuple[str, str], dict[str, int]] = {}
        for row in self.reactions.aggregate(pipeline):
            key_data = row.get("_id") or {}
            content_type = str(key_data.get("content_type") or "")
            content_ref = str(key_data.get("content_ref") or "")
            if not content_type or not content_ref:
                continue
            result[(content_type, content_ref)] = {
                "likes": int(row.get("likes") or 0),
                "dislikes": int(row.get("dislikes") or 0),
            }
        return result

    def _content_feed_rows(self) -> list[dict[str, Any]]:
        projection = {
            "code": 1,
            "title": 1,
            "description": 1,
            "year": 1,
            "quality": 1,
            "downloads": 1,
            "views": 1,
            "created_at": 1,
        }
        rows: list[dict[str, Any]] = []
        for doc in self.movies.find({}, projection):
            normalized = self._doc_without_object_id(doc)
            if not normalized:
                continue
            normalized["content_type"] = "movie"
            rows.append(normalized)
        for doc in self.serials.find({}, projection):
            normalized = self._doc_without_object_id(doc)
            if not normalized:
                continue
            normalized["content_type"] = "serial"
            rows.append(normalized)
        return rows

    def list_top_viewed_content(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._content_feed_rows()
        rows.sort(
            key=lambda item: (
                int(item.get("views") or 0),
                int(item.get("downloads") or 0),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )
        return rows[: max(1, limit)]

    def list_trending_content(self, limit: int = 10) -> list[dict[str, Any]]:
        reaction_map = self._reaction_map()
        scored: list[dict[str, Any]] = []
        for item in self._content_feed_rows():
            content_type = str(item.get("content_type") or "")
            content_ref = str(item.get("id") or "")
            reaction = reaction_map.get((content_type, content_ref), {"likes": 0, "dislikes": 0})
            created_at = self._parse_iso_dt(item.get("created_at"))
            age_days = (datetime.now(UTC) - created_at).days if created_at else 999
            freshness_bonus = max(0, 30 - max(0, age_days))
            score = (
                int(item.get("views") or 0) * 5
                + int(item.get("downloads") or 0) * 3
                + int(reaction.get("likes") or 0) * 4
                - int(reaction.get("dislikes") or 0) * 2
                + freshness_bonus
            )
            item_copy = dict(item)
            item_copy["_trend_score"] = score
            scored.append(item_copy)
        scored.sort(
            key=lambda item: (
                int(item.get("_trend_score") or 0),
                int(item.get("views") or 0),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )
        return [{k: v for k, v in row.items() if k != "_trend_score"} for row in scored[: max(1, limit)]]

    def create_payment_request(
        self,
        user_tg_id: int,
        payment_code: str,
        proof_media_type: str = "",
        proof_file_id: str = "",
        comment: str = "",
    ) -> str:
        now = utc_now_iso()
        doc = {
            "user_tg_id": user_tg_id,
            "payment_code": payment_code.strip()[:80],
            "proof_media_type": proof_media_type.strip(),
            "proof_file_id": proof_file_id.strip(),
            "comment": comment.strip()[:500],
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        result = self.payment_requests.insert_one(doc)
        return str(result.inserted_id)

    def get_payment_request(self, request_id: str) -> dict[str, Any] | None:
        object_id = self._to_object_id(request_id)
        if not object_id:
            return None
        doc = self.payment_requests.find_one({"_id": object_id})
        return self._doc_without_object_id(doc)

    def list_pending_payment_requests(self, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.payment_requests.find({"status": "pending"}).sort("created_at", DESCENDING)
        if limit > 0:
            cursor = cursor.limit(limit)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def update_payment_request_status(
        self,
        request_id: str,
        status: str,
        *,
        reviewed_by: int | None = None,
        review_note: str = "",
    ) -> bool:
        object_id = self._to_object_id(request_id)
        if not object_id:
            return False
        result = self.payment_requests.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "status": status,
                    "reviewed_by": reviewed_by,
                    "review_note": review_note.strip()[:300],
                    "reviewed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                }
            },
        )
        return result.matched_count > 0

    def add_ad_channel(self, channel_ref: str, title: str | None = None) -> bool:
        now = utc_now_iso()
        try:
            self.ad_channels.insert_one(
                {
                    "channel_ref": channel_ref.strip(),
                    "title": (title or channel_ref).strip(),
                    "created_at": now,
                    "is_active": True,
                }
            )
            return True
        except DuplicateKeyError:
            return False

    def list_ad_channels(self) -> list[dict[str, Any]]:
        cursor = self.ad_channels.find({"is_active": True}).sort("created_at", DESCENDING)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def get_ad_channel(self, channel_id: str) -> dict[str, Any] | None:
        object_id = self._to_object_id(channel_id)
        if not object_id:
            return None
        doc = self.ad_channels.find_one({"_id": object_id})
        return self._doc_without_object_id(doc)

    def remove_ad_channel(self, channel_id: str) -> bool:
        object_id = self._to_object_id(channel_id)
        if not object_id:
            return False
        result = self.ad_channels.delete_one({"_id": object_id})
        return result.deleted_count > 0

    def create_ad(
        self,
        user_tg_id: int,
        title: str,
        description: str,
        *,
        photo_file_id: str = "",
        button_text: str = "",
        button_url: str = "",
    ) -> str:
        now = utc_now_iso()
        result = self.ads.insert_one(
            {
                "user_tg_id": user_tg_id,
                "title": title.strip()[:120],
                "description": description.strip()[:850],
                "photo_file_id": photo_file_id.strip(),
                "button_text": button_text.strip()[:60],
                "button_url": button_url.strip()[:300],
                "status": "pending",
                "created_at": now,
                "updated_at": now,
            }
        )
        return str(result.inserted_id)

    def get_ad(self, ad_id: str) -> dict[str, Any] | None:
        object_id = self._to_object_id(ad_id)
        if not object_id:
            return None
        doc = self.ads.find_one({"_id": object_id})
        return self._doc_without_object_id(doc)

    def list_user_ads(self, user_tg_id: int, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.ads.find({"user_tg_id": user_tg_id}).sort("created_at", DESCENDING)
        if limit > 0:
            cursor = cursor.limit(limit)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def list_pending_ads(self, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.ads.find({"status": "pending"}).sort("created_at", DESCENDING)
        if limit > 0:
            cursor = cursor.limit(limit)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def update_ad_status(
        self,
        ad_id: str,
        status: str,
        *,
        reviewed_by: int | None = None,
        channel_ref: str = "",
        review_note: str = "",
    ) -> bool:
        object_id = self._to_object_id(ad_id)
        if not object_id:
            return False
        result = self.ads.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "status": status,
                    "reviewed_by": reviewed_by,
                    "review_note": review_note.strip()[:300],
                    "channel_ref": channel_ref.strip(),
                    "reviewed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                }
            },
        )
        return result.matched_count > 0

    def search_content(
        self,
        query: str,
        limit: int = 20,
        year: int | None = None,
        quality: str | None = None,
        genres: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_norm = self._normalize_lookup(query)
        quality_norm = self._normalize_quality(quality)
        genres_norm = sorted({g.strip().lower() for g in (genres or []) if g and g.strip()})

        mongo_filter: dict[str, Any] = {}
        if year is not None:
            mongo_filter["year"] = year
        if quality_norm:
            mongo_filter["quality_norm"] = quality_norm
        if genres_norm:
            mongo_filter["genres"] = {"$all": genres_norm}

        movies_cursor = self.movies.find(
            mongo_filter,
            {
                "code": 1,
                "title": 1,
                "description": 1,
                "year": 1,
                "quality": 1,
                "genres": 1,
                "media_type": 1,
                "file_id": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "preview_photo_file_id": 1,
                "downloads": 1,
                "views": 1,
                "created_at": 1,
            },
        ).sort("created_at", DESCENDING).limit(300)

        serials_cursor = self.serials.find(
            mongo_filter,
            {
                "code": 1,
                "title": 1,
                "description": 1,
                "year": 1,
                "quality": 1,
                "genres": 1,
                "media_type": 1,
                "file_id": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "preview_photo_file_id": 1,
                "downloads": 1,
                "views": 1,
                "created_at": 1,
            },
        ).sort("created_at", DESCENDING).limit(300)

        results: list[dict[str, Any]] = []

        def score_for_item(item_code: str, item_title: str, item_description: str) -> int:
            if not query_norm:
                return 1
            title_norm = self._normalize_lookup(item_title)
            desc_norm = self._normalize_lookup(item_description)
            code_norm = self._normalize_lookup(item_code)
            if query_norm == code_norm:
                return 16
            if code_norm.startswith(query_norm):
                return 14
            if query_norm in code_norm:
                return 12
            if query_norm == title_norm:
                return 11
            if title_norm.startswith(query_norm):
                return 10
            if query_norm in title_norm:
                return 9
            if query_norm in desc_norm:
                return 6

            # Typo-tolerant scoring for near matches.
            title_ratio = self._similarity(query_norm, title_norm)
            code_ratio = self._similarity(query_norm, code_norm)
            token_ratios = [self._similarity(query_norm, token) for token in title_norm.split() if token]
            best_ratio = max([title_ratio, code_ratio, *token_ratios], default=0.0)
            if best_ratio >= 0.90:
                return 8
            if best_ratio >= 0.78:
                return 7
            if best_ratio >= 0.65:
                return 5
            if best_ratio >= 0.52:
                return 3
            return 0

        for doc in movies_cursor:
            if not doc:
                continue
            code = str(doc.get("code") or "").strip()
            title = str(doc.get("title") or "").strip()
            description = str(doc.get("description") or "").strip()
            score = score_for_item(code, title, description)
            if query_norm and score == 0:
                continue
            normalized = self._doc_without_object_id(doc)
            if not normalized:
                continue
            normalized["content_type"] = "movie"
            normalized["_score"] = score
            results.append(normalized)

        for doc in serials_cursor:
            if not doc:
                continue
            code = str(doc.get("code") or "").strip()
            title = str(doc.get("title") or "").strip()
            description = str(doc.get("description") or "").strip()
            score = score_for_item(code, title, description)
            if query_norm and score == 0:
                continue
            normalized = self._doc_without_object_id(doc)
            if not normalized:
                continue
            normalized["content_type"] = "serial"
            normalized["_score"] = score
            results.append(normalized)

        # Keep the highest-scored and newest content.
        results = sorted(
            results,
            key=lambda item: (
                int(item.get("_score", 0)),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )

        trimmed: list[dict[str, Any]] = []
        for row in results[: max(1, limit)]:
            data = dict(row)
            data.pop("_score", None)
            trimmed.append(data)
        return trimmed

    def add_or_increment_content_request(
        self,
        user_tg_id: int,
        query_text: str,
        request_type: str,
    ) -> tuple[bool, int]:
        normalized_query = self._normalize_lookup(query_text)
        if not normalized_query:
            raise ValueError("Empty query")
        if request_type not in {"code", "search"}:
            raise ValueError("Invalid request type")

        now = utc_now_iso()
        criteria = {
            "user_tg_id": user_tg_id,
            "request_type": request_type,
            "normalized_query": normalized_query,
        }
        existing = self.content_requests.find_one(criteria, {"_id": 1, "request_count": 1})
        self.content_requests.update_one(
            criteria,
            {
                "$set": {
                    "query_text": query_text.strip()[:120],
                    "status": "open",
                    "updated_at": now,
                    "fulfilled_at": None,
                    "fulfilled_content_type": None,
                    "fulfilled_content_ref": None,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
                "$inc": {"request_count": 1},
            },
            upsert=True,
        )
        latest = self.content_requests.find_one(criteria, {"request_count": 1})
        count = int((latest or {}).get("request_count", 1))
        return existing is None, count

    def list_open_request_topics(self, limit: int = 30) -> list[dict[str, Any]]:
        pipeline = [
            {"$match": {"status": "open"}},
            {
                "$group": {
                    "_id": {
                        "request_type": "$request_type",
                        "normalized_query": "$normalized_query",
                    },
                    "query_text": {"$first": "$query_text"},
                    "total_requests": {"$sum": "$request_count"},
                    "users": {"$addToSet": "$user_tg_id"},
                    "last_updated": {"$max": "$updated_at"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "request_type": "$_id.request_type",
                    "normalized_query": "$_id.normalized_query",
                    "query_text": 1,
                    "total_requests": 1,
                    "users_count": {"$size": "$users"},
                    "last_updated": 1,
                }
            },
            {"$sort": {"total_requests": -1, "users_count": -1, "last_updated": -1}},
            {"$limit": max(1, limit)},
        ]
        return list(self.content_requests.aggregate(pipeline))

    def list_recent_fulfilled_topics(self, limit: int = 10) -> list[dict[str, Any]]:
        pipeline = [
            {"$match": {"status": "fulfilled"}},
            {
                "$group": {
                    "_id": {
                        "request_type": "$request_type",
                        "normalized_query": "$normalized_query",
                    },
                    "query_text": {"$first": "$query_text"},
                    "fulfilled_at": {"$max": "$fulfilled_at"},
                    "fulfilled_content_type": {"$max": "$fulfilled_content_type"},
                    "fulfilled_content_ref": {"$max": "$fulfilled_content_ref"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "request_type": "$_id.request_type",
                    "normalized_query": "$_id.normalized_query",
                    "query_text": 1,
                    "fulfilled_at": 1,
                    "fulfilled_content_type": 1,
                    "fulfilled_content_ref": 1,
                }
            },
            {"$sort": {"fulfilled_at": -1}},
            {"$limit": max(1, limit)},
        ]
        return list(self.content_requests.aggregate(pipeline))

    def get_matching_open_requests(self, code: str, title: str) -> list[dict[str, Any]]:
        code_norm = self._normalize_lookup(code)
        title_norm = self._normalize_lookup(title)

        matched_docs: list[dict[str, Any]] = []

        for doc in self.content_requests.find(
            {
                "status": "open",
                "request_type": "code",
                "normalized_query": code_norm,
            }
        ):
            normalized = self._doc_without_object_id(doc)
            if normalized:
                matched_docs.append(normalized)

        for doc in self.content_requests.find(
            {
                "status": "open",
                "request_type": "search",
            }
        ):
            if not doc:
                continue
            normalized_query = str(doc.get("normalized_query") or "").strip()
            if not normalized_query:
                continue
            if normalized_query in title_norm or title_norm in normalized_query:
                normalized = self._doc_without_object_id(doc)
                if normalized:
                    matched_docs.append(normalized)

        unique_by_id: dict[str, dict[str, Any]] = {}
        for row in matched_docs:
            row_id = str(row.get("id") or "").strip()
            if row_id:
                unique_by_id[row_id] = row
        return list(unique_by_id.values())

    def mark_requests_fulfilled(
        self,
        request_ids: list[str],
        content_type: str,
        content_ref: str,
    ) -> int:
        object_ids: list[ObjectId] = []
        for request_id in request_ids:
            object_id = self._to_object_id(request_id)
            if object_id:
                object_ids.append(object_id)
        if not object_ids:
            return 0

        result = self.content_requests.update_many(
            {"_id": {"$in": object_ids}},
            {
                "$set": {
                    "status": "fulfilled",
                    "fulfilled_at": utc_now_iso(),
                    "fulfilled_content_type": content_type,
                    "fulfilled_content_ref": content_ref,
                    "updated_at": utc_now_iso(),
                }
            },
        )
        return int(result.modified_count)

    def log_notification_attempt(
        self,
        user_tg_id: int,
        request_id: str,
        content_type: str,
        content_ref: str,
        status: str,
        error_message: str = "",
    ) -> None:
        self.notification_log.insert_one(
            {
                "user_tg_id": user_tg_id,
                "request_id": request_id,
                "content_type": content_type,
                "content_ref": content_ref,
                "status": status,
                "error_message": error_message[:300],
                "created_at": utc_now_iso(),
            }
        )

    def add_favorite(self, user_tg_id: int, content_type: str, content_ref: str) -> bool:
        if content_type not in {"movie", "serial"}:
            return False
        now = utc_now_iso()
        try:
            self.favorites.insert_one(
                {
                    "user_tg_id": user_tg_id,
                    "content_type": content_type,
                    "content_ref": content_ref,
                    "created_at": now,
                }
            )
            return True
        except DuplicateKeyError:
            return False

    def remove_favorite(self, user_tg_id: int, content_type: str, content_ref: str) -> bool:
        result = self.favorites.delete_one(
            {
                "user_tg_id": user_tg_id,
                "content_type": content_type,
                "content_ref": content_ref,
            }
        )
        return result.deleted_count > 0

    def is_favorite(self, user_tg_id: int, content_type: str, content_ref: str) -> bool:
        return (
            self.favorites.find_one(
                {
                    "user_tg_id": user_tg_id,
                    "content_type": content_type,
                    "content_ref": content_ref,
                },
                {"_id": 1},
            )
            is not None
        )

    def list_favorites(self, user_tg_id: int, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.favorites.find(
            {"user_tg_id": user_tg_id},
            {"content_type": 1, "content_ref": 1, "created_at": 1},
        ).sort("created_at", DESCENDING)
        if limit > 0:
            cursor = cursor.limit(limit)
        rows = [self._doc_without_object_id(doc) for doc in cursor if doc]
        result: list[dict[str, Any]] = []
        for item in rows:
            if not item:
                continue
            content_type = str(item.get("content_type") or "")
            content_ref = str(item.get("content_ref") or "")
            if content_type == "movie":
                movie = self.get_movie_by_id(content_ref) or self.get_movie(content_ref)
                if not movie:
                    continue
                result.append(
                    {
                        "favorite_id": item["id"],
                        "content_type": "movie",
                        "content_ref": movie["id"],
                        "title": movie.get("title") or "-",
                        "code": movie.get("code") or "",
                        "year": movie.get("year"),
                        "quality": movie.get("quality") or "",
                    }
                )
            elif content_type == "serial":
                serial = self.get_serial(content_ref) or self.get_serial_by_code(content_ref)
                if not serial:
                    continue
                result.append(
                    {
                        "favorite_id": item["id"],
                        "content_type": "serial",
                        "content_ref": serial["id"],
                        "title": serial.get("title") or "-",
                        "code": serial.get("code") or "",
                        "year": serial.get("year"),
                        "quality": serial.get("quality") or "",
                    }
                )
        return result

    def log_request(self, user_tg_id: int, movie_code: str, result: str) -> None:
        now = utc_now_iso()
        self.requests_log.insert_one(
            {
                "user_tg_id": user_tg_id,
                "movie_code": movie_code,
                "result": result,
                "created_at": now,
            }
        )

    def stats(self) -> dict[str, int]:
        users = self.users.count_documents({})
        movies = self.movies.count_documents({})
        serials = self.serials.count_documents({})
        serial_episodes = self.serial_episodes.count_documents({})
        channels = self.required_channels.count_documents({"is_active": True})
        requests = self.requests_log.count_documents({})
        open_content_requests = self.content_requests.count_documents({"status": "open"})
        favorites = self.favorites.count_documents({})
        active_pro_users = self.count_active_pro_users()
        pending_payments = self.payment_requests.count_documents({"status": "pending"})
        pending_ads = self.ads.count_documents({"status": "pending"})
        reactions = self.reactions.count_documents({})
        return {
            "users": users,
            "movies": movies,
            "serials": serials,
            "serial_episodes": serial_episodes,
            "channels": channels,
            "requests": requests,
            "open_content_requests": open_content_requests,
            "favorites": favorites,
            "active_pro_users": active_pro_users,
            "pending_payments": pending_payments,
            "pending_ads": pending_ads,
            "reactions": reactions,
        }


class AddChannelState(StatesGroup):
    waiting_input = State()


class AddMovieState(StatesGroup):
    waiting_code = State()
    waiting_title = State()
    waiting_description = State()
    waiting_metadata = State()
    waiting_media = State()


class AddSerialState(StatesGroup):
    waiting_code = State()
    waiting_title = State()
    waiting_description = State()
    waiting_metadata = State()
    waiting_episode = State()
    waiting_preview_media = State()
    waiting_publish_channel = State()


class DeleteMovieState(StatesGroup):
    waiting_code = State()


class AddAdminState(StatesGroup):
    waiting_tg_id = State()


class EditContentState(StatesGroup):
    waiting_code = State()
    waiting_movie_title = State()
    waiting_movie_description = State()
    waiting_movie_metadata = State()
    waiting_movie_media = State()
    waiting_serial_episode = State()


class BroadcastState(StatesGroup):
    waiting_message = State()
    waiting_button_choice = State()
    waiting_button_text = State()
    waiting_button_url = State()
    waiting_confirm = State()


class SearchState(StatesGroup):
    waiting_query = State()


class ProManageState(StatesGroup):
    waiting_input = State()


class ProPriceState(StatesGroup):
    waiting_price = State()


class ProDurationState(StatesGroup):
    waiting_days = State()


class PaymentState(StatesGroup):
    waiting_proof = State()
    waiting_comment = State()


class AddAdChannelState(StatesGroup):
    waiting_input = State()


class AdCreateState(StatesGroup):
    waiting_photo = State()
    waiting_title = State()
    waiting_description = State()
    waiting_button_choice = State()
    waiting_button_text = State()
    waiting_button_url = State()
    waiting_confirm = State()


def parse_admin_ids(value: str) -> list[int]:
    result: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item.isdigit():
            result.append(int(item))
    return result


BTN_ADMIN_PANEL = "🛠 Admin panel"
BTN_SUBS = "📢 Majburiy obuna"
BTN_ADD_MOVIE = "➕ Kino qo'shish"
BTN_ADD_SERIAL = "📺 Serial qo'shish"
BTN_DEL_MOVIE = "🗑 Kino o'chirish"
BTN_EDIT_CONTENT = "✏️ Kontent tahrirlash"
BTN_RANDOM_CODES = "🎲 Random kod"
BTN_LIST_MOVIES = "📚 Kino va serial ro'yxati"
BTN_STATS = "📊 Statistika"
BTN_ADD_ADMIN = "👤 Admin qo'shish"
BTN_BROADCAST = "📣 Habar yuborish"
BTN_REQUESTS = "📥 So'rovlar"
BTN_BACK = "⬅️ Ortga"
BTN_CANCEL = "❌ Bekor qilish"
BTN_SERIAL_DONE = "✅ Serialni yakunlash"
BTN_SEARCH_NAME = "🔎 Nom bo'yicha qidirish"
BTN_FAVORITES = "⭐ Sevimlilarim"
BTN_TRENDING = "🔥 Trending"
BTN_TOP_VIEWED = "🏆 Top ko'rilganlar"
BTN_NOTIFICATIONS = "🔔 Bildirishnomalar"
BTN_PRO_BUY = "👑 Pro olish"
BTN_PRO_STATUS = "💎 Pro holatim"
BTN_CREATE_AD = "📢 E'lon berish"
BTN_MY_ADS = "🗂 E'lonlarim"
BTN_PRO_MANAGE = "👑 Pro boshqarish"
BTN_PRO_PRICE = "💰 Pro narxi"
BTN_PRO_DURATION = "⏳ Pro muddati"
BTN_PRO_REQUESTS = "💳 Pro so'rovlar"
BTN_ADS = "📰 E'lonlar"
BTN_AD_CHANNELS = "📡 E'lon kanalari"
BTN_YES = "✅ Ha"
BTN_NO = "❌ Yo'q"
BTN_CONFIRM = "✅ Tasdiqlash"
BTN_SKIP = "/skip"
BOT_SIGNATURE = "@MirTopKinoBot"


def generate_missing_numeric_codes(existing_codes: Iterable[str], count: int) -> list[str]:
    if count <= 0:
        return []

    numeric_existing = {
        int(code.strip())
        for code in existing_codes
        if isinstance(code, str) and code.strip().isdigit() and int(code.strip()) > 0
    }
    max_existing = max(numeric_existing) if numeric_existing else 0
    upper_bound = max(max_existing + (count * 30), 500)
    max_attempts = max(2000, upper_bound * 2)

    picked: set[int] = set()
    result: list[str] = []
    attempts = 0
    while len(result) < count and attempts < max_attempts:
        attempts += 1
        candidate = random.randint(1, upper_bound)
        if candidate in numeric_existing or candidate in picked:
            continue
        picked.add(candidate)
        result.append(str(candidate))

    next_candidate = max_existing + 1 if max_existing > 0 else 1
    while len(result) < count:
        if next_candidate not in numeric_existing and next_candidate not in picked:
            picked.add(next_candidate)
            result.append(str(next_candidate))
        next_candidate += 1

    return result


def is_cancel_text(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized in {BTN_CANCEL.lower(), "bekor qilish"}


def is_serial_done_text(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized in {
        BTN_SERIAL_DONE.lower(),
        "serialni yakunlash",
        "yakunlash",
        "tayyor",
        "done",
    }


def is_skip_text(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {BTN_SKIP.lower(), "-", "skip"}


def is_yes_text(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {BTN_YES.lower(), "ha", "yes"}


def is_no_text(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {BTN_NO.lower(), "yo'q", "yoq", "no"}


def is_confirm_text(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {BTN_CONFIRM.lower(), "tasdiqlash", "yuborish"}


def main_menu_kb(is_admin: bool) -> ReplyKeyboardMarkup | ReplyKeyboardRemove:
    buttons = [
        [KeyboardButton(text=BTN_SEARCH_NAME), KeyboardButton(text=BTN_TRENDING)],
        [KeyboardButton(text=BTN_TOP_VIEWED), KeyboardButton(text=BTN_FAVORITES)],
        [KeyboardButton(text=BTN_PRO_BUY), KeyboardButton(text=BTN_PRO_STATUS)],
        [KeyboardButton(text=BTN_CREATE_AD), KeyboardButton(text=BTN_MY_ADS)],
        [KeyboardButton(text=BTN_NOTIFICATIONS)],
    ]
    if is_admin:
        buttons.append([KeyboardButton(text=BTN_ADMIN_PANEL)])
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


def admin_menu_kb() -> ReplyKeyboardMarkup:
    buttons = [
        [KeyboardButton(text=BTN_SUBS)],
        [KeyboardButton(text=BTN_ADD_MOVIE), KeyboardButton(text=BTN_ADD_SERIAL)],
        [KeyboardButton(text=BTN_DEL_MOVIE), KeyboardButton(text=BTN_EDIT_CONTENT)],
        [KeyboardButton(text=BTN_LIST_MOVIES), KeyboardButton(text=BTN_BROADCAST)],
        [KeyboardButton(text=BTN_REQUESTS), KeyboardButton(text=BTN_STATS)],
        [KeyboardButton(text=BTN_PRO_MANAGE), KeyboardButton(text=BTN_PRO_PRICE)],
        [KeyboardButton(text=BTN_PRO_DURATION), KeyboardButton(text=BTN_PRO_REQUESTS)],
        [KeyboardButton(text=BTN_ADS), KeyboardButton(text=BTN_AD_CHANNELS)],
        [KeyboardButton(text=BTN_RANDOM_CODES)],
        [KeyboardButton(text=BTN_ADD_ADMIN)],
        [KeyboardButton(text=BTN_BACK)],
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


def sub_manage_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="➕ Kanal qo'shish", callback_data="sub_add")
    builder.button(text="📋 Kanallar ro'yxati", callback_data="sub_list")
    builder.button(text="🗑 Kanal o'chirish", callback_data="sub_delete_menu")
    builder.adjust(1)
    return builder.as_markup()


def cancel_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BTN_CANCEL)]],
        resize_keyboard=True,
    )


def yes_no_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BTN_YES), KeyboardButton(text=BTN_NO)], [KeyboardButton(text=BTN_CANCEL)]],
        resize_keyboard=True,
    )


def confirm_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BTN_CONFIRM), KeyboardButton(text=BTN_CANCEL)]],
        resize_keyboard=True,
    )


def serial_upload_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BTN_SERIAL_DONE), KeyboardButton(text=BTN_CANCEL)]],
        resize_keyboard=True,
    )


def build_url_button_kb(button_text: str, button_url: str) -> InlineKeyboardMarkup | None:
    text = button_text.strip()
    url = button_url.strip()
    if not text or not url:
        return None
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=text, url=url)]])


def build_inline_choice_kb(prefix: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Ha", callback_data=f"{prefix}:yes"),
                InlineKeyboardButton(text="❌ Yo'q", callback_data=f"{prefix}:no"),
            ]
        ]
    )


def content_should_be_protected(tg_id: int | None) -> bool:
    return not bool(tg_id and db.is_admin(tg_id))


def build_pro_purchase_kb() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if PRO_PAYMENT_LINK_1:
        rows.append([InlineKeyboardButton(text="💳 To'lov havolasi 1", url=PRO_PAYMENT_LINK_1)])
    if PRO_PAYMENT_LINK_2:
        rows.append([InlineKeyboardButton(text="🌐 To'lov havolasi 2", url=PRO_PAYMENT_LINK_2)])
    rows.append([InlineKeyboardButton(text="✅ To'lov qildim", callback_data="pro_paid")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_notification_settings_kb(settings: dict[str, Any]) -> InlineKeyboardMarkup:
    def state_text(key: str) -> str:
        return "✅" if settings.get(key) else "❌"

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"{state_text('new_content')} Yangi kino", callback_data="notif:new_content")],
            [InlineKeyboardButton(text=f"{state_text('pro_updates')} Pro xabarlari", callback_data="notif:pro_updates")],
            [InlineKeyboardButton(text=f"{state_text('ads_updates')} E'lon holati", callback_data="notif:ads_updates")],
        ]
    )


def build_payment_request_review_kb(request_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Tasdiqlash", callback_data=f"proreq:approve:{request_id}"),
                InlineKeyboardButton(text="❌ Rad etish", callback_data=f"proreq:reject:{request_id}"),
            ]
        ]
    )


def build_ad_manage_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="➕ Kanal qo'shish", callback_data="adch:add")],
            [InlineKeyboardButton(text="📋 Kanallar ro'yxati", callback_data="adch:list")],
            [InlineKeyboardButton(text="🗑 Kanal o'chirish", callback_data="adch:delete_menu")],
        ]
    )


def build_ad_review_kb(ad_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📡 Kanal tanlash", callback_data=f"ad:channel:{ad_id}")],
            [InlineKeyboardButton(text="❌ Rad etish", callback_data=f"ad:reject:{ad_id}")],
        ]
    )


def build_ad_channel_pick_kb(ad_id: str, channels: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for channel in channels:
        channel_id = str(channel.get("id") or "").strip()
        title = str(channel.get("title") or channel.get("channel_ref") or "Kanal")
        if channel_id:
            rows.append([InlineKeyboardButton(text=f"📡 {title}", callback_data=f"ad:post:{ad_id}:{channel_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows or [[InlineKeyboardButton(text="Kanal yo'q", callback_data="ad:none")]])


def build_subscribe_keyboard(channels: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for channel in channels:
        join_link = channel["join_link"]
        ref = channel["channel_ref"]
        title = channel["title"] or ref
        if join_link:
            builder.row(InlineKeyboardButton(text=f"📌 {title}", url=join_link))
        elif ref.startswith("@"):
            builder.row(
                InlineKeyboardButton(
                    text=f"📌 {title}",
                    url=f"https://t.me/{ref[1:]}",
                )
            )
    builder.row(InlineKeyboardButton(text="✅ Obunani tekshirish", callback_data="check_sub"))
    return builder.as_markup()


def is_member_status(status: ChatMemberStatus) -> bool:
    return status in {
        ChatMemberStatus.CREATOR,
        ChatMemberStatus.ADMINISTRATOR,
        ChatMemberStatus.MEMBER,
        ChatMemberStatus.RESTRICTED,
    }


def normalize_code(text: str) -> str:
    return text.strip()


def normalize_lookup_text(value: str) -> str:
    cleaned = re.sub(r"[^\w\s]+", " ", value.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_metadata_input(value: str) -> dict[str, Any] | None:
    raw = value.strip()
    if not raw:
        return None
    if raw == "-":
        return {"year": None, "quality": "", "genres": []}

    parts = [part.strip() for part in raw.split("|")]
    if len(parts) != 3:
        return None

    year_part, quality_part, genres_part = parts
    year: int | None = None
    if year_part and year_part != "-":
        if not year_part.isdigit():
            return None
        year = int(year_part)
        now_year = datetime.now(UTC).year
        if year < 1900 or year > now_year + 1:
            return None

    quality = ""
    if quality_part and quality_part != "-":
        quality = quality_part[:40]

    genres: list[str] = []
    if genres_part and genres_part != "-":
        genres = [genre.strip().lower() for genre in genres_part.split(",") if genre.strip()]
        genres = sorted(dict.fromkeys(genres))[:10]

    return {"year": year, "quality": quality, "genres": genres}


def encode_payload_value(value: str) -> str:
    data = value.encode("utf-8")
    encoded = base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
    return encoded


def decode_payload_value(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None
    padding = "=" * ((4 - len(raw) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode((raw + padding).encode("ascii"))
        return decoded.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def compact_query_token(value: str) -> str:
    normalized = normalize_lookup_text(value)
    if not normalized:
        return ""
    if len(normalized) <= 40 and all(ch.isalnum() or ch in {"_", "-", " "} for ch in normalized):
        return normalized
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]


def format_meta_line(year: int | None, quality: str | None, genres: list[str] | None) -> str:
    parts: list[str] = []
    if year:
        parts.append(str(year))
    if quality:
        parts.append(str(quality))
    if genres:
        parts.append(", ".join(genres))
    return " | ".join(parts)


def append_meta_to_caption(
    caption: str | None,
    year: int | None,
    quality: str | None,
    genres: list[str] | None,
) -> str:
    base = (caption or "").strip()
    meta = format_meta_line(year, quality, genres)
    if not meta:
        return base
    if base:
        return f"{base}\n\n{meta}"
    return meta


def resolve_inline_media_preview(item: dict[str, Any]) -> tuple[str, str] | None:
    content_type = str(item.get("content_type") or "")
    media_type = str(item.get("media_type") or "")
    file_id = str(item.get("file_id") or "")
    preview_media_type = str(item.get("preview_media_type") or "")
    preview_file_id = str(item.get("preview_file_id") or "")
    preview_photo_file_id = str(item.get("preview_photo_file_id") or "")

    # Prefer extracted image previews from the original media.
    if preview_photo_file_id:
        return "photo", preview_photo_file_id
    if preview_media_type == "photo" and preview_file_id:
        return "photo", preview_file_id

    if content_type == "movie":
        if media_type == "photo" and file_id:
            return "photo", file_id
        if media_type == "video" and file_id:
            return "video", file_id
    if content_type == "serial":
        if preview_media_type == "video" and preview_file_id:
            return "video", preview_file_id

    return None


def build_not_found_request_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="📩 So'rov qoldirish", callback_data="req_create")
    builder.adjust(1)
    return builder.as_markup()


def build_movie_actions_kb(
    movie_id: str,
    is_favorite: bool,
    likes: int = 0,
    dislikes: int = 0,
) -> InlineKeyboardMarkup:
    fav_text = "💔 Sevimlidan olib tashlash" if is_favorite else "⭐ Sevimliga qo'shish"
    fav_action = "del" if is_favorite else "add"
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text=fav_text, callback_data=f"fav:{fav_action}:movie:{movie_id}"))
    builder.row(
        InlineKeyboardButton(text=f"👍 {max(0, likes)}", callback_data=f"react:like:movie:{movie_id}"),
        InlineKeyboardButton(text=f"👎 {max(0, dislikes)}", callback_data=f"react:dislike:movie:{movie_id}"),
    )
    return builder.as_markup()




def build_search_results_kb(items: list[dict[str, Any]]) -> InlineKeyboardMarkup | None:
    if not items:
        return None
    builder = InlineKeyboardBuilder()
    for item in items:
        content_type = str(item.get("content_type") or "").strip()
        content_ref = str(item.get("id") or "").strip()
        if not content_type or not content_ref:
            continue
        title = str(item.get("title") or "Noma'lum")
        code = str(item.get("code") or "")
        year = item.get("year")
        quality = str(item.get("quality") or "")
        meta = format_meta_line(year if isinstance(year, int) else None, quality, None)
        label = title
        if code:
            label = f"{code} - {title}"
        if meta:
            label = f"{label} ({meta})"
        if len(label) > 60:
            label = f"{label[:57]}..."
        icon = "🎬" if content_type == "movie" else "📺"
        builder.button(text=f"{icon} {label}", callback_data=f"open:{content_type}:{content_ref}")
    builder.adjust(1)
    markup = builder.as_markup()
    if not markup.inline_keyboard:
        return None
    return markup


def build_favorites_kb(items: list[dict[str, Any]]) -> InlineKeyboardMarkup | None:
    if not items:
        return None
    builder = InlineKeyboardBuilder()
    for item in items:
        content_type = str(item.get("content_type") or "")
        content_ref = str(item.get("content_ref") or "")
        title = str(item.get("title") or "Noma'lum")
        code = str(item.get("code") or "")
        display = f"{code} - {title}" if code else title
        if len(display) > 48:
            display = f"{display[:45]}..."
        builder.row(
            InlineKeyboardButton(
                text=f"▶️ {display}",
                callback_data=f"open:{content_type}:{content_ref}",
            ),
            InlineKeyboardButton(
                text="❌",
                callback_data=f"fav:del:{content_type}:{content_ref}",
            ),
        )
    markup = builder.as_markup()
    if not markup.inline_keyboard:
        return None
    return markup


def append_signature(caption: str | None) -> str:
    base = (caption or "").strip()
    if base:
        if base.endswith(BOT_SIGNATURE):
            return base
        return f"{base}\n\n{BOT_SIGNATURE}"
    return BOT_SIGNATURE


def split_text_chunks(text: str, max_len: int = 3800) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    current = ""
    for line in lines:
        piece = f"{line}\n"
        if current and len(current) + len(piece) > max_len:
            chunks.append(current.rstrip())
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current.rstrip())
    return chunks


def build_movie_caption(title: str | None, description: str | None) -> str:
    title_text = (title or "").strip()
    description_text = (description or "").strip()
    if title_text and description_text:
        return f"{title_text}\n\n{description_text}"
    return title_text or description_text


def append_views_to_caption(caption: str | None, views: int | None) -> str:
    views_count = max(0, int(views or 0))
    views_line = f"👁 Ko'rishlar: {views_count}"
    base = (caption or "").strip()
    if base:
        return f"{base}\n\n{views_line}"
    return views_line


def append_reaction_stats_to_caption(
    caption: str | None,
    likes: int | None,
    dislikes: int | None,
) -> str:
    likes_count = max(0, int(likes or 0))
    dislikes_count = max(0, int(dislikes or 0))
    total = likes_count + dislikes_count
    rating = round((likes_count / total) * 5, 1) if total else 0.0
    stats_line = f"👍 {likes_count} | 👎 {dislikes_count} | ⭐ Reyting: {rating}/5"
    base = (caption or "").strip()
    if base:
        return f"{base}\n{stats_line}"
    return stats_line


def build_serial_caption(
    title: str | None,
    description: str | None,
    episodes_count: int,
    year: int | None = None,
    quality: str | None = None,
    genres: list[str] | None = None,
) -> str:
    base = build_movie_caption(title, description)
    meta = format_meta_line(year, quality, genres)
    tail = f"🎞 Qismlar soni: {episodes_count}"
    if meta:
        tail = f"{meta}\n{tail}"
    if base:
        return f"{base}\n\n{tail}"
    return tail


def format_pro_payment_code(user_id: int) -> str:
    return f"PRO-{user_id}"


def normalize_button_url(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if text.startswith("t.me/"):
        text = f"https://{text}"
    if text.startswith("@"):
        text = f"https://t.me/{text[1:]}"
    if text.startswith("tg://"):
        return text
    parsed = urlparse(text if "://" in text else f"https://{text}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return parsed.geturl()


def format_ad_caption(title: str, description: str) -> str:
    title_text = title.strip()
    description_text = description.strip()
    if title_text and description_text:
        return f"{title_text}\n\n{description_text}"
    return title_text or description_text


def build_serial_episodes_kb(
    serial_id: str,
    episode_numbers: list[int],
    is_favorite: bool = False,
    likes: int = 0,
    dislikes: int = 0,
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    fav_text = "💔 Sevimlidan olib tashlash" if is_favorite else "⭐ Sevimliga qo'shish"
    fav_action = "del" if is_favorite else "add"
    builder.row(
        InlineKeyboardButton(
            text=fav_text,
            callback_data=f"fav:{fav_action}:serial:{serial_id}",
        )
    )
    builder.row(
        InlineKeyboardButton(text=f"👍 {max(0, likes)}", callback_data=f"react:like:serial:{serial_id}"),
        InlineKeyboardButton(text=f"👎 {max(0, dislikes)}", callback_data=f"react:dislike:serial:{serial_id}"),
    )
    for number in episode_numbers:
        builder.button(text=str(number), callback_data=f"serial_ep:{serial_id}:{number}")
    builder.adjust(5)
    return builder.as_markup()


def build_episode_navigation_kb(
    serial_id: str,
    episode_numbers: list[int],
    current_episode: int,
) -> InlineKeyboardMarkup | None:
    unique_sorted = sorted({int(number) for number in episode_numbers})
    if current_episode not in unique_sorted:
        return None

    idx = unique_sorted.index(current_episode)
    prev_episode = unique_sorted[idx - 1] if idx > 0 else None
    next_episode = unique_sorted[idx + 1] if idx < len(unique_sorted) - 1 else None

    if prev_episode is None and next_episode is None:
        return None

    builder = InlineKeyboardBuilder()
    if prev_episode is not None:
        builder.button(text="⬅️ Oldingi qism", callback_data=f"serial_ep:{serial_id}:{prev_episode}")
    if next_episode is not None:
        builder.button(text="➡️ Keyingi qism", callback_data=f"serial_ep:{serial_id}:{next_episode}")
    builder.adjust(2)
    return builder.as_markup()


def parse_telegram_post_link(link: str) -> tuple[str, int] | None:
    parsed = urlparse(link.strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.netloc or "").lower()
    if host not in {"t.me", "www.t.me", "telegram.me", "www.telegram.me"}:
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None

    # Private/internal channel link: /c/<internal_id>/<message_id>
    if parts[0] == "c" and len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
        chat_ref = f"-100{parts[1]}"
        return chat_ref, int(parts[2])

    # Public channel/group link: /<username>/<message_id>
    if parts[1].isdigit():
        channel = parts[0]
        if channel.startswith("@"):
            chat_ref = channel
        else:
            chat_ref = f"@{channel}"
        return chat_ref, int(parts[1])

    return None


def normalize_channel_ref_input(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None

    if raw.startswith("@"):
        username = raw[1:].strip()
        return f"@{username}" if username else None

    if raw.lstrip("-").isdigit():
        return raw

    # Plain username without @
    if raw.replace("_", "").isalnum():
        return f"@{raw}"

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        host = (parsed.netloc or "").lower()
        if host not in {"t.me", "www.t.me", "telegram.me", "www.telegram.me"}:
            return None
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            return None

        # t.me/c/<internal_id>/<message_id>
        if parts[0] == "c" and len(parts) >= 2 and parts[1].isdigit():
            return f"-100{parts[1]}"

        first = parts[0]
        if first.startswith("+") or first == "joinchat":
            return None
        username = first[1:] if first.startswith("@") else first
        return f"@{username}" if username else None

    return None


def normalize_invite_link_input(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.netloc or "").lower()
    if host not in {"t.me", "www.t.me", "telegram.me", "www.telegram.me"}:
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return None

    first = parts[0]
    if first.startswith("+") and len(first) > 1:
        return raw
    if first == "joinchat" and len(parts) >= 2 and parts[1]:
        return raw

    return None


def pack_post_ref(chat_ref: str, message_id: int) -> str:
    return f"{chat_ref}|{message_id}"


def unpack_post_ref(value: str) -> tuple[str, int] | None:
    if "|" not in value:
        return None
    left, right = value.split("|", 1)
    chat_ref = left.strip()
    msg = right.strip()
    if not chat_ref or not msg.isdigit():
        return None
    return chat_ref, int(msg)


def extract_preview_photo_file_id(message: Message) -> str | None:
    if message.photo:
        return message.photo[-1].file_id
    if message.video and message.video.thumbnail:
        return message.video.thumbnail.file_id
    if message.document and message.document.thumbnail:
        return message.document.thumbnail.file_id
    if message.animation and message.animation.thumbnail:
        return message.animation.thumbnail.file_id
    return None


def parse_message_media(message: Message) -> tuple[str, str] | None:
    if message.content_type == ContentType.VIDEO and message.video:
        return "video", message.video.file_id
    if message.content_type == ContentType.DOCUMENT and message.document:
        return "document", message.document.file_id
    if message.content_type == ContentType.PHOTO and message.photo:
        return "photo", message.photo[-1].file_id
    if message.content_type == ContentType.ANIMATION and message.animation:
        return "animation", message.animation.file_id
    if message.text:
        text = message.text.strip()
        if text.startswith("http://") or text.startswith("https://"):
            post_data = parse_telegram_post_link(text)
            if post_data:
                return "telegram_post", pack_post_ref(post_data[0], post_data[1])
            return "link", text
        return "file_id", text
    return None


async def send_stored_media(
    message: Message,
    media_type: str,
    file_id: str,
    caption: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    final_caption = append_signature(caption)
    protect_content = content_should_be_protected(message.from_user.id if message.from_user else None)
    if media_type == "video":
        await message.answer_video(file_id, caption=final_caption, reply_markup=reply_markup, protect_content=protect_content)
    elif media_type == "document":
        await message.answer_document(file_id, caption=final_caption, reply_markup=reply_markup, protect_content=protect_content)
    elif media_type == "photo":
        await message.answer_photo(file_id, caption=final_caption, reply_markup=reply_markup, protect_content=protect_content)
    elif media_type == "animation":
        await message.answer_animation(file_id, caption=final_caption, reply_markup=reply_markup, protect_content=protect_content)
    elif media_type == "telegram_post":
        post_data = unpack_post_ref(file_id)
        if not post_data:
            raise ValueError("Invalid telegram post reference")
        from_chat_id: int | str
        if post_data[0].lstrip("-").isdigit():
            from_chat_id = int(post_data[0])
        else:
            from_chat_id = post_data[0]
        await message.bot.copy_message(
            chat_id=message.chat.id,
            from_chat_id=from_chat_id,
            message_id=post_data[1],
            caption=final_caption,
            reply_markup=reply_markup,
            protect_content=protect_content,
        )
    elif media_type == "link":
        post_data = parse_telegram_post_link(file_id)
        if post_data:
            from_chat_id: int | str
            if post_data[0].lstrip("-").isdigit():
                from_chat_id = int(post_data[0])
            else:
                from_chat_id = post_data[0]
            await message.bot.copy_message(
                chat_id=message.chat.id,
                from_chat_id=from_chat_id,
                message_id=post_data[1],
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
        else:
            await message.answer(
                f"{final_caption}\n\nLink: {file_id}",
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
    else:
        await message.answer(
            f"{final_caption}\n\nID: {file_id}",
            reply_markup=reply_markup,
            protect_content=protect_content,
        )


def parse_start_payload(text: str | None) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if not raw.lower().startswith("/start"):
        return None
    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        return None
    payload = parts[1].strip()
    return payload or None


def parse_serial_payload(payload: str | None) -> str | None:
    value = (payload or "").strip()
    if not value.startswith("s_"):
        return None
    serial_id = value[2:].strip()
    return serial_id or None


def parse_movie_payload(payload: str | None) -> str | None:
    value = (payload or "").strip()
    if not value.startswith("m_"):
        return None
    movie_id = value[2:].strip()
    return movie_id or None


def build_start_deeplink(username: str, payload: str) -> str:
    return f"https://t.me/{username}?start={payload}"


async def send_serial_selector_by_id(message: Message, serial_id: str, user_id: int | None = None) -> bool:
    serial = db.get_serial(serial_id)
    if not serial:
        await message.answer("❌ Serial topilmadi.")
        return False

    episodes = db.list_serial_episodes(serial_id)
    if not episodes:
        await message.answer("📭 Bu serialga hali qism qo'shilmagan.")
        return False

    episode_numbers = [row["episode_number"] for row in episodes]
    requester_id = user_id
    if requester_id is None and message.from_user and not message.from_user.is_bot:
        requester_id = message.from_user.id
    is_favorite = bool(requester_id and db.is_favorite(requester_id, "serial", serial["id"]))
    reaction = db.get_reaction_summary("serial", serial["id"])
    displayed_views = int(serial.get("views") or 0) + 1
    serial_caption = build_serial_caption(
        serial["title"],
        serial["description"],
        episodes_count=len(episode_numbers),
        year=serial.get("year") if isinstance(serial.get("year"), int) else None,
        quality=str(serial.get("quality") or ""),
        genres=[str(g) for g in serial.get("genres", []) if str(g).strip()],
    )
    serial_caption = append_views_to_caption(serial_caption, displayed_views)
    serial_caption = append_reaction_stats_to_caption(
        serial_caption,
        reaction.get("likes"),
        reaction.get("dislikes"),
    )
    await message.answer(
        f"{serial_caption}\n\n👇 Kerakli qismni tanlang:",
        reply_markup=build_serial_episodes_kb(
            serial["id"],
            episode_numbers,
            is_favorite=is_favorite,
            likes=int(reaction.get("likes") or 0),
            dislikes=int(reaction.get("dislikes") or 0),
        ),
        protect_content=content_should_be_protected(requester_id),
    )
    db.increment_serial_views(serial["id"])
    return True


async def send_media_to_chat(
    bot: Bot,
    chat_ref: str,
    media_type: str,
    file_id: str,
    caption: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    final_caption = append_signature(caption)
    chat_id: int | str = int(chat_ref) if chat_ref.lstrip("-").isdigit() else chat_ref
    protect_content = content_should_be_protected(chat_id if isinstance(chat_id, int) and chat_id > 0 else None)

    if media_type == "video":
        await bot.send_video(
            chat_id=chat_id,
            video=file_id,
            caption=final_caption,
            reply_markup=reply_markup,
            protect_content=protect_content,
        )
        return
    if media_type == "document":
        await bot.send_document(
            chat_id=chat_id,
            document=file_id,
            caption=final_caption,
            reply_markup=reply_markup,
            protect_content=protect_content,
        )
        return
    if media_type == "photo":
        await bot.send_photo(
            chat_id=chat_id,
            photo=file_id,
            caption=final_caption,
            reply_markup=reply_markup,
            protect_content=protect_content,
        )
        return
    if media_type == "animation":
        await bot.send_animation(
            chat_id=chat_id,
            animation=file_id,
            caption=final_caption,
            reply_markup=reply_markup,
            protect_content=protect_content,
        )
        return
    if media_type == "telegram_post":
        post_data = unpack_post_ref(file_id)
        if not post_data:
            raise ValueError("Invalid telegram post reference")
        from_chat_id: int | str = int(post_data[0]) if post_data[0].lstrip("-").isdigit() else post_data[0]
        await bot.copy_message(
            chat_id=chat_id,
            from_chat_id=from_chat_id,
            message_id=post_data[1],
            caption=final_caption,
            reply_markup=reply_markup,
            protect_content=protect_content,
        )
        return
    if media_type == "link":
        post_data = parse_telegram_post_link(file_id)
        if post_data:
            from_chat_id: int | str = int(post_data[0]) if post_data[0].lstrip("-").isdigit() else post_data[0]
            await bot.copy_message(
                chat_id=chat_id,
                from_chat_id=from_chat_id,
                message_id=post_data[1],
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        await bot.send_message(
            chat_id=chat_id,
            text=f"{final_caption}\n\nLink: {file_id}",
            reply_markup=reply_markup,
            protect_content=protect_content,
        )
        return

    await bot.send_message(
        chat_id=chat_id,
        text=f"{final_caption}\n\nID: {file_id}",
        reply_markup=reply_markup,
        protect_content=protect_content,
    )


async def copy_source_message_to_chat(
    bot: Bot,
    target_chat_id: int | str,
    source_chat_id: int | str,
    source_message_id: int,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    await bot.copy_message(
        chat_id=target_chat_id,
        from_chat_id=source_chat_id,
        message_id=source_message_id,
        reply_markup=reply_markup,
    )


async def send_ad_preview(
    target_message: Message,
    title: str,
    description: str,
    *,
    photo_file_id: str = "",
    button_text: str = "",
    button_url: str = "",
    footer_text: str = "",
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    caption = format_ad_caption(title, description)
    if footer_text:
        caption = f"{caption}\n\n{footer_text}" if caption else footer_text
    markup = reply_markup or build_url_button_kb(button_text, button_url)
    if photo_file_id:
        await target_message.answer_photo(photo_file_id, caption=caption, reply_markup=markup)
        return
    await target_message.answer(caption or "E'lon", reply_markup=markup)


async def post_ad_to_channel(bot: Bot, channel_ref: str, ad: dict[str, Any]) -> None:
    chat_id: int | str = int(channel_ref) if channel_ref.lstrip("-").isdigit() else channel_ref
    caption = format_ad_caption(str(ad.get("title") or ""), str(ad.get("description") or ""))
    markup = build_url_button_kb(str(ad.get("button_text") or ""), str(ad.get("button_url") or ""))
    photo_file_id = str(ad.get("photo_file_id") or "").strip()
    if photo_file_id:
        await bot.send_photo(chat_id=chat_id, photo=photo_file_id, caption=caption, reply_markup=markup)
        return
    await bot.send_message(chat_id=chat_id, text=caption or "E'lon", reply_markup=markup)


BOT_USERNAME_CACHE: str | None = None


async def get_bot_username(bot: Bot) -> str | None:
    global BOT_USERNAME_CACHE
    if BOT_USERNAME_CACHE:
        return BOT_USERNAME_CACHE
    me = await bot.get_me()
    BOT_USERNAME_CACHE = me.username
    return BOT_USERNAME_CACHE


async def send_movie_by_id(message: Message, movie_id: str, user_id: int | None = None) -> bool:
    movie = db.get_movie_by_id(movie_id)
    if not movie:
        await message.answer("❌ Kino topilmadi.")
        return False
    displayed_views = int(movie.get("views") or 0) + 1
    reaction = db.get_reaction_summary("movie", movie["id"])
    caption = append_meta_to_caption(
        build_movie_caption(movie["title"], movie["description"]),
        movie.get("year") if isinstance(movie.get("year"), int) else None,
        str(movie.get("quality") or ""),
        [str(g) for g in movie.get("genres", []) if str(g).strip()],
    )
    caption = append_views_to_caption(caption, displayed_views)
    caption = append_reaction_stats_to_caption(caption, reaction.get("likes"), reaction.get("dislikes"))
    requester_id = user_id
    if requester_id is None and message.from_user and not message.from_user.is_bot:
        requester_id = message.from_user.id
    is_favorite = bool(requester_id and db.is_favorite(requester_id, "movie", movie["id"]))
    await send_stored_media(
        message,
        media_type=movie["media_type"],
        file_id=movie["file_id"],
        caption=caption if caption else None,
        reply_markup=build_movie_actions_kb(
            movie["id"],
            is_favorite=is_favorite,
            likes=int(reaction.get("likes") or 0),
            dislikes=int(reaction.get("dislikes") or 0),
        ),
    )
    db.increment_movie_views(movie["id"])
    db.increment_movie_downloads(movie["id"])
    return True


async def notify_requesters_for_content(
    bot: Bot,
    content_type: str,
    content_ref: str,
    code: str,
    title: str,
    *,
    movie: dict[str, Any] | None = None,
    serial_id: str | None = None,
) -> tuple[int, int]:
    matched_requests = db.get_matching_open_requests(code=code, title=title)
    if not matched_requests:
        return 0, 0

    username = await get_bot_username(bot)
    grouped_by_user: dict[int, list[dict[str, Any]]] = {}
    for row in matched_requests:
        user_tg_id = row.get("user_tg_id")
        if not isinstance(user_tg_id, int):
            continue
        grouped_by_user.setdefault(user_tg_id, []).append(row)

    delivered = 0
    failed = 0
    all_request_ids: list[str] = []
    for req in matched_requests:
        request_id = str(req.get("id") or "").strip()
        if request_id:
            all_request_ids.append(request_id)

    for user_id, user_requests in grouped_by_user.items():
        request_ids = [str(r.get("id") or "").strip() for r in user_requests if r.get("id")]
        try:
            if content_type == "movie" and movie:
                caption = append_meta_to_caption(
                    build_movie_caption(movie.get("title"), movie.get("description")),
                    movie.get("year") if isinstance(movie.get("year"), int) else None,
                    str(movie.get("quality") or ""),
                    [str(g) for g in movie.get("genres", []) if str(g).strip()],
                )
                await send_media_to_chat(
                    bot=bot,
                    chat_ref=str(user_id),
                    media_type=str(movie.get("media_type") or ""),
                    file_id=str(movie.get("file_id") or ""),
                    caption=caption if caption else None,
                    reply_markup=build_movie_actions_kb(
                        str(movie.get("id") or ""),
                        is_favorite=False,
                    ),
                )
            elif content_type == "serial" and serial_id and username:
                deeplink = build_start_deeplink(username, f"s_{serial_id}")
                kb = InlineKeyboardMarkup(
                    inline_keyboard=[[InlineKeyboardButton(text="📥 Qismlarni ochish", url=deeplink)]]
                )
                await bot.send_message(
                    chat_id=user_id,
                    text=f"Siz so'ragan kontent qo'shildi: {title}\nKod: {code}",
                    reply_markup=kb,
                )
            else:
                await bot.send_message(
                    chat_id=user_id,
                    text=f"Siz so'ragan kontent qo'shildi: {title}\nKod: {code}",
                )
            delivered += 1
            for request_id in request_ids:
                db.log_notification_attempt(
                    user_tg_id=user_id,
                    request_id=request_id,
                    content_type=content_type,
                    content_ref=content_ref,
                    status="delivered",
                )
        except (TelegramBadRequest, TelegramForbiddenError) as exc:
            failed += 1
            for request_id in request_ids:
                db.log_notification_attempt(
                    user_tg_id=user_id,
                    request_id=request_id,
                    content_type=content_type,
                    content_ref=content_ref,
                    status="failed",
                    error_message=str(exc),
                )

    db.mark_requests_fulfilled(all_request_ids, content_type=content_type, content_ref=content_ref)
    return delivered, failed


def build_pro_offer_text(user_id: int) -> str:
    settings = db.get_bot_settings()
    payment_code = format_pro_payment_code(user_id)
    return (
        "👑 PRO obuna\n\n"
        "✨ Tarif: Yagona PRO\n"
        f"💰 Narx: {settings['pro_price_text']}\n"
        f"🗓 Muddat: {settings['pro_duration_days']} kun\n\n"
        "🧾 To'lov qilayotganda izohga quyidagini yozing:\n"
        f"`{payment_code}`\n\n"
        f"Yoki kamida Telegram ID'ingizni yozing: `{user_id}`\n\n"
        "✅ To'lov qilgach, shu yerda `✅ To'lov qildim` tugmasini bosing."
    )


def build_pro_status_text(user_id: int) -> str:
    info = db.get_pro_status(user_id)
    if info["is_pro"]:
        until_text = info["pro_until"] or "-"
        return (
            "💎 Sizda PRO aktiv.\n"
            f"⏳ Amal qiladi: {until_text}\n"
            f"💰 Joriy tarif: {info['pro_price_text']}"
        )
    if info["pro_status"] == "expired":
        return (
            "⌛ Sizning PRO muddati tugagan.\n"
            f"⏳ Oxirgi muddat: {info['pro_until'] or '-'}\n"
            f"💰 Joriy narx: {info['pro_price_text']}"
        )
    return (
        "🔒 Sizda PRO aktiv emas.\n"
        f"💰 Joriy narx: {info['pro_price_text']}\n"
        "👑 PRO olish tugmasi orqali faollashtiring."
    )


async def notify_admins_about_payment_request(bot: Bot, payment_request: dict[str, Any]) -> None:
    text = (
        "💳 Yangi PRO so'rovi\n"
        f"👤 User ID: {payment_request.get('user_tg_id')}\n"
        f"🔑 Kod: {payment_request.get('payment_code') or '-'}\n"
        f"📝 Izoh: {payment_request.get('comment') or '-'}"
    )
    markup = build_payment_request_review_kb(str(payment_request.get("id") or ""))
    proof_media_type = str(payment_request.get("proof_media_type") or "")
    proof_file_id = str(payment_request.get("proof_file_id") or "")
    for admin_id in db.list_admin_ids():
        try:
            if proof_media_type and proof_file_id:
                await send_media_to_chat(
                    bot,
                    str(admin_id),
                    proof_media_type,
                    proof_file_id,
                    caption=text,
                    reply_markup=markup,
                )
            else:
                await bot.send_message(chat_id=admin_id, text=text, reply_markup=markup)
        except (TelegramBadRequest, TelegramForbiddenError):
            continue


async def notify_admins_about_ad(bot: Bot, ad: dict[str, Any]) -> None:
    footer = f"👤 User ID: {ad.get('user_tg_id')}\n🆔 Ad ID: {ad.get('id')}"
    markup = build_ad_review_kb(str(ad.get("id") or ""))
    caption = format_ad_caption(str(ad.get("title") or ""), str(ad.get("description") or ""))
    caption = f"{caption}\n\n{footer}" if caption else footer
    photo_file_id = str(ad.get("photo_file_id") or "")
    button_markup = build_url_button_kb(str(ad.get("button_text") or ""), str(ad.get("button_url") or ""))
    for admin_id in db.list_admin_ids():
        try:
            if photo_file_id:
                await bot.send_photo(chat_id=admin_id, photo=photo_file_id, caption=caption, reply_markup=markup)
            else:
                await bot.send_message(chat_id=admin_id, text=caption, reply_markup=markup)
            if button_markup:
                await bot.send_message(chat_id=admin_id, text="🔗 Foydalanuvchi tugmasi:", reply_markup=button_markup)
        except (TelegramBadRequest, TelegramForbiddenError):
            continue


async def notify_new_content_to_pro_users(
    bot: Bot,
    content_type: str,
    content_ref: str,
    title: str,
    code: str,
) -> tuple[int, int]:
    username = await get_bot_username(bot)
    if not username:
        return 0, 0
    payload = f"m_{content_ref}" if content_type == "movie" else f"s_{content_ref}"
    deeplink = build_start_deeplink(username, payload)
    delivered = 0
    failed = 0
    for user_id in db.list_active_pro_user_ids():
        settings = db.get_notification_settings(user_id)
        if not settings.get("new_content"):
            continue
        try:
            await bot.send_message(
                chat_id=user_id,
                text=(
                    f"🆕 Yangi {'kino' if content_type == 'movie' else 'serial'} qo'shildi!\n"
                    f"🎬 {title}\n"
                    f"🔢 Kod: {code}\n"
                    f"🔗 Botda ochish: {deeplink}"
                ),
            )
            delivered += 1
        except (TelegramBadRequest, TelegramForbiddenError):
            failed += 1
    return delivered, failed


load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "8537979650:AAFkSIbRnx7ha7muxZ1MDK5QMIxV5MAC4ww").strip()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:wGVAMNxMWZgocdRVBduRDnRlJePweOay@metro.proxy.rlwy.net:36399").strip()
MONGODB_DB = os.getenv("MONGODB_DB", "kino_bot").strip() or "kino_bot"
ADMIN_IDS = parse_admin_ids(os.getenv("ADMIN_IDS", "7903688837,7546181748"))
PRO_PRICE_TEXT_DEFAULT = os.getenv("PRO_PRICE_TEXT", "50 000 so'm / 30 kun").strip() or "50 000 so'm / 30 kun"
PRO_DURATION_DAYS_DEFAULT = max(1, int(os.getenv("PRO_DURATION_DAYS", "30") or 30))
PRO_PAYMENT_LINK_1 = os.getenv(
    "PRO_PAYMENT_LINK_1",
    "https://t.me/DanatlarBot/danat?startapp=Sara_Kinolar_o1",
).strip()
PRO_PAYMENT_LINK_2 = os.getenv(
    "PRO_PAYMENT_LINK_2",
    "https://danatlar.uz/Sara_Kinolar_o1",
).strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN topilmadi. .env faylga BOT_TOKEN yozing.")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI topilmadi. .env faylga MONGODB_URI yozing.")

if not ADMIN_IDS:
    raise RuntimeError("ADMIN_IDS bo'sh. Masalan: ADMIN_IDS=123456789")

db = Database(MONGODB_URI, MONGODB_DB)
db.seed_admins(ADMIN_IDS)

router = Router()


def guard_admin(message: Message) -> bool:
    return bool(message.from_user and db.is_admin(message.from_user.id))


def has_active_pro(message: Message) -> bool:
    return bool(message.from_user and db.is_pro_active(message.from_user.id))


@router.chat_join_request()
async def on_chat_join_request(join_request: ChatJoinRequest) -> None:
    chat_refs = {str(join_request.chat.id)}
    username = getattr(join_request.chat, "username", None)
    if username:
        chat_refs.add(f"@{username}")
    for chat_ref in chat_refs:
        db.mark_join_request(join_request.from_user.id, chat_ref)


async def ensure_subscription(user_id: int, bot: Bot) -> tuple[bool, list[dict[str, Any]]]:
    if db.is_pro_active(user_id):
        return True, []
    channels = db.get_required_channels()
    if not channels:
        return True, []

    pending_refs = db.get_join_request_refs(user_id)
    not_joined: list[dict[str, Any]] = []
    for channel in channels:
        channel_ref = channel["channel_ref"]
        has_pending_request = channel_ref in pending_refs
        try:
            member = await bot.get_chat_member(chat_id=channel_ref, user_id=user_id)
            if not is_member_status(member.status) and not has_pending_request:
                not_joined.append(channel)
        except (TelegramBadRequest, TelegramForbiddenError):
            if not has_pending_request:
                not_joined.append(channel)
    return len(not_joined) == 0, not_joined


async def ask_for_subscription(message: Message, channels: list[dict[str, Any]]) -> None:
    text_lines = [
        "🚫 Botdan foydalanish uchun quyidagi kanallarga obuna bo'lishingiz kerak.",
        "",
        "📢 Majburiy kanallar:",
    ]
    for ch in channels:
        title = ch["title"] or ch["channel_ref"]
        text_lines.append(f"• {title}")
    await message.answer("\n".join(text_lines), reply_markup=build_subscribe_keyboard(channels))


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return

    db.add_user(message.from_user.id, message.from_user.full_name)
    payload = parse_start_payload(message.text)
    serial_id = parse_serial_payload(payload)
    movie_id = parse_movie_payload(payload)
    if serial_id:
        ok, channels = await ensure_subscription(message.from_user.id, message.bot)
        if not ok:
            await state.update_data(pending_serial_id=serial_id)
            await ask_for_subscription(message, channels)
            return
        sent = await send_serial_selector_by_id(message, serial_id)
        if sent:
            return
    elif movie_id:
        ok, channels = await ensure_subscription(message.from_user.id, message.bot)
        if not ok:
            await state.update_data(pending_movie_id=movie_id)
            await ask_for_subscription(message, channels)
            return
        try:
            sent = await send_movie_by_id(message, movie_id)
        except (TelegramBadRequest, TelegramForbiddenError, ValueError):
            sent = False
        if sent:
            return

    admin = db.is_admin(message.from_user.id)
    text = (
        "🎬 Assalomu alaykum, Kino botga xush kelibsiz.\n\n"
        "🔎 Kino yoki serial kodini yuboring.\n"
        "🔥 Trending, 🏆 top kontent, 👑 PRO va 📢 e'lon bo'limlari ham tayyor.\n"
        + ("\n🛠 Siz adminsiz — user menyu va admin panel siz uchun ochiq." if admin else "")
    )
    await message.answer(text, reply_markup=main_menu_kb(admin))


@router.callback_query(F.data == "check_sub")
async def check_subscription(callback: CallbackQuery, state: FSMContext) -> None:
    user = callback.from_user
    if not user:
        return
    ok, channels = await ensure_subscription(user.id, callback.bot)
    if ok:
        state_data = await state.get_data()
        pending_serial_id = str(state_data.get("pending_serial_id") or "").strip()
        pending_movie_id = str(state_data.get("pending_movie_id") or "").strip()
        if pending_serial_id and callback.message:
            cleaned_state = dict(state_data)
            cleaned_state.pop("pending_serial_id", None)
            await state.set_data(cleaned_state)
            sent = await send_serial_selector_by_id(callback.message, pending_serial_id, user.id)
            if sent:
                await callback.answer("✅ Tasdiqlandi")
                return
        if pending_movie_id and callback.message:
            cleaned_state = dict(state_data)
            cleaned_state.pop("pending_movie_id", None)
            await state.set_data(cleaned_state)
            try:
                sent = await send_movie_by_id(callback.message, pending_movie_id, user.id)
            except (TelegramBadRequest, TelegramForbiddenError, ValueError):
                sent = False
            if sent:
                await callback.answer("✅ Tasdiqlandi")
                return
        await callback.message.answer(
            "✅ Obuna tasdiqlandi!\n\n🔎 Endi kino yoki serial kodini chatga yozing."
        )
        await callback.answer("✅ Tasdiqlandi")
    else:
        await callback.message.answer(
            "❗ Hali ham barcha kanallarga obuna bo'linmagan.",
            reply_markup=build_subscribe_keyboard(channels),
        )
        await callback.answer("❗ Obuna to'liq emas")


@router.message(F.text.in_({BTN_ADMIN_PANEL, "Admin panel"}))
async def open_admin_panel(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.clear()
    await message.answer("🛠 Admin panel ochildi.\nQuyidagi boshqaruv tugmalaridan foydalaning.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_BACK, "Ortga"}))
async def back_to_main(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.clear()
    await message.answer("🏠 Asosiy user menyu ochildi.", reply_markup=main_menu_kb(True))


@router.message(F.text.in_({BTN_PRO_BUY, "Pro olish"}))
async def pro_buy(message: Message) -> None:
    if not message.from_user:
        return
    db.add_user(message.from_user.id, message.from_user.full_name)
    if db.is_admin(message.from_user.id):
        await message.answer("👑 Siz adminsiz.\nSiz uchun PRO cheksiz va barcha premium funksiyalar ochiq.", reply_markup=main_menu_kb(True))
        return
    await message.answer(build_pro_offer_text(message.from_user.id), reply_markup=build_pro_purchase_kb())


@router.callback_query(F.data == "pro_paid")
async def pro_paid_start(callback: CallbackQuery, state: FSMContext) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    await state.set_state(PaymentState.waiting_proof)
    await callback.message.answer(
        "💳 To'lov skrini yoki hujjatini yuboring.\nO'tkazib yuborish uchun `/skip` yozing.",
        reply_markup=cancel_kb(),
    )
    await callback.answer()


@router.message(PaymentState.waiting_proof)
async def pro_payment_proof(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    proof_media_type = ""
    proof_file_id = ""
    if is_skip_text(text):
        pass
    elif message.photo:
        proof_media_type = "photo"
        proof_file_id = message.photo[-1].file_id
    elif message.document:
        proof_media_type = "document"
        proof_file_id = message.document.file_id
    else:
        await message.answer("Rasm yoki document yuboring, yoki `/skip` yozing.")
        return
    await state.update_data(
        payment_proof_media_type=proof_media_type,
        payment_proof_file_id=proof_file_id,
    )
    await state.set_state(PaymentState.waiting_comment)
    await message.answer(
        "📝 Qo'shimcha izoh yuboring.\nMasalan: Danatda aynan qaysi comment yozganingizni kiriting.\nO'tkazib yuborish uchun `/skip`.",
        reply_markup=cancel_kb(),
    )


@router.message(PaymentState.waiting_comment)
async def pro_payment_comment(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    comment = "" if is_skip_text(text) else text[:500]
    data = await state.get_data()
    payment_request_id = db.create_payment_request(
        user_tg_id=message.from_user.id,
        payment_code=format_pro_payment_code(message.from_user.id),
        proof_media_type=str(data.get("payment_proof_media_type") or ""),
        proof_file_id=str(data.get("payment_proof_file_id") or ""),
        comment=comment,
    )
    await state.clear()
    request_data = db.get_payment_request(payment_request_id)
    if request_data:
        await notify_admins_about_payment_request(message.bot, request_data)
    await message.answer(
        "✅ PRO so'rovingiz yuborildi.\nAdmin tekshiradi va tasdiqlasa PRO yoqiladi.",
        reply_markup=main_menu_kb(db.is_admin(message.from_user.id)),
    )


@router.message(F.text.in_({BTN_PRO_STATUS, "Pro holatim"}))
async def pro_status(message: Message) -> None:
    if not message.from_user:
        return
    await message.answer(build_pro_status_text(message.from_user.id))


@router.message(F.text.in_({BTN_NOTIFICATIONS, "Bildirishnomalar"}))
async def notification_settings(message: Message) -> None:
    if not message.from_user:
        return
    settings = db.get_notification_settings(message.from_user.id)
    await message.answer("🔔 Bildirishnoma sozlamalari\nQuyidagilarni yoqib yoki o'chirib boshqaring:", reply_markup=build_notification_settings_kb(settings))


@router.callback_query(F.data.startswith("notif:"))
async def notification_toggle(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    _, key = callback.data.split(":", 1)
    settings = db.toggle_notification_setting(callback.from_user.id, key)
    await callback.message.edit_reply_markup(reply_markup=build_notification_settings_kb(settings))
    await callback.answer("✅ Yangilandi")


@router.message(F.text.in_({BTN_TRENDING, "Trending"}))
async def trending_content(message: Message) -> None:
    if not message.from_user:
        return
    ok, channels = await ensure_subscription(message.from_user.id, message.bot)
    if not ok:
        await ask_for_subscription(message, channels)
        return
    items = db.list_trending_content(limit=20)
    kb = build_search_results_kb(items)
    if not kb:
        await message.answer("📭 Hozircha trending kontent topilmadi.")
        return
    await message.answer("🔥 Eng qizg'in trending kontentlar:", reply_markup=kb)


@router.message(F.text.in_({BTN_TOP_VIEWED, "Top ko'rilganlar"}))
async def top_viewed_content(message: Message) -> None:
    if not message.from_user:
        return
    ok, channels = await ensure_subscription(message.from_user.id, message.bot)
    if not ok:
        await ask_for_subscription(message, channels)
        return
    items = db.list_top_viewed_content(limit=20)
    kb = build_search_results_kb(items)
    if not kb:
        await message.answer("📭 Hozircha top kontent topilmadi.")
        return
    await message.answer("🏆 Eng ko'p ko'rilgan kontentlar:", reply_markup=kb)


@router.message(F.text.in_({BTN_CREATE_AD, "E'lon berish"}))
async def create_ad_start(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return
    if not has_active_pro(message):
        await message.answer(
            "🔒 E'lon berish faqat PRO foydalanuvchilar uchun.\n\n" + build_pro_offer_text(message.from_user.id),
            reply_markup=build_pro_purchase_kb(),
        )
        return
    await state.set_state(AdCreateState.waiting_photo)
    await message.answer(
        "🖼 E'lon uchun rasm yuboring.\nRasmsiz e'lon uchun `/skip` yozing.",
        reply_markup=cancel_kb(),
    )


@router.message(AdCreateState.waiting_photo)
async def create_ad_photo(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    photo_file_id = ""
    if is_skip_text(text):
        pass
    elif message.photo:
        photo_file_id = message.photo[-1].file_id
    else:
        await message.answer("Rasm yuboring yoki `/skip` yozing.")
        return
    await state.update_data(ad_photo_file_id=photo_file_id)
    await state.set_state(AdCreateState.waiting_title)
    await message.answer("📝 E'lon sarlavhasini kiriting:", reply_markup=cancel_kb())


@router.message(AdCreateState.waiting_title)
async def create_ad_title(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    if len(text) < 3:
        await message.answer("Sarlavha kamida 3 ta belgi bo'lsin.")
        return
    await state.update_data(ad_title=text[:120])
    await state.set_state(AdCreateState.waiting_description)
    await message.answer("📄 E'lon tavsifini kiriting:", reply_markup=cancel_kb())


@router.message(AdCreateState.waiting_description)
async def create_ad_description(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    if len(text) < 5:
        await message.answer("Tavsif kamida 5 ta belgi bo'lsin.")
        return
    await state.update_data(ad_description=text[:850])
    await state.set_state(AdCreateState.waiting_button_choice)
    await message.answer("🔘 Inline tugma kerakmi?", reply_markup=build_inline_choice_kb("adbtn"))


@router.callback_query(StateFilter(AdCreateState.waiting_button_choice), F.data.in_({"adbtn:yes", "adbtn:no"}))
async def create_ad_button_choice_inline(callback: CallbackQuery, state: FSMContext) -> None:
    if not callback.from_user or not callback.message:
        await state.clear()
        await callback.answer()
        return
    if callback.data == "adbtn:yes":
        await state.set_state(AdCreateState.waiting_button_text)
        await callback.message.answer("🔤 Tugma matnini kiriting:", reply_markup=cancel_kb())
        await callback.answer()
        return
    await state.update_data(ad_button_text="", ad_button_url="")
    data = await state.get_data()
    await send_ad_preview(
        callback.message,
        title=str(data.get("ad_title") or ""),
        description=str(data.get("ad_description") or ""),
        photo_file_id=str(data.get("ad_photo_file_id") or ""),
        footer_text="Tasdiqlash uchun quyidagi tugmani bosing.",
    )
    await state.set_state(AdCreateState.waiting_confirm)
    await callback.message.answer("E'lonni yuborishni tasdiqlaysizmi?", reply_markup=confirm_kb())
    await callback.answer()


@router.message(AdCreateState.waiting_button_choice)
async def create_ad_button_choice(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    if is_yes_text(text):
        await state.set_state(AdCreateState.waiting_button_text)
        await message.answer("🔤 Tugma matnini kiriting:", reply_markup=cancel_kb())
        return
    if not is_no_text(text):
        await message.answer("Iltimos, `Ha` yoki `Yo'q` tanlang.", reply_markup=build_inline_choice_kb("adbtn"))
        return
    await state.update_data(ad_button_text="", ad_button_url="")
    data = await state.get_data()
    await send_ad_preview(
        message,
        title=str(data.get("ad_title") or ""),
        description=str(data.get("ad_description") or ""),
        photo_file_id=str(data.get("ad_photo_file_id") or ""),
        footer_text="Tasdiqlash uchun quyidagi tugmani bosing.",
    )
    await state.set_state(AdCreateState.waiting_confirm)
    await message.answer("E'lonni yuborishni tasdiqlaysizmi?", reply_markup=confirm_kb())


@router.message(AdCreateState.waiting_button_text)
async def create_ad_button_text(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    if len(text) < 2:
        await message.answer("Tugma matni juda qisqa.")
        return
    await state.update_data(ad_button_text=text[:60])
    await state.set_state(AdCreateState.waiting_button_url)
    await message.answer("🔗 Tugma havolasini kiriting:", reply_markup=cancel_kb())


@router.message(AdCreateState.waiting_button_url)
async def create_ad_button_url(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    url = normalize_button_url(text)
    if not url:
        await message.answer("To'g'ri havola yuboring. Masalan: https://t.me/kanal")
        return
    await state.update_data(ad_button_url=url)
    data = await state.get_data()
    await send_ad_preview(
        message,
        title=str(data.get("ad_title") or ""),
        description=str(data.get("ad_description") or ""),
        photo_file_id=str(data.get("ad_photo_file_id") or ""),
        button_text=str(data.get("ad_button_text") or ""),
        button_url=str(data.get("ad_button_url") or ""),
        footer_text="Tasdiqlash uchun quyidagi tugmani bosing.",
    )
    await state.set_state(AdCreateState.waiting_confirm)
    await message.answer("E'lonni yuborishni tasdiqlaysizmi?", reply_markup=confirm_kb())


@router.message(AdCreateState.waiting_confirm)
async def create_ad_confirm(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))
        return
    if not is_confirm_text(text):
        await message.answer("Tasdiqlash uchun `✅ Tasdiqlash` ni bosing.", reply_markup=confirm_kb())
        return
    data = await state.get_data()
    ad_id = db.create_ad(
        user_tg_id=message.from_user.id,
        title=str(data.get("ad_title") or ""),
        description=str(data.get("ad_description") or ""),
        photo_file_id=str(data.get("ad_photo_file_id") or ""),
        button_text=str(data.get("ad_button_text") or ""),
        button_url=str(data.get("ad_button_url") or ""),
    )
    await state.clear()
    ad = db.get_ad(ad_id)
    if ad:
        await notify_admins_about_ad(message.bot, ad)
    await message.answer(
        "✅ E'lon moderatorga yuborildi.\nTasdiqlansa kanalga joylanadi.",
        reply_markup=main_menu_kb(db.is_admin(message.from_user.id)),
    )


@router.message(F.text.in_({BTN_MY_ADS, "E'lonlarim"}))
async def my_ads(message: Message) -> None:
    if not message.from_user:
        return
    ads = db.list_user_ads(message.from_user.id, limit=20)
    if not ads:
        await message.answer("📭 Sizda hali e'lon yo'q.\n📢 Yangi e'lon berish uchun tegishli tugmadan foydalaning.")
        return
    lines = ["🗂 E'lonlaringiz:"]
    for idx, ad in enumerate(ads, start=1):
        lines.append(
            f"{idx}. {ad.get('title') or '-'} | status: {ad.get('status') or '-'}"
        )
    await message.answer("\n".join(lines))


@router.message(F.text.in_({BTN_PRO_MANAGE, "Pro boshqarish"}))
async def pro_manage_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(ProManageState.waiting_input)
    await message.answer(
        "👑 PRO boshqarish\n\nFormat:\n`123456789 on`\n`123456789 off`\n\n📝 Oxiriga ixtiyoriy izoh ham yozishingiz mumkin.",
        reply_markup=cancel_kb(),
    )


@router.message(ProManageState.waiting_input)
async def pro_manage_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    parts = text.split()
    if len(parts) < 2 or not parts[0].isdigit() or parts[1].lower() not in {"on", "off"}:
        await message.answer("To'g'ri format: `123456789 on` yoki `123456789 off`")
        return
    user_id = int(parts[0])
    enabled = parts[1].lower() == "on"
    note = " ".join(parts[2:]).strip()
    db.set_pro_state(user_id, enabled, admin_id=message.from_user.id, note=note)
    await state.clear()
    if db.get_notification_settings(user_id).get("pro_updates"):
        try:
            await message.bot.send_message(
                chat_id=user_id,
                text="👑 Sizga PRO yoqildi!" if enabled else "🔒 Sizning PRO o'chirildi.",
            )
        except (TelegramBadRequest, TelegramForbiddenError):
            pass
    await message.answer(
        "✅ PRO yoqildi." if enabled else "✅ PRO o'chirildi.",
        reply_markup=admin_menu_kb(),
    )


@router.message(F.text.in_({BTN_PRO_PRICE, "Pro narxi"}))
async def pro_price_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    current = db.get_bot_settings()["pro_price_text"]
    await state.set_state(ProPriceState.waiting_price)
    await message.answer(f"💰 Joriy PRO narxi: {current}\n\n✍️ Yangi narx matnini yuboring:", reply_markup=cancel_kb())


@router.message(ProPriceState.waiting_price)
async def pro_price_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if len(text) < 3:
        await message.answer("Narx matni juda qisqa.")
        return
    db.set_pro_price_text(text)
    await state.clear()
    await message.answer("✅ PRO narxi muvaffaqiyatli yangilandi.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_PRO_DURATION, "Pro muddati"}))
async def pro_duration_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    current = db.get_bot_settings()["pro_duration_days"]
    await state.set_state(ProDurationState.waiting_days)
    await message.answer(
        f"⏳ Joriy PRO muddati: {current} kun\n\n✍️ Yangi kun sonini yuboring:",
        reply_markup=cancel_kb(),
    )


@router.message(ProDurationState.waiting_days)
async def pro_duration_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text.isdigit():
        await message.answer("⚠️ Faqat son yuboring. Masalan: `30`")
        return
    days = int(text)
    if days < 1 or days > 3650:
        await message.answer("⚠️ Muddat 1 kundan 3650 kungacha bo'lishi kerak.")
        return
    db.set_pro_duration_days(days)
    await state.clear()
    await message.answer(f"✅ PRO muddati {days} kunga yangilandi.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_PRO_REQUESTS, "Pro so'rovlar"}))
async def pro_requests(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    requests = db.list_pending_payment_requests(limit=15)
    if not requests:
        await message.answer("📭 Kutilayotgan PRO so'rovlar yo'q.")
        return
    for request in requests:
        text = (
            "💳 PRO so'rovi\n"
            f"👤 User ID: {request.get('user_tg_id')}\n"
            f"🔑 Kod: {request.get('payment_code') or '-'}\n"
            f"📝 Izoh: {request.get('comment') or '-'}"
        )
        proof_media_type = str(request.get("proof_media_type") or "")
        proof_file_id = str(request.get("proof_file_id") or "")
        markup = build_payment_request_review_kb(str(request.get("id") or ""))
        if proof_media_type and proof_file_id:
            await send_stored_media(
                message,
                proof_media_type,
                proof_file_id,
                caption=text,
                reply_markup=markup,
            )
        else:
            await message.answer(text, reply_markup=markup)


@router.callback_query(F.data.startswith("proreq:"))
async def pro_request_review(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    _, action, request_id = callback.data.split(":", 2)
    request = db.get_payment_request(request_id)
    if not request:
        await callback.answer("So'rov topilmadi", show_alert=True)
        return
    user_id = int(request.get("user_tg_id") or 0)
    if action == "approve":
        db.update_payment_request_status(request_id, "approved", reviewed_by=callback.from_user.id)
        db.set_pro_state(user_id, True, admin_id=callback.from_user.id, note="To'lov tasdiqlandi")
        if db.get_notification_settings(user_id).get("pro_updates"):
            try:
                await callback.bot.send_message(chat_id=user_id, text="✅ To'lov tasdiqlandi. PRO faollashtirildi!")
            except (TelegramBadRequest, TelegramForbiddenError):
                pass
        await callback.answer("✅ PRO yoqildi")
    else:
        db.update_payment_request_status(request_id, "rejected", reviewed_by=callback.from_user.id)
        if db.get_notification_settings(user_id).get("pro_updates"):
            try:
                await callback.bot.send_message(chat_id=user_id, text="❌ PRO so'rovingiz rad etildi. Admin bilan bog'laning.")
            except (TelegramBadRequest, TelegramForbiddenError):
                pass
        await callback.answer("❌ Rad etildi")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass


@router.message(F.text.in_({BTN_AD_CHANNELS, "E'lon kanalari"}))
async def ad_channels_menu(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await message.answer("📡 E'lon kanalari boshqaruvi\nKanal qo'shish, ko'rish va o'chirish shu yerda.", reply_markup=build_ad_manage_kb())


@router.callback_query(F.data == "adch:add")
async def ad_channel_add_start(callback: CallbackQuery, state: FSMContext) -> None:
    if not callback.from_user or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    await state.set_state(AddAdChannelState.waiting_input)
    if callback.message:
        await callback.message.answer("Kanal username yoki ID yuboring: @kanal yoki -100...", reply_markup=cancel_kb())
    await callback.answer()


@router.callback_query(F.data == "adch:list")
async def ad_channel_list(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    channels = db.list_ad_channels()
    if not channels:
        await callback.message.answer("📭 E'lon kanalari yo'q.")
    else:
        lines = ["📡 E'lon kanalari:"]
        for idx, channel in enumerate(channels, start=1):
            lines.append(f"{idx}. {channel.get('title') or channel.get('channel_ref')} ({channel.get('channel_ref')})")
        await callback.message.answer("\n".join(lines))
    await callback.answer()


@router.callback_query(F.data == "adch:delete_menu")
async def ad_channel_delete_menu(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    channels = db.list_ad_channels()
    if not channels:
        await callback.message.answer("📭 O'chirish uchun kanal yo'q.")
        await callback.answer()
        return
    rows = [
        [InlineKeyboardButton(text=f"🗑 {channel.get('title') or channel.get('channel_ref')}", callback_data=f"adch:del:{channel.get('id')}")]
        for channel in channels
        if channel.get("id")
    ]
    await callback.message.answer("O'chiriladigan kanalni tanlang:", reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
    await callback.answer()


@router.callback_query(F.data.startswith("adch:del:"))
async def ad_channel_delete(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    channel_id = callback.data.split(":", 2)[2]
    removed = db.remove_ad_channel(channel_id)
    await callback.answer("✅ O'chirildi" if removed else "❌ Topilmadi")
    if removed:
        await callback.message.edit_reply_markup(reply_markup=None)


@router.message(AddAdChannelState.waiting_input)
async def ad_channel_add_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    channel_ref = normalize_channel_ref_input(text)
    if not channel_ref:
        await message.answer("To'g'ri kanal username yoki ID yuboring.")
        return
    title = channel_ref
    try:
        chat = await message.bot.get_chat(channel_ref)
        title = str(getattr(chat, "title", "") or getattr(chat, "username", "") or channel_ref)
    except TelegramBadRequest:
        pass
    created = db.add_ad_channel(channel_ref, title=title)
    await state.clear()
    await message.answer("✅ E'lon kanali qo'shildi." if created else "ℹ️ Bu kanal allaqachon mavjud.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_ADS, "E'lonlar"}))
async def ads_review(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    ads = db.list_pending_ads(limit=15)
    if not ads:
        await message.answer("📭 Kutilayotgan e'lonlar yo'q.")
        return
    for ad in ads:
        await send_ad_preview(
            message,
            title=str(ad.get("title") or ""),
            description=str(ad.get("description") or ""),
            photo_file_id=str(ad.get("photo_file_id") or ""),
            button_text=str(ad.get("button_text") or ""),
            button_url=str(ad.get("button_url") or ""),
            footer_text=f"👤 User ID: {ad.get('user_tg_id')}\n🆔 ID: {ad.get('id')}",
            reply_markup=build_ad_review_kb(str(ad.get("id") or "")),
        )


@router.callback_query(F.data.startswith("ad:channel:"))
async def ad_choose_channel(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    ad_id = callback.data.split(":", 2)[2]
    channels = db.list_ad_channels()
    if not channels:
        await callback.answer("Avval e'lon kanalini qo'shing", show_alert=True)
        return
    await callback.message.answer(
        "Qaysi kanalga joylansin?",
        reply_markup=build_ad_channel_pick_kb(ad_id, channels),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("ad:post:"))
async def ad_post_to_channel(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    _, _, ad_id, channel_id = callback.data.split(":", 3)
    ad = db.get_ad(ad_id)
    channel = db.get_ad_channel(channel_id)
    if not ad or not channel:
        await callback.answer("Ma'lumot topilmadi", show_alert=True)
        return
    channel_ref = str(channel.get("channel_ref") or "")
    try:
        await post_ad_to_channel(callback.bot, channel_ref, ad)
    except (TelegramBadRequest, TelegramForbiddenError) as exc:
        await callback.answer("Kanalga joylab bo'lmadi", show_alert=True)
        if callback.message:
            await callback.message.answer(f"❌ Joylashda xatolik: {exc}")
        return
    db.update_ad_status(ad_id, "posted", reviewed_by=callback.from_user.id, channel_ref=channel_ref)
    user_id = int(ad.get("user_tg_id") or 0)
    if db.get_notification_settings(user_id).get("ads_updates"):
        try:
            await callback.bot.send_message(
                chat_id=user_id,
                text=f"✅ E'loningiz {channel.get('title') or channel_ref} kanaliga joylandi.",
            )
        except (TelegramBadRequest, TelegramForbiddenError):
            pass
    await callback.answer("✅ Kanalga joylandi")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass


@router.callback_query(F.data.startswith("ad:reject:"))
async def ad_reject(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    ad_id = callback.data.split(":", 2)[2]
    ad = db.get_ad(ad_id)
    if not ad:
        await callback.answer("E'lon topilmadi", show_alert=True)
        return
    db.update_ad_status(ad_id, "rejected", reviewed_by=callback.from_user.id)
    user_id = int(ad.get("user_tg_id") or 0)
    if db.get_notification_settings(user_id).get("ads_updates"):
        try:
            await callback.bot.send_message(chat_id=user_id, text="❌ E'loningiz moderator tomonidan rad etildi.")
        except (TelegramBadRequest, TelegramForbiddenError):
            pass
    await callback.answer("❌ Rad etildi")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass


@router.callback_query(F.data == "ad:none")
async def ad_none(callback: CallbackQuery) -> None:
    await callback.answer("Avval kanal qo'shing.", show_alert=True)


@router.message(F.text.in_({BTN_SUBS, "Majburiy obuna"}))
async def mandatory_subscriptions_menu(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await message.answer("📢 Majburiy obuna boshqaruvi", reply_markup=sub_manage_kb())


@router.callback_query(F.data == "sub_add")
async def add_sub_start(callback: CallbackQuery, state: FSMContext) -> None:
    if not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    prompt = (
        "📌 Kanal ma'lumotini quyidagicha yuboring:\n"
        "1) @kanal_username\n"
        "2) -1001234567890|https://t.me/+invite_link\n\n"
        "Private kanal uchun avval ID, keyin alohida invite link yuborish ham mumkin.\n"
        "Yoki kanaldan 1 ta postni forward qiling.\n"
        f"Bekor qilish: {BTN_CANCEL}"
    )
    await state.set_state(AddChannelState.waiting_input)
    await callback.message.answer(prompt, reply_markup=cancel_kb())
    await callback.answer()


@router.message(AddChannelState.waiting_input)
async def add_sub_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return

    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return

    join_link: str | None = None
    channel_ref: str | None = None
    title: str | None = None
    state_data = await state.get_data()
    pending_channel_ref = str(state_data.get("pending_channel_ref") or "").strip()
    pending_channel_title = str(state_data.get("pending_channel_title") or "").strip() or None

    # 1) Forwarded postdan kanalni avtomatik olish
    forward_chat = getattr(message, "forward_from_chat", None)
    if not forward_chat:
        origin = getattr(message, "forward_origin", None)
        if isinstance(origin, MessageOriginChannel):
            forward_chat = origin.chat
    if forward_chat:
        channel_ref = str(forward_chat.id)
        title = forward_chat.title or str(forward_chat.id)
        if getattr(forward_chat, "username", None):
            join_link = f"https://t.me/{forward_chat.username}"
    else:
        # 2) Matn formatidan olish
        if "|" in text:
            left, right = text.split("|", 1)
            channel_ref = normalize_channel_ref_input(left)
            join_link = right.strip() or None
        else:
            channel_ref = normalize_channel_ref_input(text)
            if not channel_ref:
                invite_link_only = normalize_invite_link_input(text)
                if invite_link_only and pending_channel_ref:
                    channel_ref = pending_channel_ref
                    join_link = invite_link_only
                    title = pending_channel_title

    if not channel_ref:
        if normalize_invite_link_input(text):
            await message.answer(
                "⚠️ Invite linkni alohida yuborish uchun avval kanal ID sini yuboring.\n"
                "Masalan:\n"
                "1) -1001234567890\n"
                "2) https://t.me/+invite_link\n\n"
                "Yoki bitta xabarda: -1001234567890|https://t.me/+invite_link"
            )
            return
        await message.answer(
            "⚠️ Noto'g'ri format.\n"
            "To'g'ri formatlar:\n"
            "1) @kanal_username\n"
            "2) -1001234567890|https://t.me/+invite_link\n"
            "3) Kanal postini forward qilish"
        )
        return

    chat = None
    try:
        chat = await message.bot.get_chat(channel_ref)
        title = title or chat.title or channel_ref
    except ClientDecodeError:
        # Some aiogram versions fail to parse new Telegram ChatFullInfo fields.
        logging.warning("get_chat decode xatoligi: %s. Fallback rejimda davom etiladi.", channel_ref)
        title = title or channel_ref
    except (TelegramBadRequest, TelegramForbiddenError):
        await message.answer(
            "❌ Kanal topilmadi yoki bot kanalga kira olmayapti.\n"
            "Kanalni tekshiring va botni kanalga admin qiling."
        )
        return

    try:
        me = await message.bot.get_me()
        me_member = await message.bot.get_chat_member(chat_id=channel_ref, user_id=me.id)
        if me_member.status not in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR}:
            await message.answer("⚠️ Bot majburiy obunani tekshirishi uchun kanalda admin bo'lishi shart.")
            return
        if (
            channel_ref.lstrip("-").isdigit()
            and me_member.status == ChatMemberStatus.ADMINISTRATOR
            and not getattr(me_member, "can_invite_users", False)
        ):
            await message.answer(
                "ℹ️ Eslatma: zayafka yuborilgani bilan obunani o'tkazish uchun botda "
                "'Invite users via link' (join requestlarni ko'rish) huquqi bo'lishi kerak."
            )
    except (TelegramBadRequest, TelegramForbiddenError):
        await message.answer(
            "❌ Bot bu kanalda a'zolar obunasini tekshira olmayapti.\n"
            "Botni kanalga admin qilib, qaytadan qo'shing."
        )
        return

    if not join_link:
        username = getattr(chat, "username", None) if chat is not None else None
        invite_link = getattr(chat, "invite_link", None) if chat is not None else None
        if username:
            join_link = f"https://t.me/{username}"
        elif invite_link:
            join_link = invite_link
        elif channel_ref.startswith("@"):
            join_link = f"https://t.me/{channel_ref[1:]}"

    if channel_ref.lstrip("-").isdigit() and not join_link:
        await state.update_data(
            pending_channel_ref=channel_ref,
            pending_channel_title=title or channel_ref,
        )
        await message.answer(
            "⚠️ Private kanal uchun join link kerak.\n"
            "Format: -1001234567890|https://t.me/+invite_link\n"
            "Yoki endi faqat invite link yuboring: https://t.me/+invite_link"
        )
        return

    created = db.add_required_channel(channel_ref=channel_ref, title=title, join_link=join_link)
    await state.clear()
    if created:
        await message.answer(
            f"✅ Kanal qo'shildi: {title}",
            reply_markup=admin_menu_kb(),
        )
    else:
        await message.answer(
            "ℹ️ Bu kanal allaqachon ro'yxatda.",
            reply_markup=admin_menu_kb(),
        )


@router.callback_query(F.data == "sub_list")
async def list_subscriptions(callback: CallbackQuery) -> None:
    if not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    channels = db.get_required_channels()
    if not channels:
        await callback.message.answer("📭 Majburiy obuna ro'yxati hozircha bo'sh.")
        await callback.answer()
        return

    lines = ["📢 Majburiy obuna kanallari:"]
    for ch in channels:
        title = ch["title"] or ch["channel_ref"]
        lines.append(f"• {title} ({ch['channel_ref']})")
    await callback.message.answer("\n".join(lines))
    await callback.answer()


@router.callback_query(F.data == "sub_delete_menu")
async def delete_subscriptions_menu(callback: CallbackQuery) -> None:
    if not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    channels = db.get_required_channels()
    if not channels:
        await callback.message.answer("📭 O'chirish uchun kanal topilmadi.")
        await callback.answer()
        return

    builder = InlineKeyboardBuilder()
    for ch in channels:
        title = ch["title"] or ch["channel_ref"]
        builder.button(text=f"❌ {title}", callback_data=f"sub_del:{ch['channel_ref']}")
    builder.adjust(1)
    await callback.message.answer("🗑 O'chiriladigan kanalni tanlang:", reply_markup=builder.as_markup())
    await callback.answer()


@router.callback_query(F.data.startswith("sub_del:"))
async def delete_channel(callback: CallbackQuery) -> None:
    if not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    _, channel_ref = callback.data.split(":", 1)
    deleted = db.remove_required_channel(channel_ref)
    if deleted:
        await callback.message.answer("✅ Kanal o'chirildi.")
    else:
        await callback.message.answer("⚠️ Kanal topilmadi.")
    await callback.answer()


@router.callback_query(F.data.startswith("serial_ep:"))
async def send_serial_episode(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    parts = callback.data.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2].isdigit():
        await callback.answer("Noto'g'ri so'rov")
        return

    serial_id = parts[1]
    episode_number = int(parts[2])

    ok, channels = await ensure_subscription(callback.from_user.id, callback.bot)
    if not ok:
        await callback.message.answer(
            "❗ Avval barcha majburiy kanallarga obuna bo'ling.",
            reply_markup=build_subscribe_keyboard(channels),
        )
        await callback.answer("Obuna kerak")
        return

    serial = db.get_serial(serial_id)
    episode = db.get_serial_episode(serial_id, episode_number)
    if not serial or not episode:
        await callback.answer("Qism topilmadi", show_alert=True)
        return

    episodes = db.list_serial_episodes(serial_id)
    episode_numbers = [row["episode_number"] for row in episodes]
    nav_kb = build_episode_navigation_kb(serial_id, episode_numbers, episode_number)

    caption = build_movie_caption(serial["title"], f"{episode_number}-qism")
    try:
        await send_stored_media(
            callback.message,
            media_type=episode["media_type"],
            file_id=episode["file_id"],
            caption=caption if caption else None,
            reply_markup=nav_kb,
        )
        db.increment_serial_downloads(serial_id)
        await callback.answer()
    except (TelegramBadRequest, TelegramForbiddenError, ValueError):
        await callback.answer("Xatolik", show_alert=True)
        await callback.message.answer(
            "Qismni yuborishda xatolik. Admin media faylni qayta yuklasin."
        )


@router.message(F.text.in_({BTN_ADD_MOVIE, "Kino qo'shish"}))
async def add_movie_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(AddMovieState.waiting_code)
    suggested_codes = ", ".join(generate_missing_numeric_codes(db.get_all_codes(), 5))
    await message.answer(
        "🎬 Yangi kino kodini yuboring:\n"
        f"💡 Bazada yo'q 5 ta kod: {suggested_codes}",
        reply_markup=cancel_kb(),
    )


@router.message(AddMovieState.waiting_code)
async def add_movie_code(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text:
        await message.answer("Kod bo'sh bo'lmasin.")
        return
    if db.code_exists(text):
        await message.answer("⚠️ Bu kod allaqachon band. Boshqa kod yuboring.")
        return
    await state.update_data(code=text)
    await state.set_state(AddMovieState.waiting_title)
    await message.answer("📝 Kino nomini yuboring:")


@router.message(AddMovieState.waiting_title)
async def add_movie_title(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text:
        await message.answer("Nomi bo'sh bo'lmasin.")
        return
    await state.update_data(title=text)
    await state.set_state(AddMovieState.waiting_description)
    await message.answer(
        "📄 Kino tavsifi/caption yuboring (video ostida chiqadi).\n"
        "Kerak bo'lmasa: `-` yuboring."
    )


@router.message(AddMovieState.waiting_description)
async def add_movie_description(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    description = "" if text == "-" else text
    await state.update_data(description=description)
    await state.set_state(AddMovieState.waiting_metadata)
    await message.answer(
        "🏷 Metadata yuboring (format: yil|sifat|janr1,janr2).\n"
        "Masalan: 2024|1080p|action,drama\n"
        "Agar kerak bo'lmasa: -"
    )


@router.message(AddMovieState.waiting_metadata)
async def add_movie_metadata(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return

    metadata = parse_metadata_input(text)
    if metadata is None:
        await message.answer(
            "⚠️ Format noto'g'ri. To'g'ri format: yil|sifat|janr1,janr2\n"
            "Masalan: 2024|1080p|action,drama\n"
            "Yoki: -"
        )
        return

    await state.update_data(
        year=metadata["year"],
        quality=metadata["quality"],
        genres=metadata["genres"],
    )
    await state.set_state(AddMovieState.waiting_media)
    await message.answer(
        "📤 Endi media yuboring:\n"
        "• video / document / photo\n"
        "• yoki file_id / link matn\n\n"
        "ℹ️ Telegram post link yuborsangiz kanal captioni olinmaydi,\n"
        "siz yozgan caption chiqadi."
    )


@router.message(AddMovieState.waiting_media)
async def add_movie_media(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return

    if is_cancel_text(message.text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return

    media = parse_message_media(message)
    if not media:
        await message.answer("Noto'g'ri format. Video/document/photo yoki matn yuboring.")
        return
    media_type, file_id = media
    preview_photo_file_id = extract_preview_photo_file_id(message)
    preview_media_type = "photo" if preview_photo_file_id else ""
    preview_file_id = preview_photo_file_id or ""

    data = await state.get_data()
    movie = Movie(
        code=data["code"],
        title=data["title"],
        description=data.get("description", ""),
        media_type=media_type,
        file_id=file_id,
        year=data.get("year"),
        quality=str(data.get("quality") or ""),
        genres=[str(g) for g in data.get("genres", []) if str(g).strip()],
        preview_media_type=preview_media_type,
        preview_file_id=preview_file_id,
    )
    created = db.add_movie(movie)
    await state.clear()
    if created:
        saved_movie = db.get_movie(movie.code)
        delivered, failed = await notify_requesters_for_content(
            bot=message.bot,
            content_type="movie",
            content_ref=str((saved_movie or {}).get("id") or movie.code),
            code=movie.code,
            title=movie.title,
            movie=saved_movie,
        )
        pro_delivered = 0
        pro_failed = 0
        if saved_movie:
            pro_delivered, pro_failed = await notify_new_content_to_pro_users(
                message.bot,
                "movie",
                str(saved_movie.get("id") or ""),
                movie.title,
                movie.code,
            )
        note = ""
        if delivered or failed:
            note = f"\n📣 So'rov yuborganlarga xabar: {delivered} ta yetkazildi, {failed} ta xato."
        if pro_delivered or pro_failed:
            note += f"\n👑 PRO xabar: {pro_delivered} ta yetkazildi, {pro_failed} ta xato."
        await message.answer(
            f"✅ Kino muvaffaqiyatli saqlandi!{note}",
            reply_markup=admin_menu_kb(),
        )
    else:
        await message.answer("⚠️ Bu kod allaqachon mavjud.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_ADD_SERIAL, "Serial qo'shish"}))
async def add_serial_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(AddSerialState.waiting_code)
    suggested_codes = ", ".join(generate_missing_numeric_codes(db.get_all_codes(), 5))
    await message.answer(
        "📺 Yangi serial kodi yuboring:\n"
        f"💡 Bazada yo'q 5 ta kod: {suggested_codes}",
        reply_markup=cancel_kb(),
    )


@router.message(AddSerialState.waiting_code)
async def add_serial_code(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text:
        await message.answer("Kod bo'sh bo'lmasin.")
        return
    if db.code_exists(text):
        await message.answer("⚠️ Bu kod allaqachon band. Boshqa kod yuboring.")
        return
    await state.update_data(code=text)
    await state.set_state(AddSerialState.waiting_title)
    await message.answer("📝 Serial nomini yuboring:")


@router.message(AddSerialState.waiting_title)
async def add_serial_title(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text:
        await message.answer("Nomi bo'sh bo'lmasin.")
        return
    await state.update_data(title=text)
    await state.set_state(AddSerialState.waiting_description)
    await message.answer(
        "📄 Serial tavsifini yuboring.\n"
        "Tavsif kerak bo'lmasa: `-` yuboring."
    )


@router.message(AddSerialState.waiting_description)
async def add_serial_description(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    description = "" if text == "-" else text
    await state.update_data(
        description=description,
    )
    await state.set_state(AddSerialState.waiting_metadata)
    await message.answer(
        "🏷 Metadata yuboring (format: yil|sifat|janr1,janr2).\n"
        "Masalan: 2024|1080p|action,drama\n"
        "Agar kerak bo'lmasa: -"
    )


@router.message(AddSerialState.waiting_metadata)
async def add_serial_metadata(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return

    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return

    metadata = parse_metadata_input(text)
    if metadata is None:
        await message.answer(
            "⚠️ Format noto'g'ri. To'g'ri format: yil|sifat|janr1,janr2\n"
            "Masalan: 2024|1080p|action,drama\n"
            "Yoki: -"
        )
        return

    data = await state.get_data()
    code = data.get("code")
    title = data.get("title")
    description = str(data.get("description") or "")
    if not code or not title:
        await state.clear()
        await message.answer("⚠️ Sessiya topilmadi, qaytadan boshlang.", reply_markup=admin_menu_kb())
        return

    serial_id = db.add_serial(
        code=str(code),
        title=str(title),
        description=description,
        year=metadata["year"],
        quality=metadata["quality"],
        genres=metadata["genres"],
    )
    if serial_id is None:
        await state.clear()
        await message.answer("⚠️ Bu kod allaqachon mavjud.", reply_markup=admin_menu_kb())
        return

    await state.update_data(
        year=metadata["year"],
        quality=metadata["quality"],
        genres=metadata["genres"],
        serial_id=serial_id,
        next_episode=1,
        episodes_added=0,
    )
    await state.set_state(AddSerialState.waiting_episode)
    await message.answer(
        "🎬 Endi 1-qismni yuboring.\n"
        "Video/document/photo yoki file_id/link yuborishingiz mumkin.\n"
        f"Yakunlash uchun: {BTN_SERIAL_DONE}",
        reply_markup=serial_upload_kb(),
    )


@router.message(AddSerialState.waiting_episode)
async def add_serial_episode(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return

    text = (message.text or "").strip()
    data = await state.get_data()
    serial_id_raw = data.get("serial_id")
    next_episode_raw = data.get("next_episode", 1)
    episodes_added_raw = data.get("episodes_added", 0)
    if serial_id_raw is None:
        await state.clear()
        await message.answer("⚠️ Sessiya topilmadi, qaytadan boshlang.", reply_markup=admin_menu_kb())
        return

    serial_id = str(serial_id_raw)
    next_episode = int(next_episode_raw)
    episodes_added = int(episodes_added_raw)

    if is_cancel_text(text):
        db.delete_serial(serial_id)
        await state.clear()
        await message.answer("❌ Bekor qilindi. Serial saqlanmadi.", reply_markup=admin_menu_kb())
        return

    if is_serial_done_text(text):
        if episodes_added == 0:
            await message.answer("Kamida 1 ta qism qo'shing yoki bekor qiling.")
            return
        await state.set_state(AddSerialState.waiting_preview_media)
        await message.answer(
            "🖼 Endi preview uchun rasm/video yuboring.\n"
            "O'tkazib yuborish uchun: -",
            reply_markup=cancel_kb(),
        )
        return

    media = parse_message_media(message)
    if not media:
        await message.answer(
            "Noto'g'ri format. Video/document/photo yoki matn (file_id/link) yuboring.",
            reply_markup=serial_upload_kb(),
        )
        return
    media_type, file_id = media

    created = db.add_serial_episode(serial_id, next_episode, media_type, file_id)
    if not created:
        await message.answer("⚠️ Qismni saqlab bo'lmadi, qayta urinib ko'ring.")
        return

    await state.update_data(
        next_episode=next_episode + 1,
        episodes_added=episodes_added + 1,
    )
    await message.answer(
        f"✅ {next_episode}-qism saqlandi.\n"
        f"➡️ Endi {next_episode + 1}-qismni yuboring yoki {BTN_SERIAL_DONE} tugmasini bosing.",
        reply_markup=serial_upload_kb(),
    )


@router.message(AddSerialState.waiting_preview_media)
async def add_serial_preview_media(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return

    text = (message.text or "").strip()
    if is_cancel_text(text):
        data = await state.get_data()
        serial_id = str(data.get("serial_id") or "").strip()
        if serial_id:
            db.delete_serial(serial_id)
        await state.clear()
        await message.answer("❌ Bekor qilindi. Serial saqlanmadi.", reply_markup=admin_menu_kb())
        return

    if text == "-":
        data = await state.get_data()
        episodes_added = int(data.get("episodes_added", 0))
        serial_id = str(data.get("serial_id") or "").strip()
        serial = db.get_serial(serial_id) if serial_id else None
        notify_text = ""
        if serial:
            delivered, failed = await notify_requesters_for_content(
                bot=message.bot,
                content_type="serial",
                content_ref=serial_id,
                code=str(serial.get("code") or ""),
                title=str(serial.get("title") or "Serial"),
                serial_id=serial_id,
            )
            if delivered or failed:
                notify_text = f"\n📣 So'rov yuborganlarga xabar: {delivered} ta yetkazildi, {failed} ta xato."
            pro_delivered, pro_failed = await notify_new_content_to_pro_users(
                message.bot,
                "serial",
                serial_id,
                str(serial.get("title") or "Serial"),
                str(serial.get("code") or ""),
            )
            if pro_delivered or pro_failed:
                notify_text += f"\n👑 PRO xabar: {pro_delivered} ta yetkazildi, {pro_failed} ta xato."
        await state.clear()
        await message.answer(
            f"✅ Serial muvaffaqiyatli saqlandi!\n🎞 Jami qismlar: {episodes_added}{notify_text}",
            reply_markup=admin_menu_kb(),
        )
        return

    media = parse_message_media(message)
    if not media:
        await message.answer(
            "Noto'g'ri format. Preview uchun video/document/photo yoki file_id/link yuboring.\n"
            "O'tkazib yuborish uchun: -",
            reply_markup=cancel_kb(),
        )
        return
    media_type, file_id = media
    preview_photo_file_id = extract_preview_photo_file_id(message) or ""

    await state.update_data(
        preview_media_type=media_type,
        preview_file_id=file_id,
        preview_photo_file_id=preview_photo_file_id,
    )
    await state.set_state(AddSerialState.waiting_publish_channel)
    await message.answer(
        "📣 Endi post joylanadigan kanalni yuboring.\n"
        "Format: @kanal_username yoki -1001234567890\n"
        "Kanalga joylamaslik uchun: -",
        reply_markup=cancel_kb(),
    )


@router.message(AddSerialState.waiting_publish_channel)
async def add_serial_publish_channel(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return

    text = (message.text or "").strip()
    if is_cancel_text(text):
        data = await state.get_data()
        serial_id = str(data.get("serial_id") or "").strip()
        if serial_id:
            db.delete_serial(serial_id)
        await state.clear()
        await message.answer("❌ Bekor qilindi. Serial saqlanmadi.", reply_markup=admin_menu_kb())
        return

    data = await state.get_data()
    episodes_added = int(data.get("episodes_added", 0))
    serial_id = str(data.get("serial_id") or "").strip()
    title = str(data.get("title") or "").strip()
    description = str(data.get("description") or "").strip()
    preview_media_type = str(data.get("preview_media_type") or "").strip()
    preview_file_id = str(data.get("preview_file_id") or "").strip()
    preview_photo_file_id = str(data.get("preview_photo_file_id") or "").strip()

    if not serial_id:
        await state.clear()
        await message.answer("⚠️ Sessiya topilmadi, qaytadan boshlang.", reply_markup=admin_menu_kb())
        return

    if text == "-":
        if preview_media_type and preview_file_id:
            db.update_serial_preview(
                serial_id,
                preview_media_type,
                preview_file_id,
                preview_photo_file_id=preview_photo_file_id,
            )
        notify_text = ""
        serial = db.get_serial(serial_id) if serial_id else None
        if serial:
            delivered, failed = await notify_requesters_for_content(
                bot=message.bot,
                content_type="serial",
                content_ref=serial_id,
                code=str(serial.get("code") or ""),
                title=str(serial.get("title") or "Serial"),
                serial_id=serial_id,
            )
            if delivered or failed:
                notify_text = f"\n📣 So'rov yuborganlarga xabar: {delivered} ta yetkazildi, {failed} ta xato."
        await state.clear()
        await message.answer(
            f"✅ Serial muvaffaqiyatli saqlandi!\n🎞 Jami qismlar: {episodes_added}{notify_text}",
            reply_markup=admin_menu_kb(),
        )
        return

    channel_ref = normalize_channel_ref_input(text)
    if not channel_ref:
        await message.answer("⚠️ Noto'g'ri kanal formati. Masalan: @kanal_username yoki -1001234567890")
        return

    if not preview_media_type or not preview_file_id:
        await state.clear()
        await message.answer("⚠️ Preview media topilmadi.", reply_markup=admin_menu_kb())
        return

    try:
        chat = await message.bot.get_chat(channel_ref)
        me = await message.bot.get_me()
        if not me.username:
            await state.clear()
            await message.answer("❌ Bot username topilmadi. Deep link yaratib bo'lmadi.", reply_markup=admin_menu_kb())
            return

        payload = f"s_{serial_id}"
        deeplink = f"https://t.me/{me.username}?start={payload}"
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="📥 Yuklab olish", url=deeplink)]]
        )

        caption_lines = [f"🎬 {title or 'Serial'}"]
        if description:
            caption_lines.append(description)
        meta = format_meta_line(
            data.get("year") if isinstance(data.get("year"), int) else None,
            str(data.get("quality") or ""),
            [str(g) for g in data.get("genres", []) if str(g).strip()],
        )
        if meta:
            caption_lines.append(meta)
        caption_lines.append(f"🎞 Jami qismlar: {episodes_added}")
        caption_lines.append("Tomosha 👇")
        caption = "\n\n".join(caption_lines)

        await send_media_to_chat(
            bot=message.bot,
            chat_ref=channel_ref,
            media_type=preview_media_type,
            file_id=preview_file_id,
            caption=caption,
            reply_markup=keyboard,
        )
        db.update_serial_preview(
            serial_id,
            preview_media_type,
            preview_file_id,
            preview_photo_file_id=preview_photo_file_id,
        )
        notify_text = ""
        delivered, failed = await notify_requesters_for_content(
            bot=message.bot,
            content_type="serial",
            content_ref=serial_id,
            code=str(data.get("code") or ""),
            title=title or "Serial",
            serial_id=serial_id,
        )
        if delivered or failed:
            notify_text = f"\n📣 So'rov yuborganlarga xabar: {delivered} ta yetkazildi, {failed} ta xato."
        await state.clear()
        await message.answer(
            f"✅ Serial saqlandi va kanalga joylandi: {chat.title or channel_ref}{notify_text}",
            reply_markup=admin_menu_kb(),
        )
    except (TelegramBadRequest, TelegramForbiddenError, ValueError):
        await message.answer(
            "❌ Kanalga joylab bo'lmadi.\n"
            "Kanalni tekshiring va botni kanalga admin qiling."
        )


@router.message(F.text.in_({BTN_DEL_MOVIE, "Kino o'chirish"}))
async def delete_movie_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(DeleteMovieState.waiting_code)
    await message.answer("🗑 O'chirish uchun kino yoki serial kodini yuboring:", reply_markup=cancel_kb())


@router.message(DeleteMovieState.waiting_code)
async def delete_movie_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text:
        await message.answer("Kod yuboring.")
        return
    deleted_types: list[str] = []
    if db.delete_movie(text):
        deleted_types.append("kino")

    serial = db.get_serial_by_code(text)
    if serial and db.delete_serial(serial["id"]):
        deleted_types.append("serial")

    await state.clear()
    if deleted_types:
        if len(deleted_types) == 1:
            deleted_name = "Kino" if deleted_types[0] == "kino" else "Serial"
            await message.answer(f"✅ {deleted_name} o'chirildi.", reply_markup=admin_menu_kb())
        else:
            await message.answer("✅ Kino va serial o'chirildi.", reply_markup=admin_menu_kb())
    else:
        await message.answer("❌ Bu kod bo'yicha kino yoki serial topilmadi.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_EDIT_CONTENT, "Kontent tahrirlash"}))
async def edit_content_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(EditContentState.waiting_code)
    await message.answer("✏️ Tahrirlash uchun kino yoki serial kodini yuboring:", reply_markup=cancel_kb())


@router.message(EditContentState.waiting_code)
async def edit_content_code(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text:
        await message.answer("Kod yuboring.")
        return

    code = normalize_code(text)
    movie = db.get_movie(code)
    if movie:
        current_title = str(movie.get("title") or "")
        current_description = str(movie.get("description") or "")
        await state.update_data(
            edit_type="movie",
            edit_code=code,
            movie_title=current_title,
            movie_description=current_description,
            movie_media_type=str(movie.get("media_type") or ""),
            movie_file_id=str(movie.get("file_id") or ""),
            movie_year=movie.get("year"),
            movie_quality=str(movie.get("quality") or ""),
            movie_genres=[str(g) for g in movie.get("genres", []) if str(g).strip()],
            movie_preview_media_type=str(movie.get("preview_media_type") or ""),
            movie_preview_file_id=str(movie.get("preview_file_id") or ""),
        )
        await state.set_state(EditContentState.waiting_movie_title)
        await message.answer(
            "🎬 Kino topildi.\n"
            f"Joriy nom: {current_title or '-'}\n"
            "Yangi nom yuboring.\n"
            "O'zgartirmaslik uchun: -"
        )
        return

    serial = db.get_serial_by_code(code)
    if serial:
        serial_id = str(serial["id"])
        next_episode = db.get_next_serial_episode_number(serial_id)
        serial_title = str(serial.get("title") or code)
        await state.update_data(
            edit_type="serial",
            serial_id=serial_id,
            serial_code=code,
            serial_title=serial_title,
            next_episode=next_episode,
            episodes_added=0,
        )
        await state.set_state(EditContentState.waiting_serial_episode)
        await message.answer(
            f"📺 Serial topildi: {serial_title}\n"
            f"🎞 Navbatdagi qism: {next_episode}\n"
            "Yangi qism media yuboring.\n"
            f"Yakunlash uchun: {BTN_SERIAL_DONE}",
            reply_markup=serial_upload_kb(),
        )
        return

    await message.answer("❌ Bu kod bo'yicha kino ham serial ham topilmadi.")


@router.message(EditContentState.waiting_movie_title)
async def edit_movie_title(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text:
        await message.answer("Nom bo'sh bo'lmasin yoki o'zgartirmaslik uchun `-` yuboring.")
        return

    data = await state.get_data()
    old_title = str(data.get("movie_title") or "")
    old_description = str(data.get("movie_description") or "")
    new_title = old_title if text == "-" else text
    await state.update_data(movie_new_title=new_title)
    await state.set_state(EditContentState.waiting_movie_description)
    await message.answer(
        f"📝 Joriy tavsif: {old_description or '-'}\n"
        "Yangi tavsif yuboring.\n"
        "O'zgartirmaslik uchun: -"
    )


@router.message(EditContentState.waiting_movie_description)
async def edit_movie_description(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text:
        await message.answer("Tavsif bo'sh bo'lmasin yoki o'zgartirmaslik uchun `-` yuboring.")
        return

    data = await state.get_data()
    old_description = str(data.get("movie_description") or "")
    new_description = old_description if text == "-" else text
    await state.update_data(movie_new_description=new_description)
    await state.set_state(EditContentState.waiting_movie_metadata)
    await message.answer(
        "🏷 Yangi metadata yuboring (format: yil|sifat|janr1,janr2).\n"
        "Masalan: 2024|1080p|action,drama\n"
        "O'zgartirmaslik uchun: -"
    )


@router.message(EditContentState.waiting_movie_metadata)
async def edit_movie_metadata(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    data = await state.get_data()
    old_year = data.get("movie_year")
    old_quality = str(data.get("movie_quality") or "")
    old_genres = [str(g) for g in data.get("movie_genres", []) if str(g).strip()]

    if text == "-":
        year = old_year if isinstance(old_year, int) else None
        quality = old_quality
        genres = old_genres
    else:
        metadata = parse_metadata_input(text)
        if metadata is None:
            await message.answer(
                "⚠️ Format noto'g'ri. To'g'ri format: yil|sifat|janr1,janr2\n"
                "Masalan: 2024|1080p|action,drama\n"
                "Yoki: -"
            )
            return
        year = metadata["year"]
        quality = metadata["quality"]
        genres = metadata["genres"]

    await state.update_data(
        movie_new_year=year,
        movie_new_quality=quality,
        movie_new_genres=genres,
    )
    await state.set_state(EditContentState.waiting_movie_media)
    await message.answer(
        "🎞 Yangi media yuboring (video/document/photo yoki file_id/link).\n"
        "Media o'zgartirilmasin desangiz: -"
    )


@router.message(EditContentState.waiting_movie_media)
async def edit_movie_media(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return

    data = await state.get_data()
    code = str(data.get("edit_code") or "").strip()
    if not code:
        await state.clear()
        await message.answer("⚠️ Sessiya topilmadi, qaytadan boshlang.", reply_markup=admin_menu_kb())
        return

    if text == "-":
        media_type = str(data.get("movie_media_type") or "")
        file_id = str(data.get("movie_file_id") or "")
        preview_media_type = str(data.get("movie_preview_media_type") or "")
        preview_file_id = str(data.get("movie_preview_file_id") or "")
    else:
        parsed_media = parse_message_media(message)
        if not parsed_media:
            await message.answer(
                "Noto'g'ri format. Video/document/photo yoki matn (file_id/link) yuboring.\n"
                "Media o'zgartirilmasin desangiz: -"
            )
            return
        media_type, file_id = parsed_media
        preview_photo_file_id = extract_preview_photo_file_id(message)
        if preview_photo_file_id:
            preview_media_type = "photo"
            preview_file_id = preview_photo_file_id
        else:
            preview_media_type = str(data.get("movie_preview_media_type") or "")
            preview_file_id = str(data.get("movie_preview_file_id") or "")

    title = str(data.get("movie_new_title") or data.get("movie_title") or "").strip()
    description = str(data.get("movie_new_description") or data.get("movie_description") or "").strip()
    year_raw = data.get("movie_new_year", data.get("movie_year"))
    year = int(year_raw) if isinstance(year_raw, int) else None
    quality = str(data.get("movie_new_quality") or data.get("movie_quality") or "").strip()
    genres = [str(g) for g in data.get("movie_new_genres", data.get("movie_genres", [])) if str(g).strip()]
    if not title or not media_type or not file_id:
        await state.clear()
        await message.answer("⚠️ Tahrirlash uchun ma'lumot yetarli emas.", reply_markup=admin_menu_kb())
        return

    updated = db.update_movie(
        code=code,
        title=title,
        description=description,
        media_type=media_type,
        file_id=file_id,
        year=year,
        quality=quality,
        genres=genres,
        preview_media_type=preview_media_type,
        preview_file_id=preview_file_id,
    )
    await state.clear()
    if updated:
        await message.answer("✅ Kino muvaffaqiyatli yangilandi.", reply_markup=admin_menu_kb())
    else:
        await message.answer("❌ Kino topilmadi yoki yangilanmadi.", reply_markup=admin_menu_kb())


@router.message(EditContentState.waiting_serial_episode)
async def edit_serial_add_episode(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return

    text = (message.text or "").strip()
    data = await state.get_data()
    serial_id = str(data.get("serial_id") or "").strip()
    if not serial_id:
        await state.clear()
        await message.answer("⚠️ Sessiya topilmadi, qaytadan boshlang.", reply_markup=admin_menu_kb())
        return

    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return

    episodes_added = int(data.get("episodes_added", 0))
    if is_serial_done_text(text):
        await state.clear()
        if episodes_added > 0:
            await message.answer(
                f"✅ Serial yangilandi.\n🎞 Qo'shilgan yangi qismlar: {episodes_added}",
                reply_markup=admin_menu_kb(),
            )
        else:
            await message.answer("ℹ️ Hech qanday yangi qism qo'shilmadi.", reply_markup=admin_menu_kb())
        return

    media = parse_message_media(message)
    if not media:
        await message.answer(
            "Noto'g'ri format. Video/document/photo yoki matn (file_id/link) yuboring.",
            reply_markup=serial_upload_kb(),
        )
        return
    media_type, file_id = media

    next_episode = int(data.get("next_episode", 1))
    created = db.add_serial_episode(serial_id, next_episode, media_type, file_id)
    if not created:
        # Parallel update bo'lsa, keyingi bo'sh raqamni hisoblab yana urinib ko'ramiz.
        next_episode = db.get_next_serial_episode_number(serial_id)
        created = db.add_serial_episode(serial_id, next_episode, media_type, file_id)
    if not created:
        await message.answer("⚠️ Qismni saqlab bo'lmadi, qayta urinib ko'ring.")
        return

    new_added = episodes_added + 1
    await state.update_data(
        next_episode=next_episode + 1,
        episodes_added=new_added,
    )
    await message.answer(
        f"✅ {next_episode}-qism qo'shildi.\n"
        f"➡️ Keyingi qism: {next_episode + 1}\n"
        f"Yoki {BTN_SERIAL_DONE} tugmasini bosing.",
        reply_markup=serial_upload_kb(),
    )


@router.message(F.text.in_({BTN_BROADCAST, "Habar yuborish", "Xabar yuborish"}))
async def broadcast_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(BroadcastState.waiting_message)
    await message.answer(
        "📣 Broadcast uchun xabar yuboring.\n\n✅ Qo'llab-quvvatlanadi: matn, rasm, video, gif, document, audio, voice.",
        reply_markup=cancel_kb(),
    )


@router.message(BroadcastState.waiting_message)
async def broadcast_collect_message(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    await state.update_data(
        broadcast_source_chat_id=message.chat.id,
        broadcast_source_message_id=message.message_id,
        broadcast_button_text="",
        broadcast_button_url="",
    )
    await state.set_state(BroadcastState.waiting_button_choice)
    await message.answer("Inline tugma kerakmi?", reply_markup=build_inline_choice_kb("bcbtn"))


@router.callback_query(StateFilter(BroadcastState.waiting_button_choice), F.data.in_({"bcbtn:yes", "bcbtn:no"}))
async def broadcast_button_choice_inline(callback: CallbackQuery, state: FSMContext) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await state.clear()
        await callback.answer()
        return
    if callback.data == "bcbtn:yes":
        await state.set_state(BroadcastState.waiting_button_text)
        await callback.message.answer("🔤 Tugma matnini kiriting:", reply_markup=cancel_kb())
        await callback.answer()
        return
    data = await state.get_data()
    await copy_source_message_to_chat(
        callback.bot,
        callback.message.chat.id,
        int(data["broadcast_source_chat_id"]),
        int(data["broadcast_source_message_id"]),
        reply_markup=None,
    )
    await state.set_state(BroadcastState.waiting_confirm)
    await callback.message.answer("Yuborishni tasdiqlaysizmi?", reply_markup=confirm_kb())
    await callback.answer()


@router.message(BroadcastState.waiting_button_choice)
async def broadcast_button_choice(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if is_yes_text(text):
        await state.set_state(BroadcastState.waiting_button_text)
        await message.answer("🔤 Tugma matnini kiriting:", reply_markup=cancel_kb())
        return
    if not is_no_text(text):
        await message.answer("Iltimos, `Ha` yoki `Yo'q` tanlang.", reply_markup=build_inline_choice_kb("bcbtn"))
        return
    data = await state.get_data()
    await copy_source_message_to_chat(
        message.bot,
        message.chat.id,
        int(data["broadcast_source_chat_id"]),
        int(data["broadcast_source_message_id"]),
        reply_markup=None,
    )
    await state.set_state(BroadcastState.waiting_confirm)
    await message.answer("Yuborishni tasdiqlaysizmi?", reply_markup=confirm_kb())


@router.message(BroadcastState.waiting_button_text)
async def broadcast_button_text(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if len(text) < 2:
        await message.answer("Tugma matni juda qisqa.")
        return
    await state.update_data(broadcast_button_text=text[:60])
    await state.set_state(BroadcastState.waiting_button_url)
    await message.answer("🔗 Tugma havolasini kiriting:", reply_markup=cancel_kb())


@router.message(BroadcastState.waiting_button_url)
async def broadcast_button_url(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    url = normalize_button_url(text)
    if not url:
        await message.answer("To'g'ri havola yuboring. Masalan: https://t.me/kanal")
        return
    await state.update_data(broadcast_button_url=url)
    data = await state.get_data()
    markup = build_url_button_kb(str(data.get("broadcast_button_text") or ""), str(data.get("broadcast_button_url") or ""))
    await copy_source_message_to_chat(
        message.bot,
        message.chat.id,
        int(data["broadcast_source_chat_id"]),
        int(data["broadcast_source_message_id"]),
        reply_markup=markup,
    )
    await state.set_state(BroadcastState.waiting_confirm)
    await message.answer("Yuborishni tasdiqlaysizmi?", reply_markup=confirm_kb())


@router.message(BroadcastState.waiting_confirm)
async def broadcast_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not is_confirm_text(text):
        await message.answer("Tasdiqlash uchun `✅ Tasdiqlash` ni bosing.", reply_markup=confirm_kb())
        return
    data = await state.get_data()
    user_ids = db.list_user_ids()
    if not user_ids:
        await state.clear()
        await message.answer("📭 Yuborish uchun foydalanuvchilar topilmadi.", reply_markup=admin_menu_kb())
        return
    reply_markup = build_url_button_kb(
        str(data.get("broadcast_button_text") or ""),
        str(data.get("broadcast_button_url") or ""),
    )
    source_chat_id = int(data["broadcast_source_chat_id"])
    source_message_id = int(data["broadcast_source_message_id"])
    await message.answer(f"📤 {len(user_ids)} ta foydalanuvchiga yuborilmoqda...")
    success = 0
    failed = 0
    for user_id in user_ids:
        try:
            await copy_source_message_to_chat(
                message.bot,
                user_id,
                source_chat_id,
                source_message_id,
                reply_markup=reply_markup,
            )
            success += 1
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await copy_source_message_to_chat(
                    message.bot,
                    user_id,
                    source_chat_id,
                    source_message_id,
                    reply_markup=reply_markup,
                )
                success += 1
            except (TelegramBadRequest, TelegramForbiddenError):
                failed += 1
        except (TelegramBadRequest, TelegramForbiddenError):
            failed += 1
    await state.clear()
    await message.answer(
        "✅ Yuborish yakunlandi.\n"
        f"Yetkazildi: {success}\n"
        f"Yetkazilmadi: {failed}",
        reply_markup=admin_menu_kb(),
    )


@router.message(F.text.in_({BTN_RANDOM_CODES, "Random kod"}))
async def random_missing_codes(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    codes = generate_missing_numeric_codes(db.get_all_codes(), 10)
    lines = ["🎲 Bazada yo'q 10 ta random kod:", *[f"• {code}" for code in codes]]
    await message.answer("\n".join(lines))


@router.message(F.text.in_({BTN_LIST_MOVIES, "Kino ro'yxati"}))
async def movie_list(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    movies = db.list_movies(limit=None)
    serials = db.list_serials(limit=None)
    if not movies and not serials:
        await message.answer("📭 Kino va serial bazasi hozircha bo'sh.")
        return

    lines: list[str] = ["📚 Barcha kino va seriallar ro'yxati", ""]
    lines.append(f"🎬 Kinolar ({len(movies)}):")
    if movies:
        for item in movies:
            code = item.get("code", "-")
            title = item.get("title") or "Noma'lum"
            meta = format_meta_line(
                item.get("year") if isinstance(item.get("year"), int) else None,
                str(item.get("quality") or ""),
                None,
            )
            suffix = f" ({meta})" if meta else ""
            lines.append(f"{code} - {title}{suffix}")
    else:
        lines.append("— Kinolar yo'q")

    lines.append("")
    lines.append(f"📺 Seriallar ({len(serials)}):")
    if serials:
        for item in serials:
            code = item.get("code", "-")
            title = item.get("title") or "Noma'lum"
            meta = format_meta_line(
                item.get("year") if isinstance(item.get("year"), int) else None,
                str(item.get("quality") or ""),
                None,
            )
            suffix = f" ({meta})" if meta else ""
            lines.append(f"{code} - {title}{suffix}")
    else:
        lines.append("— Seriallar yo'q")

    text = "\n".join(lines)
    for chunk in split_text_chunks(text):
        await message.answer(chunk)


@router.message(F.text.in_({BTN_STATS, "Statistika"}))
async def stats(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    s = db.stats()
    text = (
        "📊 Bot statistikasi:\n"
        f"👥 Foydalanuvchilar: {s['users']}\n"
        f"🎬 Kinolar: {s['movies']}\n"
        f"📺 Seriallar: {s['serials']}\n"
        f"🎞 Serial qismlari: {s['serial_episodes']}\n"
        f"📢 Majburiy kanallar: {s['channels']}\n"
        f"📥 Kod so'rovlari: {s['requests']}\n"
        f"📝 Ochiq kontent so'rovlari: {s['open_content_requests']}\n"
        f"⭐ Sevimlilar: {s['favorites']}\n"
        f"👑 Aktiv PRO userlar: {s['active_pro_users']}\n"
        f"💳 Kutilayotgan PRO so'rovlar: {s['pending_payments']}\n"
        f"📰 Kutilayotgan e'lonlar: {s['pending_ads']}\n"
        f"👍👎 Reaksiyalar: {s['reactions']}"
    )
    await message.answer(text)


@router.message(F.text.in_({BTN_REQUESTS, "So'rovlar"}))
async def requests_dashboard(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return

    open_topics = db.list_open_request_topics(limit=20)
    fulfilled_topics = db.list_recent_fulfilled_topics(limit=8)

    lines: list[str] = ["📥 Kontent so'rovlari paneli", ""]
    lines.append("🔥 Ochiq so'rovlar (top):")
    if open_topics:
        for idx, row in enumerate(open_topics, start=1):
            req_type = "kod" if row.get("request_type") == "code" else "qidiruv"
            query = str(row.get("query_text") or row.get("normalized_query") or "-")
            total = int(row.get("total_requests") or 0)
            users_count = int(row.get("users_count") or 0)
            lines.append(f"{idx}. [{req_type}] {query} | so'rov: {total} | user: {users_count}")
    else:
        lines.append("— Ochiq so'rovlar yo'q")

    lines.append("")
    lines.append("✅ Oxirgi yopilganlar:")
    if fulfilled_topics:
        for row in fulfilled_topics:
            req_type = "kod" if row.get("request_type") == "code" else "qidiruv"
            query = str(row.get("query_text") or row.get("normalized_query") or "-")
            fulfilled_type = str(row.get("fulfilled_content_type") or "-")
            lines.append(f"• [{req_type}] {query} -> {fulfilled_type}")
    else:
        lines.append("— Yaqinda yopilgan so'rov yo'q")

    for chunk in split_text_chunks("\n".join(lines)):
        await message.answer(chunk)


@router.message(F.text.in_({BTN_ADD_ADMIN, "Admin qo'shish"}))
async def add_admin_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(AddAdminState.waiting_tg_id)
    await message.answer("Yangi admin Telegram ID yuboring:", reply_markup=cancel_kb())


@router.message(AddAdminState.waiting_tg_id)
async def add_admin_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        await state.clear()
        return
    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not text.isdigit():
        await message.answer("Faqat raqamli Telegram ID yuboring.")
        return
    created = db.add_admin(int(text))
    await state.clear()
    if created:
        await message.answer("Yangi admin qo'shildi.", reply_markup=admin_menu_kb())
    else:
        await message.answer("Bu foydalanuvchi allaqachon admin.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_SEARCH_NAME, "Nom bo'yicha qidirish"}))
async def search_by_name_start(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return
    ok, channels = await ensure_subscription(message.from_user.id, message.bot)
    if not ok:
        await ask_for_subscription(message, channels)
        return
    await state.set_state(SearchState.waiting_query)
    await message.answer(
        "🔎 Qidiriladigan kino/serial nomini yuboring.\n"
        "Bekor qilish uchun: ❌ Bekor qilish",
        reply_markup=cancel_kb(),
    )


@router.message(SearchState.waiting_query)
async def search_by_name_finish(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        await state.clear()
        return

    text = (message.text or "").strip()
    if is_cancel_text(text):
        await state.clear()
        await message.answer(
            "❌ Bekor qilindi.",
            reply_markup=main_menu_kb(db.is_admin(message.from_user.id)),
        )
        return
    if len(text) < 2:
        await message.answer("Kamida 2 ta belgi kiriting.")
        return

    ok, channels = await ensure_subscription(message.from_user.id, message.bot)
    if not ok:
        await ask_for_subscription(message, channels)
        return

    results = db.search_content(query=text, limit=20)
    if not results:
        await state.update_data(
            pending_request_query=text,
            pending_request_type="search",
        )
        await message.answer(
            "📭 Qidiruv bo'yicha natija topilmadi.\n"
            "Xohlasangiz, so'rov qoldiring. Kontent qo'shilsa bot sizga yuboradi.",
            reply_markup=build_not_found_request_kb(),
        )
        return

    await state.clear()
    kb = build_search_results_kb(results)
    await message.answer(
        f"✅ Topildi: {len(results)} ta natija.\nKerakli kontentni tanlang:",
        reply_markup=kb,
    )


@router.message(F.text.in_({BTN_FAVORITES, "Sevimlilarim"}))
async def list_favorites(message: Message) -> None:
    if not message.from_user:
        return
    favorites = db.list_favorites(message.from_user.id, limit=100)
    if not favorites:
        await message.answer("📭 Sevimlilar ro'yxati bo'sh.")
        return
    await message.answer(
        f"⭐ Sevimlilar ro'yxati ({len(favorites)} ta):",
        reply_markup=build_favorites_kb(favorites),
    )


@router.callback_query(F.data == "req_create")
async def create_content_request(callback: CallbackQuery, state: FSMContext) -> None:
    if not callback.from_user:
        await callback.answer()
        return

    data = await state.get_data()
    query_text = str(data.get("pending_request_query") or "").strip()
    request_type = str(data.get("pending_request_type") or "search").strip()
    if not query_text or request_type not in {"code", "search"}:
        await callback.answer("So'rov ma'lumoti topilmadi", show_alert=True)
        return

    try:
        created, count = db.add_or_increment_content_request(
            user_tg_id=callback.from_user.id,
            query_text=query_text,
            request_type=request_type,
        )
    except ValueError:
        await callback.answer("So'rov xato", show_alert=True)
        return

    cleaned = dict(data)
    cleaned.pop("pending_request_query", None)
    cleaned.pop("pending_request_type", None)
    await state.set_data(cleaned)

    if callback.message:
        if created:
            await callback.message.answer(
                "✅ So'rov qabul qilindi.\n"
                "Kontent qo'shilsa sizga avtomatik yuboriladi."
            )
        else:
            await callback.message.answer(
                f"✅ So'rov yangilandi (siz bu so'rovni {count} marta yuborgansiz)."
            )
    await callback.answer("So'rov saqlandi")


@router.callback_query(F.data.startswith("fav:"))
async def favorite_toggle(callback: CallbackQuery) -> None:
    if not callback.from_user:
        await callback.answer()
        return
    parts = callback.data.split(":", 3)
    if len(parts) != 4:
        await callback.answer("Noto'g'ri so'rov")
        return
    _, action, content_type, content_ref = parts
    if action not in {"add", "del"} or content_type not in {"movie", "serial"} or not content_ref:
        await callback.answer("Noto'g'ri so'rov")
        return

    if action == "add":
        if content_type == "movie" and not db.get_movie_by_id(content_ref):
            await callback.answer("Kino topilmadi", show_alert=True)
            return
        if content_type == "serial" and not db.get_serial(content_ref):
            await callback.answer("Serial topilmadi", show_alert=True)
            return

    if action == "add":
        created = db.add_favorite(callback.from_user.id, content_type, content_ref)
        await callback.answer("⭐ Sevimlilarga qo'shildi" if created else "ℹ️ Allaqachon sevimlida")
    else:
        removed = db.remove_favorite(callback.from_user.id, content_type, content_ref)
        await callback.answer("💔 Sevimlidan olindi" if removed else "ℹ️ Sevimlida topilmadi")

    if not callback.message:
        return

    current_text = (callback.message.text or "").strip().lower()
    if current_text.startswith("⭐ sevimlilar ro'yxati"):
        favorites = db.list_favorites(callback.from_user.id, limit=100)
        if favorites:
            await callback.message.edit_text(
                f"⭐ Sevimlilar ro'yxati ({len(favorites)} ta):",
                reply_markup=build_favorites_kb(favorites),
            )
        else:
            await callback.message.edit_text("📭 Sevimlilar ro'yxati bo'sh.")
        return

    try:
        if content_type == "movie":
            is_favorite = db.is_favorite(callback.from_user.id, "movie", content_ref)
            reaction = db.get_reaction_summary("movie", content_ref)
            await callback.message.edit_reply_markup(
                reply_markup=build_movie_actions_kb(
                    content_ref,
                    is_favorite,
                    likes=int(reaction.get("likes") or 0),
                    dislikes=int(reaction.get("dislikes") or 0),
                ),
            )
        else:
            serial = db.get_serial(content_ref)
            if serial:
                episodes = db.list_serial_episodes(content_ref)
                episode_numbers = [row["episode_number"] for row in episodes]
                is_favorite = db.is_favorite(callback.from_user.id, "serial", content_ref)
                reaction = db.get_reaction_summary("serial", content_ref)
                await callback.message.edit_reply_markup(
                    reply_markup=build_serial_episodes_kb(
                        serial_id=content_ref,
                        episode_numbers=episode_numbers,
                        is_favorite=is_favorite,
                        likes=int(reaction.get("likes") or 0),
                        dislikes=int(reaction.get("dislikes") or 0),
                    )
                )
    except TelegramBadRequest:
        pass


@router.callback_query(F.data.startswith("react:"))
async def reaction_toggle(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    parts = callback.data.split(":", 3)
    if len(parts) != 4:
        await callback.answer("Xato")
        return
    _, reaction_name, content_type, content_ref = parts
    summary = db.set_reaction(callback.from_user.id, content_type, content_ref, reaction_name)
    is_favorite = db.is_favorite(callback.from_user.id, content_type, content_ref)
    try:
        if content_type == "movie":
            await callback.message.edit_reply_markup(
                reply_markup=build_movie_actions_kb(
                    content_ref,
                    is_favorite,
                    likes=int(summary.get("likes") or 0),
                    dislikes=int(summary.get("dislikes") or 0),
                )
            )
        elif content_type == "serial":
            episodes = db.list_serial_episodes(content_ref)
            await callback.message.edit_reply_markup(
                reply_markup=build_serial_episodes_kb(
                    serial_id=content_ref,
                    episode_numbers=[row["episode_number"] for row in episodes],
                    is_favorite=is_favorite,
                    likes=int(summary.get("likes") or 0),
                    dislikes=int(summary.get("dislikes") or 0),
                )
            )
    except TelegramBadRequest:
        pass
    await callback.answer(f"⭐ Reyting: {summary.get('rating', 0.0)}/5")


@router.callback_query(F.data.startswith("short:ask:movie:"))
async def ask_movie_shorts(callback: CallbackQuery) -> None:
    await callback.answer("Qisqa video funksiyasi o'chirilgan.", show_alert=True)


@router.callback_query(F.data.startswith("short:gen:movie:"))
async def generate_movie_shorts_callback(callback: CallbackQuery) -> None:
    await callback.answer("Qisqa video funksiyasi o'chirilgan.", show_alert=True)


@router.callback_query(F.data.startswith("open:"))
async def open_content_from_callback(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    parts = callback.data.split(":", 2)
    if len(parts) != 3:
        await callback.answer("Noto'g'ri so'rov")
        return
    _, content_type, content_ref = parts
    if content_type not in {"movie", "serial"} or not content_ref:
        await callback.answer("Noto'g'ri so'rov")
        return

    ok, channels = await ensure_subscription(callback.from_user.id, callback.bot)
    if not ok:
        await callback.message.answer(
            "❗ Avval barcha majburiy kanallarga obuna bo'ling.",
            reply_markup=build_subscribe_keyboard(channels),
        )
        await callback.answer("Obuna kerak")
        return

    try:
        if content_type == "movie":
            sent = await send_movie_by_id(callback.message, content_ref, callback.from_user.id)
            if not sent:
                await callback.answer("Kino topilmadi", show_alert=True)
                return
        else:
            sent = await send_serial_selector_by_id(callback.message, content_ref, callback.from_user.id)
            if not sent:
                await callback.answer("Serial topilmadi", show_alert=True)
                return
    except (TelegramBadRequest, TelegramForbiddenError, ValueError):
        await callback.answer("Xatolik", show_alert=True)
        return

    await callback.answer("Yuborildi")


@router.inline_query()
async def inline_search(inline_query: InlineQuery) -> None:
    if not inline_query.from_user:
        await inline_query.answer([], is_personal=True, cache_time=1)
        return

    query = (inline_query.query or "").strip()
    if len(query) < 2 and not query.isdigit():
        await inline_query.answer(
            [],
            is_personal=True,
            cache_time=1,
            switch_pm_text="Kamida 2 harf yoki kod kiriting",
            switch_pm_parameter="inline",
        )
        return

    ok, _ = await ensure_subscription(inline_query.from_user.id, inline_query.bot)
    if not ok:
        await inline_query.answer(
            [],
            is_personal=True,
            cache_time=1,
            switch_pm_text="Avval obuna bo'ling",
            switch_pm_parameter="start",
        )
        return

    username = await get_bot_username(inline_query.bot)
    if not username:
        await inline_query.answer([], is_personal=True, cache_time=1)
        return

    items = db.search_content(query=query, limit=30)
    if not items:
        await inline_query.answer(
            [],
            is_personal=True,
            cache_time=3,
            switch_pm_text="Natija topilmadi. Boshqa yozuv bilan urinib ko'ring",
            switch_pm_parameter="search",
        )
        return

    answers: list[Any] = []
    seen_result_keys: set[str] = set()
    for item in items:
        content_type = str(item.get("content_type") or "")
        content_id = str(item.get("id") or "")
        if content_type not in {"movie", "serial"} or not content_id:
            continue
        result_key = f"{content_type}:{content_id}"
        if result_key in seen_result_keys:
            continue
        seen_result_keys.add(result_key)

        payload = f"m_{content_id}" if content_type == "movie" else f"s_{content_id}"
        deeplink = build_start_deeplink(username, payload)
        title = str(item.get("title") or "Noma'lum")
        code = str(item.get("code") or "")
        year = item.get("year")
        quality = str(item.get("quality") or "")
        meta = format_meta_line(year if isinstance(year, int) else None, quality, None)
        views = int(item.get("views") or 0)

        article_title = title
        if isinstance(year, int):
            article_title = f"{article_title} ({year})"
        description = f"Ko'rishlar: {views}"

        content_text = f"{title}\nKod: {code or '-'}"
        if meta:
            content_text = f"{content_text}\n{meta}"
        content_text = f"{content_text}\n\nBotda ochish: {deeplink}"

        result_id = hashlib.sha1(f"{content_type}:{content_id}:{query}".encode("utf-8")).hexdigest()[:32]
        reply_markup = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="📥 Botda ochish", url=deeplink)]]
        )
        preview = resolve_inline_media_preview(item)
        if not preview and content_type == "serial":
            preview = db.get_serial_inline_media_preview(content_id)
        if preview and preview[0] == "photo":
            answers.append(
                InlineQueryResultCachedPhoto(
                    id=result_id,
                    photo_file_id=preview[1],
                    title=article_title[:100],
                    description=description[:250],
                    caption=content_text[:1024],
                    reply_markup=reply_markup,
                )
            )
            continue

        if preview and preview[0] == "video":
            answers.append(
                InlineQueryResultCachedVideo(
                    id=result_id,
                    video_file_id=preview[1],
                    title=article_title[:100],
                    description=description[:250],
                    caption=content_text[:1024],
                    reply_markup=reply_markup,
                )
            )
            continue

        # Fallback without synthetic poster: keep result, but no fake thumbnail URL.
        answers.append(
            InlineQueryResultArticle(
                id=result_id,
                title=article_title[:100],
                description=description[:250],
                input_message_content=InputTextMessageContent(message_text=content_text[:4000]),
                reply_markup=reply_markup,
            )
        )

    if not answers:
        await inline_query.answer(
            [],
            is_personal=True,
            cache_time=2,
            switch_pm_text="Natija topilmadi. Botda qidirib ko'ring",
            switch_pm_parameter="start",
        )
        return

    await inline_query.answer(
        answers,
        is_personal=True,
        cache_time=5,
    )


@router.message(F.text.in_({BTN_CANCEL, "Bekor qilish"}))
async def cancel_any(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "❌ Amal bekor qilindi.",
        reply_markup=main_menu_kb(bool(message.from_user and db.is_admin(message.from_user.id))),
    )


@router.message(StateFilter(None), F.text)
async def handle_code_request(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return

    text = (message.text or "").strip()
    if not text:
        return

    protected_words = {
        BTN_ADMIN_PANEL.lower(),
        BTN_SUBS.lower(),
        BTN_ADD_MOVIE.lower(),
        BTN_ADD_SERIAL.lower(),
        BTN_DEL_MOVIE.lower(),
        BTN_EDIT_CONTENT.lower(),
        BTN_RANDOM_CODES.lower(),
        BTN_LIST_MOVIES.lower(),
        BTN_STATS.lower(),
        BTN_ADD_ADMIN.lower(),
        BTN_BROADCAST.lower(),
        BTN_REQUESTS.lower(),
        BTN_BACK.lower(),
        BTN_CANCEL.lower(),
        BTN_SERIAL_DONE.lower(),
        BTN_SEARCH_NAME.lower(),
        BTN_FAVORITES.lower(),
        BTN_TRENDING.lower(),
        BTN_TOP_VIEWED.lower(),
        BTN_NOTIFICATIONS.lower(),
        BTN_PRO_BUY.lower(),
        BTN_PRO_STATUS.lower(),
        BTN_CREATE_AD.lower(),
        BTN_MY_ADS.lower(),
        BTN_PRO_MANAGE.lower(),
        BTN_PRO_PRICE.lower(),
        BTN_PRO_DURATION.lower(),
        BTN_PRO_REQUESTS.lower(),
        BTN_ADS.lower(),
        BTN_AD_CHANNELS.lower(),
        BTN_YES.lower(),
        BTN_NO.lower(),
        BTN_CONFIRM.lower(),
        "admin panel",
        "majburiy obuna",
        "kino qo'shish",
        "serial qo'shish",
        "kino o'chirish",
        "kontent tahrirlash",
        "random kod",
        "kino ro'yxati",
        "kino va serial ro'yxati",
        "statistika",
        "admin qo'shish",
        "habar yuborish",
        "xabar yuborish",
        "so'rovlar",
        "ortga",
        "serialni yakunlash",
        "bekor qilish",
        "nom bo'yicha qidirish",
        "sevimlilarim",
        "trending",
        "top ko'rilganlar",
        "bildirishnomalar",
        "pro olish",
        "pro holatim",
        "e'lon berish",
        "e'lonlarim",
        "pro boshqarish",
        "pro narxi",
        "pro muddati",
        "pro so'rovlar",
        "e'lonlar",
        "e'lon kanalari",
        "ha",
        "yo'q",
        "tasdiqlash",
    }
    if text.lower() in protected_words:
        return

    ok, channels = await ensure_subscription(message.from_user.id, message.bot)
    if not ok:
        await ask_for_subscription(message, channels)
        return

    code = normalize_code(text)
    movie = db.get_movie(code)
    if movie:
        try:
            sent = await send_movie_by_id(message, str(movie["id"]), message.from_user.id)
            if not sent:
                db.log_request(message.from_user.id, code, "not_found")
                await message.answer("❌ Kino topilmadi.")
                return
            data = await state.get_data()
            cleaned = dict(data)
            cleaned.pop("pending_request_query", None)
            cleaned.pop("pending_request_type", None)
            await state.set_data(cleaned)
            db.log_request(message.from_user.id, code, "success")
        except (TelegramBadRequest, TelegramForbiddenError, ValueError):
            db.log_request(message.from_user.id, code, "send_error")
            await message.answer(
                "⚠️ Media yuborishda xatolik yuz berdi.\nAdmin faylni qayta yuklasin."
            )
        return

    serial = db.get_serial_by_code(code)
    if not serial:
        db.log_request(message.from_user.id, code, "not_found")
        await state.update_data(
            pending_request_query=code,
            pending_request_type="code",
        )
        await message.answer(
            "❌ Bunday kod topilmadi.\n"
            "Xohlasangiz so'rov qoldiring, kontent qo'shilsa bot sizga yuboradi.",
            reply_markup=build_not_found_request_kb(),
        )
        return

    sent = await send_serial_selector_by_id(message, serial["id"], message.from_user.id)
    if not sent:
        db.log_request(message.from_user.id, code, "serial_no_episodes")
        return
    data = await state.get_data()
    cleaned = dict(data)
    cleaned.pop("pending_request_query", None)
    cleaned.pop("pending_request_type", None)
    await state.set_data(cleaned)
    db.log_request(message.from_user.id, code, "success")


async def main() -> None:
    logging.info("Bot polling ishga tushmoqda...")
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties())
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

