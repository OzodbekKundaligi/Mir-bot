import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Iterable
from urllib.parse import urlparse

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatMemberStatus, ContentType
from aiogram.exceptions import ClientDecodeError, TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.filters import Command, CommandStart, StateFilter
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
    stream_sources: str | None = None
    visibility: str = "public"
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
        self.referrals = self.db["referrals"]
        self.content_posts = self.db["content_posts"]

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
    def _normalize_visibility(value: str | None) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"pro", "admin"}:
            return raw
        return "public"

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
        self.referrals.create_index("referred_user_id", unique=True)
        self.referrals.create_index([("referrer_user_id", ASCENDING), ("created_at", DESCENDING)])
        self.content_posts.create_index(
            [("content_type", ASCENDING), ("content_ref", ASCENDING), ("chat_ref", ASCENDING), ("message_id", ASCENDING)],
            unique=True,
        )
        self.content_posts.create_index([("content_type", ASCENDING), ("content_ref", ASCENDING), ("created_at", DESCENDING)])

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

    def remove_admin(self, tg_id: int) -> bool:
        result = self.admins.delete_one({"tg_id": tg_id})
        return bool(result.deleted_count)

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
            "content_mode": normalize_content_mode(doc.get("content_mode") or CONTENT_MODE_DEFAULT),
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

    def set_content_mode(self, mode: str) -> None:
        self.settings.update_one(
            {"_id": "config"},
            {
                "$set": {
                    "content_mode": normalize_content_mode(mode),
                    "updated_at": utc_now_iso(),
                }
            },
            upsert=True,
        )

    def get_site_notice(self) -> dict[str, Any]:
        doc = self.settings.find_one({"_id": "config"}, {"site_notice": 1}) or {}
        notice = doc.get("site_notice") or {}
        if not isinstance(notice, dict):
            return {}
        text = str(notice.get("text") or "").strip()
        if not text:
            return {}
        return {
            "text": text,
            "link": str(notice.get("link") or "").strip(),
            "updated_at": str(notice.get("updated_at") or ""),
            "created_by": notice.get("created_by"),
        }

    def set_site_notice(self, text: str, *, admin_id: int | None = None, link: str = "") -> None:
        cleaned_text = text.strip()[:500]
        if not cleaned_text:
            self.settings.update_one(
                {"_id": "config"},
                {
                    "$unset": {"site_notice": ""},
                    "$set": {"updated_at": utc_now_iso()},
                },
                upsert=True,
            )
            return
        self.settings.update_one(
            {"_id": "config"},
            {
                "$set": {
                    "site_notice": {
                        "text": cleaned_text,
                        "link": link.strip()[:300],
                        "updated_at": utc_now_iso(),
                        "created_by": admin_id,
                    },
                    "updated_at": utc_now_iso(),
                }
            },
            upsert=True,
        )

    def get_daily_reco_state(self) -> dict[str, Any]:
        doc = self.settings.find_one({"_id": "config"}, {"daily_reco_state": 1}) or {}
        state = doc.get("daily_reco_state") or {}
        return state if isinstance(state, dict) else {}

    def set_daily_reco_pick(self, pick_date: str, movie_id: str) -> None:
        self.settings.update_one(
            {"_id": "config"},
            {
                "$set": {
                    "daily_reco_state.pick_date": pick_date.strip()[:20],
                    "daily_reco_state.movie_id": movie_id.strip(),
                    "daily_reco_state.picked_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                }
            },
            upsert=True,
        )

    def set_daily_reco_sent(self, sent_date: str) -> None:
        self.settings.update_one(
            {"_id": "config"},
            {
                "$set": {
                    "daily_reco_state.sent_date": sent_date.strip()[:20],
                    "daily_reco_state.sent_at": utc_now_iso(),
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
                    "referred_by": None,
                    "referral_rewarded": False,
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
                    "daily_reco": True,
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
                "referred_by": 1,
                "referral_rewarded": 1,
            },
        )
        return self._doc_without_object_id(doc)

    def search_users(self, query: str = "", limit: int = 20) -> list[dict[str, Any]]:
        text = query.strip()
        mongo_query: dict[str, Any] = {}
        if text:
            or_filters: list[dict[str, Any]] = [{"full_name": {"$regex": re.escape(text), "$options": "i"}}]
            if text.isdigit():
                or_filters.insert(0, {"tg_id": int(text)})
            mongo_query = {"$or": or_filters}

        rows = self.users.find(
            mongo_query,
            {
                "tg_id": 1,
                "full_name": 1,
                "joined_at": 1,
                "is_pro": 1,
                "pro_until": 1,
                "pro_status": 1,
                "pro_note": 1,
                "pro_given_at": 1,
                "pro_given_by": 1,
            },
        ).sort("joined_at", DESCENDING).limit(max(1, int(limit)))
        return [self._doc_without_object_id(row) or {} for row in rows]

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

    def extend_pro_days(
        self,
        tg_id: int,
        days: int,
        *,
        admin_id: int | None = None,
        note: str = "",
    ) -> str:
        days = max(1, int(days))
        now_dt = datetime.now(UTC)
        now_iso = now_dt.isoformat()
        user = self.get_user(tg_id) or {"tg_id": tg_id}
        current_until = str(user.get("pro_until") or "").strip()
        base_until = now_dt
        if current_until:
            try:
                parsed = datetime.fromisoformat(current_until)
            except ValueError:
                parsed = None
            if parsed and parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            if parsed and parsed > now_dt:
                base_until = parsed
        new_until = (base_until + timedelta(days=days)).isoformat()
        self.users.update_one(
            {"tg_id": tg_id},
            {
                "$set": {
                    "is_pro": True,
                    "pro_until": new_until,
                    "pro_status": "active",
                    "pro_note": note.strip()[:200],
                    "pro_given_at": now_iso,
                    "pro_given_by": admin_id,
                },
                "$setOnInsert": {"joined_at": now_iso, "full_name": ""},
            },
            upsert=True,
        )
        self.pro_history.insert_one(
            {
                "user_tg_id": tg_id,
                "action": "extended",
                "admin_id": admin_id,
                "note": note.strip()[:200],
                "created_at": now_iso,
                "pro_until": new_until,
            }
        )
        return new_until

    def add_referral(self, referrer_user_id: int, referred_user_id: int) -> bool:
        if referrer_user_id <= 0 or referred_user_id <= 0:
            return False
        if referrer_user_id == referred_user_id:
            return False
        now = utc_now_iso()
        try:
            self.referrals.insert_one(
                {
                    "referrer_user_id": int(referrer_user_id),
                    "referred_user_id": int(referred_user_id),
                    "created_at": now,
                }
            )
        except DuplicateKeyError:
            return False
        self.users.update_one(
            {"tg_id": int(referred_user_id)},
            {
                "$set": {"referred_by": int(referrer_user_id), "referred_at": now},
                "$setOnInsert": {"joined_at": now, "full_name": ""},
            },
            upsert=True,
        )
        return True

    def count_referrals(self, referrer_user_id: int) -> int:
        return int(self.referrals.count_documents({"referrer_user_id": int(referrer_user_id)}))

    def is_referral_rewarded(self, user_tg_id: int) -> bool:
        doc = self.users.find_one({"tg_id": int(user_tg_id)}, {"referral_rewarded": 1}) or {}
        return bool(doc.get("referral_rewarded"))

    def mark_referral_rewarded(self, user_tg_id: int) -> None:
        self.users.update_one(
            {"tg_id": int(user_tg_id)},
            {"$set": {"referral_rewarded": True, "referral_rewarded_at": utc_now_iso()}},
            upsert=True,
        )

    def add_content_post(self, content_type: str, content_ref: str, chat_ref: str, message_id: int) -> bool:
        content_type = str(content_type or "").strip()
        content_ref = str(content_ref or "").strip()
        chat_ref = str(chat_ref or "").strip()
        if content_type not in {"movie", "serial"} or not content_ref or not chat_ref or not isinstance(message_id, int):
            return False
        now = utc_now_iso()
        try:
            self.content_posts.insert_one(
                {
                    "content_type": content_type,
                    "content_ref": content_ref,
                    "chat_ref": chat_ref,
                    "message_id": int(message_id),
                    "created_at": now,
                }
            )
            return True
        except DuplicateKeyError:
            return False

    def list_content_posts(self, content_type: str, content_ref: str, limit: int = 50) -> list[dict[str, Any]]:
        content_type = str(content_type or "").strip()
        content_ref = str(content_ref or "").strip()
        if content_type not in {"movie", "serial"} or not content_ref:
            return []
        cursor = (
            self.content_posts.find(
                {"content_type": content_type, "content_ref": content_ref},
                {"chat_ref": 1, "message_id": 1, "created_at": 1},
            )
            .sort("created_at", DESCENDING)
            .limit(max(1, int(limit or 50)))
        )
        return [self._doc_without_object_id(doc) or {} for doc in cursor]

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
                    "daily_reco": True,
                    "created_at": now,
                }
            },
            upsert=True,
        )
        doc = self.notification_settings.find_one({"user_tg_id": user_tg_id})
        normalized = self._doc_without_object_id(doc) or {
            "user_tg_id": user_tg_id,
            "new_content": True,
            "pro_updates": True,
            "ads_updates": True,
            "daily_reco": True,
        }
        missing: dict[str, Any] = {}
        for key, default in {
            "new_content": True,
            "pro_updates": True,
            "ads_updates": True,
            "daily_reco": True,
        }.items():
            if key not in normalized:
                missing[key] = default
                normalized[key] = default
        if missing:
            self.notification_settings.update_one(
                {"user_tg_id": user_tg_id},
                {"$set": {**missing, "updated_at": now}},
                upsert=True,
            )
        return normalized

    def toggle_notification_setting(self, user_tg_id: int, key: str) -> dict[str, Any]:
        if key not in {"new_content", "pro_updates", "ads_updates", "daily_reco"}:
            return self.get_notification_settings(user_tg_id)
        current = self.get_notification_settings(user_tg_id)
        next_value = not bool(current.get(key))
        self.notification_settings.update_one(
            {"user_tg_id": user_tg_id},
            {"$set": {key: next_value, "updated_at": utc_now_iso()}},
            upsert=True,
        )
        return self.get_notification_settings(user_tg_id)

    def list_daily_reco_user_ids(self) -> list[int]:
        user_ids: list[int] = []
        cursor = self.notification_settings.find(
            {
                "$or": [
                    {"daily_reco": True},
                    {"daily_reco": {"$exists": False}},
                ]
            },
            {"user_tg_id": 1},
        )
        for doc in cursor:
            if not doc:
                continue
            user_id = doc.get("user_tg_id")
            if isinstance(user_id, int):
                user_ids.append(user_id)
        return user_ids

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
            "stream_sources": movie.stream_sources or "",
            "visibility": self._normalize_visibility(movie.visibility),
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
                "visibility": 1,
                "year": 1,
                "quality": 1,
                "genres": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "stream_sources": 1,
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
                "visibility": 1,
                "year": 1,
                "quality": 1,
                "genres": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "stream_sources": 1,
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
        stream_sources: str | None = None,
        visibility: str = "public",
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
                    "stream_sources": stream_sources or "",
                    "visibility": self._normalize_visibility(visibility),
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

    def get_random_public_movie(self) -> dict[str, Any] | None:
        pipeline = [
            {"$match": {"visibility": "public"}},
            {"$sample": {"size": 1}},
            {
                "$project": {
                    "code": 1,
                    "title": 1,
                    "description": 1,
                    "media_type": 1,
                    "file_id": 1,
                    "stream_sources": 1,
                    "visibility": 1,
                    "year": 1,
                    "quality": 1,
                    "genres": 1,
                    "preview_media_type": 1,
                    "preview_file_id": 1,
                    "downloads": 1,
                    "views": 1,
                }
            },
        ]
        doc = next(iter(self.movies.aggregate(pipeline)), None)
        return self._doc_without_object_id(doc) if doc else None

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
        visibility: str = "public",
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
            "visibility": self._normalize_visibility(visibility),
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
        stream_sources: str | None = None,
    ) -> bool:
        now = utc_now_iso()
        doc = {
            "serial_id": serial_id,
            "episode_number": episode_number,
            "media_type": media_type,
            "file_id": file_id,
            "stream_sources": stream_sources or "",
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
                "visibility": 1,
                "preview_media_type": 1,
                "preview_file_id": 1,
                "preview_photo_file_id": 1,
                "stream_sources": 1,
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
                "visibility": 1,
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
            {"episode_number": 1, "media_type": 1, "file_id": 1, "stream_sources": 1},
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
            {"media_type": 1, "file_id": 1, "stream_sources": 1},
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
            "review_messages": [],
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

    def resolve_payment_request(
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
            {"_id": object_id, "status": "pending"},
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

    def add_payment_request_review_message(self, request_id: str, chat_id: int, message_id: int) -> None:
        object_id = self._to_object_id(request_id)
        if not object_id:
            return
        self.payment_requests.update_one(
            {"_id": object_id},
            {
                "$addToSet": {
                    "review_messages": {
                        "chat_id": int(chat_id),
                        "message_id": int(message_id),
                    }
                },
                "$set": {"updated_at": utc_now_iso()},
            },
        )

    def get_payment_request_review_messages(self, request_id: str) -> list[dict[str, int]]:
        request = self.get_payment_request(request_id) or {}
        result: list[dict[str, int]] = []
        for row in request.get("review_messages", []) or []:
            chat_id = row.get("chat_id")
            message_id = row.get("message_id")
            if isinstance(chat_id, int) and isinstance(message_id, int):
                result.append({"chat_id": chat_id, "message_id": message_id})
        return result

    def clear_payment_request_review_messages(self, request_id: str) -> None:
        object_id = self._to_object_id(request_id)
        if not object_id:
            return
        self.payment_requests.update_one(
            {"_id": object_id},
            {"$set": {"review_messages": [], "updated_at": utc_now_iso()}},
        )

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
                "review_messages": [],
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

    def resolve_ad(
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
            {"_id": object_id, "status": "pending"},
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

    def add_ad_review_message(self, ad_id: str, chat_id: int, message_id: int) -> None:
        object_id = self._to_object_id(ad_id)
        if not object_id:
            return
        self.ads.update_one(
            {"_id": object_id},
            {
                "$addToSet": {
                    "review_messages": {
                        "chat_id": int(chat_id),
                        "message_id": int(message_id),
                    }
                },
                "$set": {"updated_at": utc_now_iso()},
            },
        )

    def get_ad_review_messages(self, ad_id: str) -> list[dict[str, int]]:
        ad = self.get_ad(ad_id) or {}
        result: list[dict[str, int]] = []
        for row in ad.get("review_messages", []) or []:
            chat_id = row.get("chat_id")
            message_id = row.get("message_id")
            if isinstance(chat_id, int) and isinstance(message_id, int):
                result.append({"chat_id": chat_id, "message_id": message_id})
        return result

    def clear_ad_review_messages(self, ad_id: str) -> None:
        object_id = self._to_object_id(ad_id)
        if not object_id:
            return
        self.ads.update_one(
            {"_id": object_id},
            {"$set": {"review_messages": [], "updated_at": utc_now_iso()}},
        )

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
                "visibility": 1,
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
                "visibility": 1,
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


BTN_ADMIN_PANEL = "🛠 Admin Panel"
BTN_MINI_APP = "📱 Ilova"
MINI_APP_MENU_LINK = os.getenv("MINI_APP_MENU_LINK", "https://t.me/MirTopKinoBot/mirtopkino").strip()
BTN_SUBS = "📡 Kanallar"
BTN_ADD_MOVIE = "➕ Kino qo'shish"
BTN_ADD_SERIAL = "📺 Serial qo'shish"
BTN_DEL_MOVIE = "🗑 O'chirish"
BTN_EDIT_CONTENT = "✏️ Tahrirlash"
BTN_RANDOM_CODES = "🎲 Random kod"
BTN_LIST_MOVIES = "📚 Baza"
BTN_STATS = "📊 Stat"
BTN_ADD_ADMIN = "👤 Admin qo'shish"
BTN_BROADCAST = "📣 Broadcast"
BTN_REQUESTS = "📥 So'rovlar"
BTN_BACK = "🏠 Menyu"
BTN_CANCEL = "❌ Bekor qilish"
BTN_SERIAL_DONE = "✅ Serialni yakunlash"
BTN_SEARCH_NAME = "🔎 Qidirish"
BTN_RANDOM_MOVIE = "🎲 Random kino"
BTN_FAVORITES = "⭐ Saqlangan"
BTN_TOP_VIEWED = "🔥 TOP kinolar"
BTN_SETTINGS = "⚙️ Sozlamalar"
BTN_NOTIFICATIONS = "🔔 Bildirishnoma"
BTN_PRO_BUY = "👑 PRO"
BTN_PRO_STATUS = "💎 PRO holatim"
BTN_CREATE_AD = "📢 E'lon berish"
BTN_MY_ADS = "🗂 E'lonlarim"
BTN_HELP = "❓ Yordam"
BTN_FREE_PRO = "🎁 Bepul PRO olish"
BTN_PRO_MANAGE = "👑 PRO boshqaruv"
BTN_PRO_PRICE = "💰 PRO narxi"
BTN_PRO_DURATION = "⏳ PRO muddati"
BTN_PRO_REQUESTS = "💳 PRO so'rovlar"
BTN_CONTENT_MODE = "🔐 Media rejimi"
BTN_ADS = "📰 E'lonlar"
BTN_AD_CHANNELS = "📡 E'lon kanallari"
BTN_YES = "✅ Ha"
BTN_NO = "❌ Yo'q"
BTN_CONFIRM = "✅ Tasdiqlash"
BTN_SKIP = "/skip"
BOT_SIGNATURE = "@MirTopKinoBot"

LEGACY_MENU_TEXTS = {
    "🛠 admin panel",
    "🛠 panel",
    "ilova",
    "🧩 mini ilova",
    "⬅️ ortga",
    "🏠 menyu",
    "🏠 asosiy menyu",
    "🔎 nom bo'yicha qidirish",
    "🔎 qidirish",
    "⭐ sevimlilarim",
    "⭐ saqlangan",
    "🔥 trend",
    "🔥 trending",
    "trending",
    "trend",
    "🔥 top kinolar",
    "top kinolar",
    "🏆 top ko'rilganlar",
    "🏆 top",
    "⚙️ sozlamalar",
    "🔔 bildirishnomalar",
    "🔔 sozlama",
    "👑 pro olish",
    "👑 pro",
    "💎 pro holatim",
    "💎 holat",
    "❓ yordam",
    "📢 e'lon berish",
    "📢 e'lon",
    "🗂 e'lonlarim",
    "🗂 postlarim",
    "📢 majburiy obuna",
    "📢 obuna",
    "➕ kino qo'shish",
    "➕ kino",
    "📺 serial qo'shish",
    "📺 serial",
    "🗑 kino o'chirish",
    "🗑 o'chirish",
    "✏️ kontent tahrirlash",
    "✏️ tahrir",
    "📚 kino va serial ro'yxati",
    "📚 baza",
    "📊 statistika",
    "📊 stat",
    "📥 so'rovlar",
    "📥 so'rov",
    "📣 habar yuborish",
    "📣 xabar yuborish",
    "📣 xabar",
    "👤 admin qo'shish",
    "👤 admin",
    "🎲 random kod",
    "🎲 kod",
    "👑 pro boshqarish",
    "👑 pro boshqaruv",
    "💰 pro narxi",
    "💰 pro narx",
    "⏳ pro muddati",
    "⏳ pro muddat",
    "💳 pro so'rovlar",
    "💳 pro so'rov",
    "🔐 media",
    "🔐 media rejimi",
    "media rejimi",
    "📰 e'lonlar",
    "📰 postlar",
    "📡 e'lon kanalari",
    "📡 kanallar",
    "📡 e'lon kanallari",
}


CONTENT_MODE_PRIVATE = "private"
CONTENT_MODE_PUBLIC = "public"
CONTENT_MODE_DEFAULT = CONTENT_MODE_PRIVATE


def normalize_content_mode(value: Any) -> str:
    return CONTENT_MODE_PUBLIC if str(value or "").strip().lower() == CONTENT_MODE_PUBLIC else CONTENT_MODE_PRIVATE


def content_mode_label(mode: str) -> str:
    normalized = normalize_content_mode(mode)
    return "🌐 Ochiq" if normalized == CONTENT_MODE_PUBLIC else "🔒 Yopiq"


def get_public_base_url() -> str:
    for env_name in ("PUBLIC_URL", "RENDER_EXTERNAL_URL", "RAILWAY_PUBLIC_DOMAIN"):
        value = os.getenv(env_name, "").strip()
        if not value:
            continue
        if value.startswith(("http://", "https://")):
            return value.rstrip("/")
        return f"https://{value}".rstrip("/")
    return ""


def get_mini_app_url() -> str:
    public_base_url = get_public_base_url()
    if not public_base_url:
        return ""
    return f"{public_base_url}/app/"


def get_mini_app_launch_url() -> str:
    direct_link = str(MINI_APP_MENU_LINK or "").strip()
    if direct_link.startswith(("http://", "https://")):
        return direct_link
    return get_mini_app_url()


def get_mini_app_launch_url_with_payload(payload: str | None) -> str:
    base = get_mini_app_launch_url()
    if not base or not payload:
        return base
    if "startapp=" in base:
        if base.endswith("="):
            return f"{base}{payload}"
        if base.endswith(payload) or base.endswith(f"_{payload}"):
            return base
        return f"{base}_{payload}"
    return base


def build_mini_app_open_kb(payload: str | None = None) -> InlineKeyboardMarkup | None:
    launch_url = get_mini_app_launch_url_with_payload(payload)
    if not launch_url:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📱 Ilovani ochish", url=launch_url)]
        ]
    )


def merge_inline_keyboards(*markups: InlineKeyboardMarkup | None) -> InlineKeyboardMarkup | None:
    rows: list[list[InlineKeyboardButton]] = []
    for markup in markups:
        if not markup or not getattr(markup, "inline_keyboard", None):
            continue
        rows.extend(markup.inline_keyboard)
    if not rows:
        return None
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_content_mode_text() -> str:
    mode = db.get_bot_settings()["content_mode"]
    return (
        "🔐 Media rejimi\n"
        f"Hozir: {content_mode_label(mode)}\n\n"
        "🔒 Yopiq — yuklash va jo'natish yopiq\n"
        "🌐 Ochiq — hammasi ochiq"
    )


def is_content_visible_for_user(row: dict[str, Any] | None, user_id: int | None) -> bool:
    visibility = str((row or {}).get("visibility") or "public").lower()
    if visibility == "public":
        return True
    if visibility == "pro":
        return bool(user_id and (db.is_pro_active(user_id) or db.is_admin(user_id)))
    if visibility == "admin":
        return bool(user_id and db.is_admin(user_id))
    return True


def build_content_mode_kb(mode: str | None = None) -> InlineKeyboardMarkup:
    current_mode = normalize_content_mode(mode)
    private_text = f"{'✅ ' if current_mode == CONTENT_MODE_PRIVATE else ''}🔒 Yopiq"
    public_text = f"{'✅ ' if current_mode == CONTENT_MODE_PUBLIC else ''}🌐 Ochiq"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=private_text, callback_data="contentmode:private"),
                InlineKeyboardButton(text=public_text, callback_data="contentmode:public"),
            ]
        ]
    )


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
    del is_admin
    buttons: list[list[KeyboardButton]] = [
        [KeyboardButton(text=BTN_SEARCH_NAME), KeyboardButton(text=BTN_TOP_VIEWED)],
        [KeyboardButton(text=BTN_RANDOM_MOVIE), KeyboardButton(text=BTN_FAVORITES)],
        [KeyboardButton(text=BTN_PRO_BUY), KeyboardButton(text=BTN_SETTINGS)],
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


def settings_menu_kb(is_admin: bool, is_pro: bool) -> ReplyKeyboardMarkup:
    buttons: list[list[KeyboardButton]] = [
        [KeyboardButton(text=BTN_NOTIFICATIONS), KeyboardButton(text=BTN_MINI_APP)],
        [KeyboardButton(text=BTN_HELP), KeyboardButton(text=BTN_PRO_BUY)],
        [KeyboardButton(text=BTN_FREE_PRO)],
    ]
    if is_pro:
        buttons.append([KeyboardButton(text=BTN_CREATE_AD), KeyboardButton(text=BTN_MY_ADS)])
    if is_admin:
        buttons.append([KeyboardButton(text=BTN_ADMIN_PANEL)])
    buttons.append([KeyboardButton(text=BTN_BACK)])
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
        [KeyboardButton(text=BTN_CONTENT_MODE)],
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
    del tg_id
    return db.get_bot_settings()["content_mode"] == CONTENT_MODE_PRIVATE


def build_pro_purchase_kb() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if PRO_PAYMENT_LINK_1:
        rows.append([InlineKeyboardButton(text="💳 To'lov qilish", url=PRO_PAYMENT_LINK_1)])
    if PRO_PAYMENT_LINK_2:
        rows.append([InlineKeyboardButton(text="🌐 To'lov qilish (2)", url=PRO_PAYMENT_LINK_2)])
    rows.append([InlineKeyboardButton(text="✅ To'lov qildim", callback_data="pro_paid")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_notification_settings_kb(settings: dict[str, Any]) -> InlineKeyboardMarkup:
    def state_text(key: str) -> str:
        return "✅" if settings.get(key) else "❌"

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"🍿 Kino tavsiyasi  {state_text('daily_reco')}", callback_data="notif:daily_reco")],
            [InlineKeyboardButton(text=f"📢 E'lonlar  {state_text('ads_updates')}", callback_data="notif:ads_updates")],
            [InlineKeyboardButton(text=f"🔥 Trend xabarlar  {state_text('new_content')}", callback_data="notif:new_content")],
        ]
    )


def build_payment_request_review_kb(request_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Tasdiq", callback_data=f"proreq:approve:{request_id}"),
                InlineKeyboardButton(text="❌ Rad", callback_data=f"proreq:reject:{request_id}"),
            ]
        ]
    )


def build_ad_manage_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="➕ Qo'shish", callback_data="adch:add")],
            [InlineKeyboardButton(text="📋 Ro'yxat", callback_data="adch:list")],
            [InlineKeyboardButton(text="🗑 O'chirish", callback_data="adch:delete_menu")],
        ]
    )


def build_ad_review_kb(ad_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📡 Kanal", callback_data=f"ad:channel:{ad_id}")],
            [InlineKeyboardButton(text="❌ Rad", callback_data=f"ad:reject:{ad_id}")],
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
            builder.row(InlineKeyboardButton(text=f"🔗 {title}", url=join_link))
        elif ref.startswith("@"):
            builder.row(
                InlineKeyboardButton(
                    text=f"🔗 {title}",
                    url=f"https://t.me/{ref[1:]}",
                )
            )
    builder.row(InlineKeyboardButton(text="✅ Tekshirish", callback_data="check_sub"))
    return builder.as_markup()


def is_member_status(status: ChatMemberStatus) -> bool:
    return status in {
        ChatMemberStatus.CREATOR,
        ChatMemberStatus.ADMINISTRATOR,
        ChatMemberStatus.MEMBER,
        ChatMemberStatus.RESTRICTED,
    }


def normalize_code(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    # Allow codes like "k12345" (common in channel posts) to work as "12345".
    if (raw[0] in {"k", "K"}) and raw[1:].isdigit():
        return raw[1:]
    return raw


def format_public_code(code: str) -> str:
    """Format a content code for public display (e.g., channel posts).

    If the stored code is numeric (e.g., "12345"), we display it as "k12345"
    to make it visually distinct for users. If it's already non-numeric, we
    keep it as-is.
    """

    raw = (code or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return f"k{raw}"
    return raw


def normalize_lookup_text(value: str) -> str:
    cleaned = re.sub(r"[^\w\s]+", " ", value.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_metadata_input(value: str) -> dict[str, Any] | None:
    raw = value.strip()
    if not raw:
        return None
    if raw == "-":
        return {"year": None, "quality": "", "genres": [], "visibility": "public"}

    parts = [part.strip() for part in raw.split("|")]
    if len(parts) not in {3, 4}:
        return None

    year_part, quality_part, genres_part = parts[:3]
    visibility_part = parts[3] if len(parts) == 4 else ""
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

    visibility = Database._normalize_visibility(visibility_part)
    return {"year": year, "quality": quality, "genres": genres, "visibility": visibility}


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
        year = item.get("year")
        quality = str(item.get("quality") or "")
        meta = format_meta_line(year if isinstance(year, int) else None, quality, None)
        label = title
        if meta:
            label = f"{label} ({meta})"
        if len(label) > 60:
            label = f"{label[:57]}..."
        icon = "🎬" if content_type == "movie" else "📺"
        action = "preview" if content_type == "movie" else "open"
        builder.button(text=f"{icon} {label}", callback_data=f"{action}:{content_type}:{content_ref}")
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
        display = title
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


def format_payment_request_text(request: dict[str, Any]) -> str:
    return (
        "💳 PRO\n"
        f"👤 {request.get('user_tg_id')}\n"
        f"🔑 {request.get('payment_code') or '-'}\n"
        f"📝 {request.get('comment') or '-'}"
    )


async def close_review_messages(bot: Bot, rows: list[dict[str, int]]) -> None:
    seen: set[tuple[int, int]] = set()
    for row in rows:
        chat_id = row.get("chat_id")
        message_id = row.get("message_id")
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            continue
        key = (chat_id, message_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
        except TelegramBadRequest:
            try:
                await bot.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=None)
            except TelegramBadRequest:
                continue


async def send_payment_request_review_to_chat(
    bot: Bot,
    chat_id: int,
    request: dict[str, Any],
    reply_markup: InlineKeyboardMarkup,
) -> Message | None:
    text = format_payment_request_text(request)
    proof_media_type = str(request.get("proof_media_type") or "")
    proof_file_id = str(request.get("proof_file_id") or "")
    if proof_media_type == "photo" and proof_file_id:
        return await bot.send_photo(chat_id=chat_id, photo=proof_file_id, caption=text, reply_markup=reply_markup)
    if proof_media_type == "document" and proof_file_id:
        return await bot.send_document(chat_id=chat_id, document=proof_file_id, caption=text, reply_markup=reply_markup)
    return await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)


async def send_payment_request_review_to_message(
    message: Message,
    request: dict[str, Any],
    reply_markup: InlineKeyboardMarkup,
) -> Message | None:
    text = format_payment_request_text(request)
    proof_media_type = str(request.get("proof_media_type") or "")
    proof_file_id = str(request.get("proof_file_id") or "")
    if proof_media_type == "photo" and proof_file_id:
        return await message.answer_photo(proof_file_id, caption=text, reply_markup=reply_markup)
    if proof_media_type == "document" and proof_file_id:
        return await message.answer_document(proof_file_id, caption=text, reply_markup=reply_markup)
    return await message.answer(text, reply_markup=reply_markup)


def build_serial_episodes_kb(
    serial_id: str,
    episode_numbers: list[int],
    is_favorite: bool = False,
    likes: int = 0,
    dislikes: int = 0,
    page: int = 1,
    per_page: int = 10,
) -> InlineKeyboardMarkup:
    del likes, dislikes
    emoji_numbers = {
        1: "1️⃣",
        2: "2️⃣",
        3: "3️⃣",
        4: "4️⃣",
        5: "5️⃣",
        6: "6️⃣",
        7: "7️⃣",
        8: "8️⃣",
        9: "9️⃣",
        10: "🔟",
    }

    unique_sorted = sorted({int(n) for n in (episode_numbers or []) if int(n) > 0})
    per_page = max(1, min(20, int(per_page or 10)))
    total_pages = max(1, (len(unique_sorted) + per_page - 1) // per_page)
    page = max(1, min(total_pages, int(page or 1)))
    start = (page - 1) * per_page
    subset = unique_sorted[start : start + per_page]

    fav_text = "💔 Sevimlidan olib tashlash" if is_favorite else "⭐ Sevimliga qo'shish"
    fav_action = "del" if is_favorite else "add"

    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text=fav_text, callback_data=f"fav:{fav_action}:serial:{serial_id}")]
    ]

    # Episode grid (5 columns).
    chunk: list[InlineKeyboardButton] = []
    for number in subset:
        label = emoji_numbers.get(number, str(number))
        chunk.append(InlineKeyboardButton(text=label, callback_data=f"serial_ep:{serial_id}:{number}"))
        if len(chunk) == 5:
            rows.append(chunk)
            chunk = []
    if chunk:
        rows.append(chunk)

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 1:
            nav.append(InlineKeyboardButton(text="⬅️ Oldingi", callback_data=f"serial_page:{serial_id}:{page - 1}"))
        if page < total_pages:
            nav.append(InlineKeyboardButton(text="➡️ Keyingi", callback_data=f"serial_page:{serial_id}:{page + 1}"))
        if nav:
            rows.append(nav)

    return InlineKeyboardMarkup(inline_keyboard=rows)


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


def parse_stream_sources_input(text: str) -> str | None:
    raw = text.strip()
    if not raw:
        return None
    if "{q}" in raw or "{quality}" in raw:
        return raw
    if raw[:1] in {"{", "["}:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            return raw
    has_url = any(token in raw for token in ("http://", "https://", "cdn://", "cdn:"))
    if has_url and ("|" in raw or "\n" in raw or "=" in raw):
        return raw
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
        stream_sources = parse_stream_sources_input(text)
        if stream_sources:
            return "stream", stream_sources
        if text.startswith("http://") or text.startswith("https://"):
            post_data = parse_telegram_post_link(text)
            if post_data:
                return "telegram_post", pack_post_ref(post_data[0], post_data[1])
            return "link", text
        return "file_id", text
    return None


def extract_message_notice_text(message: Message) -> str:
    text = str(message.text or message.caption or "").strip()
    if text:
        return text[:500]
    if message.content_type == ContentType.PHOTO:
        return "Yangi rasmli e'lon"
    if message.content_type == ContentType.VIDEO:
        return "Yangi videoli e'lon"
    if message.content_type == ContentType.ANIMATION:
        return "Yangi gif e'lon"
    if message.content_type == ContentType.DOCUMENT:
        return "Yangi hujjat e'lon"
    if message.content_type == ContentType.AUDIO:
        return "Yangi audio e'lon"
    if message.content_type == ContentType.VOICE:
        return "Yangi voice e'lon"
    return "Yangi admin xabari"


async def send_stored_media(
    message: Message,
    media_type: str,
    file_id: str,
    caption: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    requester_id: int | None = None,
) -> None:
    # Keep captions within Telegram limits (1024) while preserving the bot signature.
    final_caption = clamp_media_caption(caption)
    viewer_id = requester_id
    if viewer_id is None and message.from_user and not message.from_user.is_bot:
        viewer_id = message.from_user.id
    protect_content = content_should_be_protected(viewer_id)
    logging.info(
        "send_stored_media viewer_id=%s admin=%s protect_content=%s media_type=%s",
        viewer_id,
        db.is_admin(viewer_id) if isinstance(viewer_id, int) else False,
        protect_content,
        media_type,
    )
    fallback_text = f"{final_caption}\n\nID: {file_id}".strip() if file_id else (final_caption or "Media")

    if media_type == "video":
        try:
            await message.answer_video(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        # Fallback: try as document, then plain text.
        try:
            await message.answer_document(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.answer_document(
                    file_id,
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                pass
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        await message.answer(fallback_text, reply_markup=reply_markup, protect_content=protect_content)
        return

    if media_type == "document":
        try:
            await message.answer_document(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        # Fallback: try as video, then plain text.
        try:
            await message.answer_video(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.answer_video(
                    file_id,
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                pass
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        await message.answer(fallback_text, reply_markup=reply_markup, protect_content=protect_content)
        return

    if media_type == "photo":
        try:
            await message.answer_photo(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        # Fallback: try as document, then plain text.
        try:
            await message.answer_document(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.answer_document(
                    file_id,
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                pass
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        await message.answer(fallback_text, reply_markup=reply_markup, protect_content=protect_content)
        return

    if media_type == "animation":
        try:
            await message.answer_animation(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        # Fallback: try as document, then plain text.
        try:
            await message.answer_document(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.answer_document(
                    file_id,
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                pass
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        await message.answer(fallback_text, reply_markup=reply_markup, protect_content=protect_content)
        return

    if media_type in {"file_id", "stream"}:
        if not file_id:
            await message.answer(final_caption or "Media", reply_markup=reply_markup, protect_content=protect_content)
            return
        try:
            await message.answer_video(
                file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.answer_video(
                    file_id,
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                pass
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            try:
                await message.answer_document(
                    file_id,
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except TelegramRetryAfter as exc:
                await asyncio.sleep(float(exc.retry_after))
                try:
                    await message.answer_document(
                        file_id,
                        caption=final_caption,
                        reply_markup=reply_markup,
                        protect_content=protect_content,
                    )
                    return
                except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                    pass
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                await message.answer(
                    f"{final_caption}\n\nID: {file_id}",
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
        await message.answer(fallback_text, reply_markup=reply_markup, protect_content=protect_content)
        return

    if media_type == "telegram_post":
        post_data = unpack_post_ref(file_id)
        if not post_data:
            await message.answer(fallback_text, reply_markup=reply_markup, protect_content=protect_content)
            return
        from_chat_id: int | str
        if post_data[0].lstrip("-").isdigit():
            from_chat_id = int(post_data[0])
        else:
            from_chat_id = post_data[0]
        try:
            await message.bot.copy_message(
                chat_id=message.chat.id,
                from_chat_id=from_chat_id,
                message_id=post_data[1],
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.bot.copy_message(
                    chat_id=message.chat.id,
                    from_chat_id=from_chat_id,
                    message_id=post_data[1],
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                pass
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            pass
        await message.answer(fallback_text, reply_markup=reply_markup, protect_content=protect_content)
        return

    if media_type == "link":
        post_data = parse_telegram_post_link(file_id)
        if post_data:
            from_chat_id: int | str
            if post_data[0].lstrip("-").isdigit():
                from_chat_id = int(post_data[0])
            else:
                from_chat_id = post_data[0]
            try:
                await message.bot.copy_message(
                    chat_id=message.chat.id,
                    from_chat_id=from_chat_id,
                    message_id=post_data[1],
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except TelegramRetryAfter as exc:
                await asyncio.sleep(float(exc.retry_after))
                try:
                    await message.bot.copy_message(
                        chat_id=message.chat.id,
                        from_chat_id=from_chat_id,
                        message_id=post_data[1],
                        caption=final_caption,
                        reply_markup=reply_markup,
                        protect_content=protect_content,
                    )
                    return
                except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                    pass
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                pass
            # Fall back to showing the link if we can't copy the post.
            await message.answer(
                f"{final_caption}\n\nLink: {file_id}",
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        else:
            await message.answer(
                f"{final_caption}\n\nLink: {file_id}",
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return

    await message.answer(
        fallback_text,
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


def parse_web_action_payload(payload: str | None) -> str | None:
    value = (payload or "").strip()
    if not value.startswith("wa_"):
        return None
    action = value[3:].strip()
    return action or None


def parse_referral_payload(payload: str | None) -> int | None:
    value = (payload or "").strip()
    if not value.startswith("ref_"):
        return None
    raw = value[4:].strip()
    if not raw.isdigit():
        return None
    return int(raw)


def build_start_deeplink(username: str, payload: str) -> str:
    return f"https://t.me/{username}?start={payload}"


async def send_serial_selector_by_id(
    message: Message,
    serial_id: str,
    user_id: int | None = None,
    *,
    compact: bool = False,
) -> bool:
    serial = db.get_serial(serial_id) or db.get_serial_by_code(serial_id)
    if not serial:
        await message.answer("❌ Serial topilmadi.")
        return False
    resolved_id = str(serial.get("id") or serial_id).strip()
    if not resolved_id:
        await message.answer("❌ Serial topilmadi.")
        return False
    if not is_content_visible_for_user(serial, user_id):
        await message.answer("🔒 Bu kontent faqat PRO yoki admin uchun.")
        return False

    episodes = db.list_serial_episodes(resolved_id)
    if not episodes:
        await message.answer("📭 Bu serialga hali qism qo'shilmagan.")
        return False

    episode_numbers = [row["episode_number"] for row in episodes]
    requester_id = user_id
    if requester_id is None and message.from_user and not message.from_user.is_bot:
        requester_id = message.from_user.id
    is_favorite = bool(requester_id and db.is_favorite(requester_id, "serial", resolved_id))
    reaction = db.get_reaction_summary("serial", resolved_id)

    if compact:
        caption = "📺 Serial topildi!\n\n👇 Kerakli qismni tanlang"
    else:
        title = str(serial.get("title") or "Serial").strip()
        year = serial.get("year") if isinstance(serial.get("year"), int) else None
        genres = [str(g) for g in serial.get("genres", []) if str(g).strip()]
        rating = float(reaction.get("rating") or 0.0)
        caption_lines = [
            f"📺 {title}",
            "",
            f"📅 {year if year else '-'}",
            f"🎭 {format_genres_line(genres)}",
            f"⭐ {rating}",
            "",
            "👇 Kerakli qismni tanlang",
        ]
        caption = "\n".join(caption_lines)
    caption = clamp_media_caption(caption)

    kb = build_serial_episodes_kb(
        resolved_id,
        episode_numbers,
        is_favorite=is_favorite,
        page=1,
    )
    protect_content = content_should_be_protected(requester_id)
    preview_photo_file_id = str(serial.get("preview_photo_file_id") or "").strip()
    preview_media_type = str(serial.get("preview_media_type") or "").strip()
    preview_file_id = str(serial.get("preview_file_id") or "").strip()
    if preview_photo_file_id:
        try:
            await message.answer_photo(
                preview_photo_file_id,
                caption=caption,
                reply_markup=kb,
                protect_content=protect_content,
            )
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.answer_photo(
                    preview_photo_file_id,
                    caption=caption,
                    reply_markup=kb,
                    protect_content=protect_content,
                )
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                await message.answer(caption, reply_markup=kb, protect_content=protect_content)
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            await message.answer(caption, reply_markup=kb, protect_content=protect_content)
    elif preview_media_type == "photo" and preview_file_id:
        try:
            await message.answer_photo(
                preview_file_id,
                caption=caption,
                reply_markup=kb,
                protect_content=protect_content,
            )
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.answer_photo(
                    preview_file_id,
                    caption=caption,
                    reply_markup=kb,
                    protect_content=protect_content,
                )
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                await message.answer(caption, reply_markup=kb, protect_content=protect_content)
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            await message.answer(caption, reply_markup=kb, protect_content=protect_content)
    else:
        await message.answer(caption, reply_markup=kb, protect_content=protect_content)
    db.increment_serial_views(resolved_id)
    return True


async def send_media_to_chat(
    bot: Bot,
    chat_ref: str,
    media_type: str,
    file_id: str,
    caption: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    final_caption = clamp_media_caption(caption)
    chat_id: int | str = int(chat_ref) if chat_ref.lstrip("-").isdigit() else chat_ref
    protect_content = content_should_be_protected(chat_id if isinstance(chat_id, int) and chat_id > 0 else None)
    logging.info(
        "send_media_to_chat chat_id=%s admin=%s protect_content=%s media_type=%s",
        chat_id,
        db.is_admin(chat_id) if isinstance(chat_id, int) else False,
        protect_content,
        media_type,
    )

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
    if media_type in {"file_id", "stream"}:
        if not file_id:
            merged = merge_inline_keyboards(build_mini_app_open_kb(), reply_markup)
            await bot.send_message(
                chat_id=chat_id,
                text=final_caption or "Media",
                reply_markup=merged,
                protect_content=protect_content,
            )
            return
        try:
            await bot.send_video(
                chat_id=chat_id,
                video=file_id,
                caption=final_caption,
                reply_markup=reply_markup,
                protect_content=protect_content,
            )
            return
        except TelegramBadRequest:
            try:
                await bot.send_document(
                    chat_id=chat_id,
                    document=file_id,
                    caption=final_caption,
                    reply_markup=reply_markup,
                    protect_content=protect_content,
                )
                return
            except TelegramBadRequest:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"{final_caption}\n\nID: {file_id}".strip(),
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
) -> Message:
    caption = format_ad_caption(title, description)
    if footer_text:
        caption = f"{caption}\n\n{footer_text}" if caption else footer_text
    markup = reply_markup or build_url_button_kb(button_text, button_url)
    if photo_file_id:
        return await target_message.answer_photo(photo_file_id, caption=caption, reply_markup=markup)
    return await target_message.answer(caption or "E'lon", reply_markup=markup)


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


def format_int_with_spaces(value: int | None) -> str:
    num = max(0, int(value or 0))
    # Telegram UI often uses space as a thousands separator in CIS locales.
    return f"{num:,}".replace(",", " ")


def clamp_media_caption(caption: str | None, max_len: int = 1024) -> str:
    base = (caption or "").strip()
    if not base:
        return BOT_SIGNATURE
    signature = f"\n\n{BOT_SIGNATURE}"
    # Keep signature intact when trimming.
    if base.endswith(BOT_SIGNATURE):
        trimmed = base[:max_len].rstrip()
        return trimmed
    if len(base) + len(signature) <= max_len:
        return f"{base}{signature}"
    budget = max(0, max_len - len(signature))
    trimmed = base[:budget].rstrip()
    if not trimmed:
        return BOT_SIGNATURE
    return f"{trimmed}{signature}"


def format_genres_line(genres: list[str]) -> str:
    cleaned = [str(g).strip() for g in (genres or []) if str(g).strip()]
    if not cleaned:
        return "-"
    # Stored genres are usually lowercase; make them readable.
    pretty = [g[:1].upper() + g[1:] if g else g for g in cleaned]
    return " | ".join(pretty[:6])


def build_movie_preview_caption(movie: dict[str, Any], reaction: dict[str, Any]) -> str:
    title = str(movie.get("title") or "Kino").strip()
    description = str(movie.get("description") or "").strip()
    year = movie.get("year") if isinstance(movie.get("year"), int) else None
    genres = [str(g) for g in movie.get("genres", []) if str(g).strip()]
    likes = int(reaction.get("likes") or 0)
    dislikes = int(reaction.get("dislikes") or 0)
    rating = float(reaction.get("rating") or 0.0)
    views = int(movie.get("views") or 0)

    lines: list[str] = []
    lines.append(f"🎬 {title}")
    lines.append("")
    lines.append(f"📅 Yil: {year if year else '-'}")
    lines.append(f"🎭 Janr: {format_genres_line(genres)}")
    lines.append(f"⭐ Reyting: {rating} / 5")
    if description:
        lines.append("")
        lines.append("📖 Tavsif:")
        # Avoid hitting Telegram caption limit.
        lines.append(description[:700])
    lines.append("")
    lines.append("━━━━━━━━━━━━━━")
    lines.append(f"👁 Ko'rishlar: {format_int_with_spaces(views)}")
    lines.append(f"👍 {max(0, likes)}   👎 {max(0, dislikes)}")
    return clamp_media_caption("\n".join(lines))


def build_daily_reco_caption(movie: dict[str, Any], reaction: dict[str, Any]) -> str:
    title = str(movie.get("title") or "Kino").strip()
    year = movie.get("year") if isinstance(movie.get("year"), int) else None
    genres = [str(g) for g in movie.get("genres", []) if str(g).strip()]
    rating = float(reaction.get("rating") or 0.0)
    lines = [
        "🎬 Bugungi kino tavsiyasi",
        "",
        title,
        "",
        f"⭐ {rating}",
        f"🎭 {format_genres_line(genres)}",
        f"📅 {year if year else '-'}",
        "",
        "🌙 Kechki tomosha uchun ideal kino!",
    ]
    return clamp_media_caption("\n".join(lines))


def build_random_movie_caption(movie: dict[str, Any], reaction: dict[str, Any]) -> str:
    title = str(movie.get("title") or "Kino").strip()
    year = movie.get("year") if isinstance(movie.get("year"), int) else None
    genres = [str(g) for g in movie.get("genres", []) if str(g).strip()]
    rating = float(reaction.get("rating") or 0.0)
    lines = [
        "🎲 Random kino",
        "",
        "Bugun sizga tavsiya qilamiz 👇",
        "",
        f"🎬 {title}",
        f"⭐ {rating} / 5",
        f"🎭 {format_genres_line(genres)}",
        f"📅 {year if year else '-'}",
    ]
    return clamp_media_caption("\n".join(lines))


def build_movie_preview_kb(movie_id: str, is_favorite: bool, likes: int = 0, dislikes: int = 0) -> InlineKeyboardMarkup:
    fav_text = "💔 Sevimlidan olib tashlash" if is_favorite else "⭐ Sevimliga qo'shish"
    fav_action = "del" if is_favorite else "add"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="▶️ Ko'rish", callback_data=f"open:movie:{movie_id}")],
            [InlineKeyboardButton(text=fav_text, callback_data=f"fav:{fav_action}:movie:{movie_id}")],
            [
                InlineKeyboardButton(text=f"👍 {max(0, int(likes or 0))}", callback_data=f"react:like:movie:{movie_id}"),
                InlineKeyboardButton(text=f"👎 {max(0, int(dislikes or 0))}", callback_data=f"react:dislike:movie:{movie_id}"),
            ],
        ]
    )


def build_daily_reco_kb(movie_id: str, is_favorite: bool) -> InlineKeyboardMarkup:
    fav_text = "⭐ Saqlash" if not is_favorite else "💔 O'chirish"
    fav_action = "add" if not is_favorite else "del"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="▶️ Kinoni ko'rish", callback_data=f"open:movie:{movie_id}")],
            [InlineKeyboardButton(text=fav_text, callback_data=f"fav:{fav_action}:movie:{movie_id}")],
            [InlineKeyboardButton(text="🔥 Yana tavsiya", callback_data="reco:again")],
        ]
    )


def build_random_movie_kb(movie_id: str, is_favorite: bool) -> InlineKeyboardMarkup:
    fav_text = "⭐ Saqlash" if not is_favorite else "💔 O'chirish"
    fav_action = "add" if not is_favorite else "del"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="▶️ Ko'rish", callback_data=f"open:movie:{movie_id}")],
            [InlineKeyboardButton(text="🎲 Yana random", callback_data="rand:again")],
            [InlineKeyboardButton(text=fav_text, callback_data=f"fav:{fav_action}:movie:{movie_id}")],
        ]
    )


def inline_keyboard_has_callback(markup: InlineKeyboardMarkup | None, callback_data: str) -> bool:
    if not markup or not getattr(markup, "inline_keyboard", None):
        return False
    for row in markup.inline_keyboard:
        for btn in row:
            if str(getattr(btn, "callback_data", "") or "") == callback_data:
                return True
    return False


def inline_keyboard_has_prefix(markup: InlineKeyboardMarkup | None, prefix: str) -> bool:
    if not markup or not getattr(markup, "inline_keyboard", None):
        return False
    for row in markup.inline_keyboard:
        for btn in row:
            data = str(getattr(btn, "callback_data", "") or "")
            if data.startswith(prefix):
                return True
    return False


def detect_movie_card_kind(message: Message) -> str:
    markup = message.reply_markup if message else None
    if inline_keyboard_has_callback(markup, "reco:again"):
        return "daily"
    if inline_keyboard_has_callback(markup, "rand:again"):
        return "random"
    if inline_keyboard_has_prefix(markup, "open:movie:"):
        return "preview"
    return "actions"


async def refresh_movie_preview_message(message: Message, movie_id: str, user_id: int) -> None:
    movie = db.get_movie_by_id(movie_id) or db.get_movie(movie_id)
    if not movie:
        return
    resolved_id = str(movie.get("id") or movie_id).strip()
    if not resolved_id:
        return
    reaction = db.get_reaction_summary("movie", resolved_id)
    is_favorite = db.is_favorite(user_id, "movie", resolved_id)
    caption = build_movie_preview_caption(movie, reaction)
    markup = build_movie_preview_kb(
        resolved_id,
        is_favorite=is_favorite,
        likes=int(reaction.get("likes") or 0),
        dislikes=int(reaction.get("dislikes") or 0),
    )
    try:
        if message.photo or message.caption:
            await message.edit_caption(caption=caption, reply_markup=markup)
        else:
            await message.edit_text(caption, reply_markup=markup)
    except TelegramBadRequest:
        return


async def refresh_movie_card_message(message: Message, movie_id: str, user_id: int) -> None:
    kind = detect_movie_card_kind(message)
    movie = db.get_movie_by_id(movie_id) or db.get_movie(movie_id)
    if not movie:
        return
    resolved_id = str(movie.get("id") or movie_id).strip()
    if not resolved_id:
        return
    reaction = db.get_reaction_summary("movie", resolved_id)
    is_favorite = db.is_favorite(user_id, "movie", resolved_id)
    try:
        if kind == "daily":
            caption = build_daily_reco_caption(movie, reaction)
            await message.edit_text(caption, reply_markup=build_daily_reco_kb(resolved_id, is_favorite))
        elif kind == "random":
            caption = build_random_movie_caption(movie, reaction)
            await message.edit_text(caption, reply_markup=build_random_movie_kb(resolved_id, is_favorite))
        elif kind == "preview":
            await refresh_movie_preview_message(message, resolved_id, user_id)
        else:
            await message.edit_reply_markup(
                reply_markup=build_movie_actions_kb(
                    resolved_id,
                    is_favorite=is_favorite,
                    likes=int(reaction.get("likes") or 0),
                    dislikes=int(reaction.get("dislikes") or 0),
                )
            )
    except TelegramBadRequest:
        return


MOVIE_POST_SYNC_LAST_AT: dict[str, float] = {}


def clamp_plain_text(text: str, max_len: int = 1024) -> str:
    base = (text or "").strip()
    if len(base) <= max_len:
        return base
    return base[:max_len].rstrip()


def build_movie_channel_post_caption(movie: dict[str, Any], reaction: dict[str, Any], bot_username: str) -> str:
    title = str(movie.get("title") or "Kino").strip()
    description = str(movie.get("description") or "").strip()
    code = str(movie.get("code") or "").strip()
    public_code = format_public_code(code)
    likes = int(reaction.get("likes") or 0)
    dislikes = int(reaction.get("dislikes") or 0)
    rating = float(reaction.get("rating") or 0.0)
    views = int(movie.get("views") or 0)
    lines: list[str] = []
    lines.append(f"🎬 {title}")
    if description:
        lines.append(f"📖 Tavsif: {description[:800]}")
    lines.append("")
    lines.append(f"⭐ Reyting: {rating}/5")
    lines.append(f"👍 {max(0, likes)} | 👎 {max(0, dislikes)}")
    lines.append(f"👁 Ko'rishlar: {format_int_with_spaces(views)}")
    if public_code:
        lines.append("")
        lines.append(f"Kinoni ko'rish uchun @{bot_username} ga {public_code} kodi yuboring")
    return clamp_plain_text("\n".join(lines), 1024)


def build_movie_channel_post_kb(deeplink: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="▶️ Ko'rish", url=deeplink)]])


async def maybe_post_movie_to_channel(bot: Bot, movie_id: str) -> None:
    channel_ref = str(CONTENT_POST_CHANNEL_REF or "").strip()
    if not channel_ref:
        return
    movie = db.get_movie_by_id(movie_id)
    if not movie:
        return
    username = await get_bot_username(bot)
    if not username:
        return
    code = str(movie.get("code") or "").strip()
    payload_ref = (format_public_code(code) or code) or movie_id
    deeplink = build_start_deeplink(username, f"m_{payload_ref}")
    reaction = db.get_reaction_summary("movie", movie_id)
    caption = build_movie_channel_post_caption(movie, reaction, username)
    kb = build_movie_channel_post_kb(deeplink)
    chat_id: int | str = int(channel_ref) if channel_ref.lstrip("-").isdigit() else channel_ref
    preview_media_type = str(movie.get("preview_media_type") or "")
    preview_file_id = str(movie.get("preview_file_id") or "")
    try:
        if preview_media_type == "photo" and preview_file_id:
            sent = await bot.send_photo(chat_id=chat_id, photo=preview_file_id, caption=caption, reply_markup=kb)
        else:
            sent = await bot.send_message(chat_id=chat_id, text=caption, reply_markup=kb)
    except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
        return
    try:
        db.add_content_post("movie", movie_id, channel_ref, int(sent.message_id))
    except Exception:
        pass


async def sync_movie_channel_posts(bot: Bot, movie_id: str, *, min_interval_seconds: float = 8.0) -> None:
    movie_id = str(movie_id or "").strip()
    if not movie_id:
        return
    now = time.monotonic()
    last = MOVIE_POST_SYNC_LAST_AT.get(movie_id, 0.0)
    if now - last < float(min_interval_seconds):
        return
    MOVIE_POST_SYNC_LAST_AT[movie_id] = now

    posts = db.list_content_posts("movie", movie_id, limit=60)
    if not posts:
        return
    movie = db.get_movie_by_id(movie_id)
    if not movie:
        return
    username = await get_bot_username(bot)
    if not username:
        return
    code = str(movie.get("code") or "").strip()
    payload_ref = (format_public_code(code) or code) or movie_id
    deeplink = build_start_deeplink(username, f"m_{payload_ref}")
    reaction = db.get_reaction_summary("movie", movie_id)
    caption = build_movie_channel_post_caption(movie, reaction, username)
    kb = build_movie_channel_post_kb(deeplink)
    for row in posts:
        chat_ref = str(row.get("chat_ref") or "").strip()
        msg_id = row.get("message_id")
        if not chat_ref or not isinstance(msg_id, int):
            continue
        chat_id: int | str = int(chat_ref) if chat_ref.lstrip("-").isdigit() else chat_ref
        try:
            await bot.edit_message_caption(chat_id=chat_id, message_id=msg_id, caption=caption, reply_markup=kb)
        except TelegramBadRequest:
            try:
                await bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=caption, reply_markup=kb)
            except TelegramBadRequest:
                continue
        except (TelegramForbiddenError, ClientDecodeError):
            continue


async def send_movie_preview_by_id(message: Message, movie_id: str, user_id: int | None = None) -> bool:
    movie = db.get_movie_by_id(movie_id) or db.get_movie(movie_id)
    if not movie:
        normalized = normalize_code(movie_id)
        if normalized and normalized != movie_id:
            movie = db.get_movie(normalized)
        if not movie and normalized.isdigit():
            movie = db.get_movie(f"k{normalized}") or db.get_movie(f"K{normalized}")
    if not movie:
        await message.answer("❌ Kino topilmadi.")
        return False
    resolved_id = str(movie.get("id") or movie_id).strip()
    if not resolved_id:
        await message.answer("❌ Kino topilmadi.")
        return False
    if not is_content_visible_for_user(movie, user_id):
        await message.answer("🔒 Bu kontent faqat PRO yoki admin uchun.")
        return False

    reaction = db.get_reaction_summary("movie", resolved_id)
    requester_id = user_id
    if requester_id is None and message.from_user and not message.from_user.is_bot:
        requester_id = message.from_user.id
    is_favorite = bool(requester_id and db.is_favorite(requester_id, "movie", resolved_id))
    caption = build_movie_preview_caption(movie, reaction)
    markup = build_movie_preview_kb(
        resolved_id,
        is_favorite=is_favorite,
        likes=int(reaction.get("likes") or 0),
        dislikes=int(reaction.get("dislikes") or 0),
    )

    preview_media_type = str(movie.get("preview_media_type") or "")
    preview_file_id = str(movie.get("preview_file_id") or "")
    if preview_media_type == "photo" and preview_file_id:
        try:
            await message.answer_photo(preview_file_id, caption=caption, reply_markup=markup)
            return True
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after))
            try:
                await message.answer_photo(preview_file_id, caption=caption, reply_markup=markup)
                return True
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                pass
        except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
            # Poster file_id may be invalid; fall back to text-only preview.
            pass

    # Fallback: no poster saved.
    await message.answer(caption, reply_markup=markup)
    return True


async def send_movie_by_id(message: Message, movie_id: str, user_id: int | None = None) -> bool:
    movie = db.get_movie_by_id(movie_id) or db.get_movie(movie_id)
    if not movie:
        normalized = normalize_code(movie_id)
        if normalized and normalized != movie_id:
            movie = db.get_movie(normalized)
        if not movie and normalized.isdigit():
            movie = db.get_movie(f"k{normalized}") or db.get_movie(f"K{normalized}")
    if not movie:
        await message.answer("❌ Kino topilmadi.")
        return False
    resolved_id = str(movie.get("id") or movie_id).strip()
    if not resolved_id:
        await message.answer("❌ Kino topilmadi.")
        return False
    if not is_content_visible_for_user(movie, user_id):
        await message.answer("🔒 Bu kontent faqat PRO yoki admin uchun.")
        return False
    displayed_views = int(movie.get("views") or 0) + 1
    reaction = db.get_reaction_summary("movie", resolved_id)
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
    is_favorite = bool(requester_id and db.is_favorite(requester_id, "movie", resolved_id))
    actions_kb = build_movie_actions_kb(
        resolved_id,
        is_favorite=is_favorite,
        likes=int(reaction.get("likes") or 0),
        dislikes=int(reaction.get("dislikes") or 0),
    )
    media_type = str(movie.get("media_type") or "")
    file_id = str(movie.get("file_id") or "")
    if media_type == "stream" or not file_id:
        await message.answer(
            caption or f"🎬 {movie.get('title') or 'Kino'}",
            reply_markup=merge_inline_keyboards(build_mini_app_open_kb(f"m_{movie.get('id') or resolved_id}"), actions_kb),
            protect_content=content_should_be_protected(requester_id),
        )
    else:
        await send_stored_media(
            message,
            media_type=media_type,
            file_id=file_id,
            caption=caption if caption else None,
            requester_id=requester_id,
            reply_markup=actions_kb,
        )
    db.increment_movie_views(resolved_id)
    db.increment_movie_downloads(resolved_id)
    try:
        await sync_movie_channel_posts(message.bot, resolved_id)
    except Exception:
        logging.exception("content_post: failed to sync movie posts (movie_id=%s)", resolved_id)
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
        "👑 MirTopKinoBot PRO\n\n"
        f"Atigi {settings['pro_price_text']}\n\n"
        "PRO foydalanuvchilar:\n"
        "🚫 Majburiy obunasiz foydalanadi\n"
        "⚡ Kinolar tez ochiladi\n"
        "🎬 Maxsus kinolar\n"
        "🍿 Reklamalarsiz foydalanish\n\n"
        "🧾 To'lov izohiga shu kodni yozing:\n"
        f"{payment_code}\n\n"
        f"ID: {user_id}\n\n"
        "✅ To'lovdan keyin `✅ To'lov qildim` ni bosing."
    )


def build_pro_status_text(user_id: int) -> str:
    info = db.get_pro_status(user_id)
    if info["is_pro"]:
        until_text = info["pro_until"] or "-"
        return (
            "💎 PRO aktiv\n"
            f"⏳ Amal qiladi: {until_text}\n"
            f"💰 Tarif: {info['pro_price_text']}"
        )
    if info["pro_status"] == "expired":
        return (
            "⌛ Sizning PRO muddati tugagan.\n"
            f"⏳ Oxirgi muddat: {info['pro_until'] or '-'}\n"
            f"💰 Joriy narx: {info['pro_price_text']}"
        )
    return (
        "🔒 PRO aktiv emas.\n"
        f"💰 Narx: {info['pro_price_text']}\n"
        "👑 `PRO` tugmasi orqali yoqing."
    )


async def notify_admins_about_payment_request(bot: Bot, payment_request: dict[str, Any]) -> None:
    request_id = str(payment_request.get("id") or "")
    if not request_id:
        return
    markup = build_payment_request_review_kb(str(payment_request.get("id") or ""))
    for admin_id in db.list_admin_ids():
        try:
            sent = await send_payment_request_review_to_chat(bot, admin_id, payment_request, markup)
            if sent:
                db.add_payment_request_review_message(request_id, admin_id, sent.message_id)
        except (TelegramBadRequest, TelegramForbiddenError):
            continue


async def notify_admins_about_ad(bot: Bot, ad: dict[str, Any]) -> None:
    ad_id = str(ad.get("id") or "")
    if not ad_id:
        return
    button_text = str(ad.get("button_text") or "").strip()
    button_url = str(ad.get("button_url") or "").strip()
    footer = f"👤 {ad.get('user_tg_id')}\n🆔 {ad_id}"
    if button_text and button_url:
        footer = f"{footer}\n🔗 {button_text}"
    markup = build_ad_review_kb(str(ad.get("id") or ""))
    caption = format_ad_caption(str(ad.get("title") or ""), str(ad.get("description") or ""))
    caption = f"{caption}\n\n{footer}" if caption else footer
    photo_file_id = str(ad.get("photo_file_id") or "")
    for admin_id in db.list_admin_ids():
        try:
            if photo_file_id:
                sent = await bot.send_photo(chat_id=admin_id, photo=photo_file_id, caption=caption, reply_markup=markup)
            else:
                sent = await bot.send_message(chat_id=admin_id, text=caption, reply_markup=markup)
            db.add_ad_review_message(ad_id, admin_id, sent.message_id)
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
PRO_PRICE_TEXT_DEFAULT = os.getenv("PRO_PRICE_TEXT", "10 000 so'm / 30 kun").strip() or "10 000 so'm / 30 kun"
PRO_DURATION_DAYS_DEFAULT = max(1, int(os.getenv("PRO_DURATION_DAYS", "30") or 30))
PRO_PAYMENT_LINK_1 = os.getenv(
    "PRO_PAYMENT_LINK_1",
    "https://t.me/DanatlarBot/danat?startapp=Sara_Kinolar_o1",
).strip()
PRO_PAYMENT_LINK_2 = os.getenv(
    "PRO_PAYMENT_LINK_2",
    "https://danatlar.uz/Sara_Kinolar_o1",
).strip()

# Optional: auto-post new movies to a channel (set to chat id like -100... or @channel).
CONTENT_POST_CHANNEL_REF = os.getenv("CONTENT_POST_CHANNEL_REF", "").strip()

DAILY_RECO_ENABLED = os.getenv("DAILY_RECO_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
DAILY_RECO_AT = os.getenv("DAILY_RECO_AT", "20:00").strip() or "20:00"
DAILY_RECO_UTC_OFFSET_MINUTES = int(os.getenv("DAILY_RECO_UTC_OFFSET_MINUTES", "300") or 300)
DAILY_RECO_UTC_OFFSET_MINUTES = max(-720, min(840, DAILY_RECO_UTC_OFFSET_MINUTES))

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
    await message.answer(
        "🔒 Davom etish uchun kanallarga obuna bo'ling\n\n"
        "Quyidagi kanallarga kirib keyin tekshiring 👇",
        reply_markup=build_subscribe_keyboard(channels),
    )


def parse_hhmm(value: str, *, default_hour: int = 20, default_minute: int = 0) -> tuple[int, int]:
    match = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", value or "")
    if not match:
        return default_hour, default_minute
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return default_hour, default_minute
    return hour, minute


def daily_reco_local_now() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=DAILY_RECO_UTC_OFFSET_MINUTES)


def daily_reco_local_date() -> str:
    return daily_reco_local_now().date().isoformat()


def get_or_pick_daily_reco_movie_id(local_date: str) -> str | None:
    state = db.get_daily_reco_state()
    movie_id = str(state.get("movie_id") or "").strip()
    pick_date = str(state.get("pick_date") or "").strip()
    if movie_id and pick_date == local_date:
        movie = db.get_movie_by_id(movie_id)
        if movie and str(movie.get("visibility") or "") == "public":
            return movie_id
    movie = db.get_random_public_movie()
    if not movie:
        return None
    picked_id = str(movie.get("id") or "").strip()
    if picked_id:
        db.set_daily_reco_pick(local_date, picked_id)
    return picked_id or None


async def send_daily_recommendation(bot: Bot) -> None:
    local_date = daily_reco_local_date()
    state = db.get_daily_reco_state()
    if str(state.get("sent_date") or "").strip() == local_date:
        return

    movie_id = get_or_pick_daily_reco_movie_id(local_date)
    if not movie_id:
        logging.info("daily_reco: no movies to recommend")
        return

    movie = db.get_movie_by_id(movie_id)
    if not movie or str(movie.get("visibility") or "") != "public":
        logging.info("daily_reco: picked movie is missing or not public (movie_id=%s)", movie_id)
        return

    user_ids = db.list_daily_reco_user_ids()
    if not user_ids:
        return

    reaction = db.get_reaction_summary("movie", movie_id)
    final_text = build_daily_reco_caption(movie, reaction)
    final_kb = build_daily_reco_kb(movie_id, is_favorite=False)

    semaphore = asyncio.Semaphore(25)

    async def safe_send_countdown(chat_id: int) -> tuple[int, int] | None:
        async with semaphore:
            try:
                msg = await bot.send_message(chat_id=chat_id, text="🍿 3...")
                return chat_id, msg.message_id
            except TelegramRetryAfter as exc:
                await asyncio.sleep(float(exc.retry_after))
                try:
                    msg = await bot.send_message(chat_id=chat_id, text="🍿 3...")
                    return chat_id, msg.message_id
                except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                    return None
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                return None

    async def safe_edit_text(chat_id: int, message_id: int, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> None:
        async with semaphore:
            try:
                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, reply_markup=reply_markup)
            except TelegramRetryAfter as exc:
                await asyncio.sleep(float(exc.retry_after))
                try:
                    await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, reply_markup=reply_markup)
                except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                    return
            except (TelegramBadRequest, TelegramForbiddenError, ClientDecodeError):
                return

    logging.info("daily_reco: sending countdown movie_id=%s to %s users", movie_id, len(user_ids))
    sent_refs = await asyncio.gather(*(safe_send_countdown(user_id) for user_id in user_ids))
    targets: list[tuple[int, int]] = []
    for item in sent_refs:
        if not item:
            continue
        targets.append(item)
    if not targets:
        return

    await asyncio.sleep(0.7)
    await asyncio.gather(*(safe_edit_text(chat_id, msg_id, "🍿 2...") for chat_id, msg_id in targets))
    await asyncio.sleep(0.7)
    await asyncio.gather(*(safe_edit_text(chat_id, msg_id, "🍿 1...") for chat_id, msg_id in targets))
    await asyncio.sleep(0.7)
    await asyncio.gather(*(safe_edit_text(chat_id, msg_id, final_text, reply_markup=final_kb) for chat_id, msg_id in targets))

    db.set_daily_reco_sent(local_date)
    logging.info("daily_reco: done date=%s movie_id=%s sent=%s", local_date, movie_id, len(targets))


async def daily_recommendation_loop(bot: Bot) -> None:
    if not DAILY_RECO_ENABLED:
        logging.info("daily_reco: disabled (DAILY_RECO_ENABLED=0)")
        return

    hour, minute = parse_hhmm(DAILY_RECO_AT)
    logging.info(
        "daily_reco: enabled at %02d:%02d (utc_offset_minutes=%s)",
        hour,
        minute,
        DAILY_RECO_UTC_OFFSET_MINUTES,
    )

    while True:
        now_local = daily_reco_local_now()
        target = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now_local >= target:
            target += timedelta(days=1)
        sleep_seconds = max(1.0, (target - now_local).total_seconds())
        await asyncio.sleep(sleep_seconds)
        try:
            await send_daily_recommendation(bot)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("daily_reco: unexpected error")
        await asyncio.sleep(2.0)


async def dispatch_web_action(
    message: Message,
    state: FSMContext,
    action: str,
    payload: dict[str, Any] | None = None,
) -> bool:
    if not message.from_user:
        return False

    payload = payload or {}
    action = str(action or "").strip()
    if not action:
        return False

    if action == "open_content":
        content_type = str(payload.get("content_type") or "").strip()
        content_ref = str(payload.get("content_ref") or "").strip()
        if content_type not in {"movie", "serial"} or not content_ref:
            await message.answer("Kontent topilmadi.")
            return True
        ok, channels = await ensure_subscription(message.from_user.id, message.bot)
        if not ok:
            await ask_for_subscription(message, channels)
            return True
        if content_type == "movie":
            try:
                sent = await send_movie_by_id(message, content_ref, message.from_user.id)
            except (TelegramBadRequest, TelegramForbiddenError, ValueError):
                sent = False
            if not sent:
                await message.answer("Kino topilmadi.")
            return True
        try:
            sent = await send_serial_selector_by_id(message, content_ref, message.from_user.id)
        except (TelegramBadRequest, TelegramForbiddenError, ValueError):
            sent = False
        if not sent:
            await message.answer("Serial topilmadi.")
        return True

    if action == "open_pro":
        await pro_buy(message)
        return True

    if action == "open_notifications":
        await notification_settings(message)
        return True

    if action in {
        "open_admin_panel",
        "admin_subs",
        "admin_add_movie",
        "admin_add_serial",
        "admin_delete_content",
        "admin_edit_content",
        "admin_list_content",
        "admin_broadcast",
        "admin_requests",
        "admin_stats",
        "admin_add_admin",
    }:
        if not guard_admin(message):
            await message.answer("Admin huquqi kerak.")
            return True
        if action == "open_admin_panel":
            await open_admin_panel(message, state)
            return True
        if action == "admin_subs":
            await mandatory_subscriptions_menu(message)
            return True
        if action == "admin_add_movie":
            await add_movie_start(message, state)
            return True
        if action == "admin_add_serial":
            await add_serial_start(message, state)
            return True
        if action == "admin_delete_content":
            await delete_movie_start(message, state)
            return True
        if action == "admin_edit_content":
            await edit_content_start(message, state)
            return True
        if action == "admin_list_content":
            await movie_list(message)
            return True
        if action == "admin_broadcast":
            await broadcast_start(message, state)
            return True
        if action == "admin_requests":
            await requests_dashboard(message)
            return True
        if action == "admin_stats":
            await stats(message)
            return True
        if action == "admin_add_admin":
            await add_admin_start(message, state)
            return True

    return False


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return

    was_existing_user = db.get_user(message.from_user.id) is not None
    db.add_user(message.from_user.id, message.from_user.full_name)
    # Ensure default notification settings exist so daily 20:00 recommendations work for new users.
    try:
        db.get_notification_settings(message.from_user.id)
    except Exception:
        logging.exception("notif: failed to ensure notification settings for user=%s", message.from_user.id)
    payload = parse_start_payload(message.text)
    web_action = parse_web_action_payload(payload)
    serial_id = parse_serial_payload(payload)
    movie_id = parse_movie_payload(payload)
    referrer_id = parse_referral_payload(payload)

    if referrer_id and not was_existing_user:
        if db.add_referral(referrer_id, message.from_user.id):
            count = db.count_referrals(referrer_id)
            try:
                await message.bot.send_message(
                    chat_id=referrer_id,
                    text=f"🎁 Yangi do'st qo'shildi! ({min(3, count)}/3)\n3 ta do'st = 7 kunlik PRO.",
                )
            except (TelegramBadRequest, TelegramForbiddenError):
                pass
            if count >= 3 and not db.is_referral_rewarded(referrer_id):
                db.mark_referral_rewarded(referrer_id)
                new_until = db.extend_pro_days(referrer_id, 7, note="Referral bonus (3 do'st)")
                if db.get_notification_settings(referrer_id).get("pro_updates"):
                    try:
                        await message.bot.send_message(
                            chat_id=referrer_id,
                            text=f"💎 Tabriklaymiz! Sizga 7 kunlik PRO berildi.\n⏳ Amal qiladi: {new_until}",
                        )
                    except (TelegramBadRequest, TelegramForbiddenError):
                        pass
    if web_action:
        if await dispatch_web_action(message, state, web_action, {"action": web_action}):
            return
    elif serial_id:
        ok, channels = await ensure_subscription(message.from_user.id, message.bot)
        if not ok:
            await state.update_data(pending_serial_id=serial_id)
            await ask_for_subscription(message, channels)
            return
        sent = await send_serial_selector_by_id(message, serial_id, compact=True)
        if sent:
            return
    elif movie_id:
        ok, channels = await ensure_subscription(message.from_user.id, message.bot)
        if not ok:
            await state.update_data(pending_movie_id=movie_id)
            await ask_for_subscription(message, channels)
            return
        await message.answer("🎥 Kino yuklanmoqda...")
        try:
            sent = await send_movie_by_id(message, movie_id)
        except (TelegramBadRequest, TelegramForbiddenError, ValueError):
            sent = False
        if sent:
            return

    admin = db.is_admin(message.from_user.id)
    first_name = (message.from_user.first_name or "foydalanuvchi").strip()
    await message.answer(
        f"🎬 Assalomu alaykum, {first_name}!\n\n"
        "MirTopKinoBot ga xush kelibsiz 🍿\n\n"
        "📥 Kino ko'rish uchun:\n"
        "Kino kodini yuboring yoki menyudan foydalaning",
        reply_markup=main_menu_kb(admin),
    )


@router.message(F.text.in_({"/me", "/whoami", "/chat_id"}))
async def whoami(message: Message) -> None:
    if not message.from_user:
        return
    user_id = message.from_user.id
    is_admin_user = db.is_admin(user_id)
    content_mode = db.get_bot_settings()["content_mode"]
    protect_content = content_should_be_protected(user_id)
    await message.answer(
        "🆔 ID: {user_id}\n👤 Admin: {is_admin_user}\n🔐 Rejim: {content_mode}\n🔒 Protect: {protect_content}".format(
            user_id=user_id,
            is_admin_user=is_admin_user,
            content_mode=content_mode_label(content_mode),
            protect_content=protect_content,
        )
    )


@router.message(F.web_app_data)
async def handle_web_app_data(message: Message, state: FSMContext) -> None:
    if not message.from_user or not message.web_app_data:
        return
    raw = str(message.web_app_data.data or "").strip()
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await message.answer("Ilova ma'lumoti o'qilmadi.")
        return

    action = str(payload.get("action") or "").strip()
    if await dispatch_web_action(message, state, action, payload):
        return

    await message.answer("Ilova bu amalni yubordi, lekin bot uni tanimadi.")


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
        pending_movie_preview_id = str(state_data.get("pending_movie_preview_id") or "").strip()
        pending_code = str(state_data.get("pending_code") or "").strip()
        if pending_serial_id and callback.message:
            cleaned_state = dict(state_data)
            cleaned_state.pop("pending_serial_id", None)
            await state.set_data(cleaned_state)
            sent = await send_serial_selector_by_id(callback.message, pending_serial_id, user.id)
            if sent:
                await callback.answer("✅ Obuna tasdiqlandi!")
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
                await callback.answer("✅ Obuna tasdiqlandi!")
                return
        if pending_movie_preview_id and callback.message:
            cleaned_state = dict(state_data)
            cleaned_state.pop("pending_movie_preview_id", None)
            await state.set_data(cleaned_state)
            try:
                sent = await send_movie_preview_by_id(callback.message, pending_movie_preview_id, user.id)
            except (TelegramBadRequest, TelegramForbiddenError, ValueError):
                sent = False
            if sent:
                await callback.answer("✅ Obuna tasdiqlandi!")
                return
        if pending_code and callback.message:
            cleaned_state = dict(state_data)
            cleaned_state.pop("pending_code", None)
            await state.set_data(cleaned_state)
            raw = str(pending_code or "").strip()
            code = normalize_code(raw)
            movie = db.get_movie(code) or (db.get_movie(raw) if code != raw else None)
            if not movie and code.isdigit():
                movie = db.get_movie(f"k{code}") or db.get_movie(f"K{code}")
            if movie:
                try:
                    sent = await send_movie_preview_by_id(callback.message, str(movie.get("id") or ""), user.id)
                except (TelegramBadRequest, TelegramForbiddenError, ValueError):
                    sent = False
                if sent:
                    await callback.answer("✅ Obuna tasdiqlandi!")
                    return
            serial = db.get_serial_by_code(code) or (db.get_serial_by_code(raw) if code != raw else None)
            if serial:
                try:
                    sent = await send_serial_selector_by_id(callback.message, str(serial.get("id") or ""), user.id)
                except (TelegramBadRequest, TelegramForbiddenError, ValueError):
                    sent = False
                if sent:
                    await callback.answer("✅ Obuna tasdiqlandi!")
                    return
        await callback.message.answer("✅ Obuna tasdiqlandi.\nKod yuboring.")
        await callback.answer("✅ Obuna tasdiqlandi!")
    else:
        await callback.message.answer(
            "Hali barcha kanallarga kirmagansiz.",
            reply_markup=build_subscribe_keyboard(channels),
        )
        await callback.answer("❗ Obuna to'liq emas")


@router.message(F.text.in_({BTN_ADMIN_PANEL, "Admin panel"}))
async def open_admin_panel(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.clear()
    await message.answer("🛠 Admin Panel", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_BACK, "Ortga"}))
async def back_to_main(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return
    await state.clear()
    await message.answer("🏠 Menyu", reply_markup=main_menu_kb(db.is_admin(message.from_user.id)))


@router.message(F.text.in_({BTN_SETTINGS, "Sozlamalar"}))
async def open_settings(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return
    await state.clear()
    await message.answer(
        "⚙️ Sozlamalar",
        reply_markup=settings_menu_kb(
            db.is_admin(message.from_user.id),
            db.is_pro_active(message.from_user.id),
        ),
    )


@router.message(F.text.in_({BTN_MINI_APP, "Ilova"}))
async def open_mini_app(message: Message) -> None:
    if not message.from_user:
        return
    launch_url = get_mini_app_launch_url()
    if not launch_url:
        await message.answer("Ilova havolasi hali sozlanmagan.")
        return
    await message.answer("📱 Ilovani ochish uchun tugmani bosing.", reply_markup=build_mini_app_open_kb())


@router.message(F.text.in_({BTN_PRO_BUY, "Pro olish"}))
async def pro_buy(message: Message) -> None:
    if not message.from_user:
        return
    db.add_user(message.from_user.id, message.from_user.full_name)
    if db.is_admin(message.from_user.id):
        await message.answer(build_pro_status_text(message.from_user.id), reply_markup=main_menu_kb(True))
        return
    if db.is_pro_active(message.from_user.id):
        await message.answer(
            build_pro_status_text(message.from_user.id) + "\n\nQayta olish muddati tugagach ochiladi.",
            reply_markup=main_menu_kb(False),
        )
        return
    await message.answer(build_pro_offer_text(message.from_user.id), reply_markup=build_pro_purchase_kb())


@router.message(F.text.in_({BTN_FREE_PRO, "Bepul PRO olish"}))
async def free_pro_info(message: Message) -> None:
    if not message.from_user:
        return
    username = await get_bot_username(message.bot)
    if not username:
        await message.answer("Bot username topilmadi. Keyinroq urinib ko'ring.")
        return
    link = build_start_deeplink(username, f"ref_{message.from_user.id}")
    count = db.count_referrals(message.from_user.id)
    rewarded = db.is_referral_rewarded(message.from_user.id)
    progress = f"{min(3, count)}/3"
    status = "✅ Bonus berilgan" if rewarded else "⏳ Bonus kutilmoqda"
    await message.answer(
        "🎁 Bepul PRO olish\n\n"
        "3 ta do'st taklif qiling\n"
        "7 kunlik PRO oling\n\n"
        f"📊 Holat: {progress} | {status}\n\n"
        f"Havola:\n{link}"
    )


@router.callback_query(F.data == "pro_paid")
async def pro_paid_start(callback: CallbackQuery, state: FSMContext) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    if db.is_admin(callback.from_user.id) or db.is_pro_active(callback.from_user.id):
        await callback.answer("PRO allaqachon faol.", show_alert=True)
        return
    await state.set_state(PaymentState.waiting_proof)
    await callback.message.answer(
        "To'lov skrinini yuboring.\n`/skip` ham bo'ladi.",
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
        "📝 Izoh yuboring.\n`/skip` ham bo'ladi.",
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
        "✅ So'rov yuborildi.\nAdmin tekshiradi.",
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
    await message.answer("🔔 Bildirishnomalar", reply_markup=build_notification_settings_kb(settings))


@router.message(Command("help"))
@router.message(F.text.in_({BTN_HELP, "Yordam"}))
async def help_menu(message: Message) -> None:
    if not message.from_user:
        return
    await message.answer(
        "❓ Yordam\n\n"
        "🎬 Kino kodini yuboring.\n"
        f"{BTN_SEARCH_NAME} orqali kino/serial qidiring.\n"
        f"{BTN_TOP_VIEWED} orqali eng ko'p ko'rilganlarni ko'ring.\n"
        f"{BTN_RANDOM_MOVIE} orqali tasodifiy kino oling.\n"
        f"{BTN_SETTINGS} bo'limida sozlamalar bor.",
        reply_markup=settings_menu_kb(
            db.is_admin(message.from_user.id),
            db.is_pro_active(message.from_user.id),
        ),
    )


@router.callback_query(F.data.startswith("notif:"))
async def notification_toggle(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    _, key = callback.data.split(":", 1)
    settings = db.toggle_notification_setting(callback.from_user.id, key)
    await callback.message.edit_reply_markup(reply_markup=build_notification_settings_kb(settings))
    await callback.answer("✅ Yangilandi")


def extract_movie_id_from_markup(markup: InlineKeyboardMarkup | None) -> str:
    if not markup or not getattr(markup, "inline_keyboard", None):
        return ""
    for row in markup.inline_keyboard:
        for btn in row:
            data = str(getattr(btn, "callback_data", "") or "")
            if data.startswith("open:movie:"):
                return data.split(":", 2)[2].strip()
            if data.startswith("fav:") and ":movie:" in data:
                parts = data.split(":")
                if len(parts) >= 4:
                    return str(parts[3]).strip()
    return ""


def pick_random_public_movie_id(*, exclude_movie_id: str = "", max_attempts: int = 6) -> str | None:
    exclude_movie_id = str(exclude_movie_id or "").strip()
    for _ in range(max(1, int(max_attempts or 1))):
        movie = db.get_random_public_movie()
        movie_id = str((movie or {}).get("id") or "").strip()
        if movie_id and movie_id != exclude_movie_id:
            return movie_id
    return None


@router.callback_query(F.data == "rand:again")
async def random_movie_again(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    current_movie_id = extract_movie_id_from_markup(callback.message.reply_markup)
    movie_id = pick_random_public_movie_id(exclude_movie_id=current_movie_id)
    if not movie_id:
        await callback.answer("Kino topilmadi", show_alert=True)
        return
    movie = db.get_movie_by_id(movie_id)
    if not movie:
        await callback.answer("Kino topilmadi", show_alert=True)
        return
    reaction = db.get_reaction_summary("movie", movie_id)
    is_favorite = db.is_favorite(callback.from_user.id, "movie", movie_id)
    caption = build_random_movie_caption(movie, reaction)
    try:
        await callback.message.edit_text(caption, reply_markup=build_random_movie_kb(movie_id, is_favorite))
    except TelegramBadRequest:
        await callback.message.answer(caption, reply_markup=build_random_movie_kb(movie_id, is_favorite))
    await callback.answer()


@router.callback_query(F.data == "reco:again")
async def daily_reco_again(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    current_movie_id = extract_movie_id_from_markup(callback.message.reply_markup)
    movie_id = pick_random_public_movie_id(exclude_movie_id=current_movie_id)
    if not movie_id:
        await callback.answer("Kino topilmadi", show_alert=True)
        return
    movie = db.get_movie_by_id(movie_id)
    if not movie:
        await callback.answer("Kino topilmadi", show_alert=True)
        return
    reaction = db.get_reaction_summary("movie", movie_id)
    is_favorite = db.is_favorite(callback.from_user.id, "movie", movie_id)
    caption = build_daily_reco_caption(movie, reaction)
    try:
        await callback.message.edit_text(caption, reply_markup=build_daily_reco_kb(movie_id, is_favorite))
    except TelegramBadRequest:
        await callback.message.answer(caption, reply_markup=build_daily_reco_kb(movie_id, is_favorite))
    await callback.answer()


@router.message(F.text.in_({BTN_RANDOM_MOVIE, "Kino tavsiyasi", "Tavsiya"}))
async def random_movie(message: Message) -> None:
    if not message.from_user:
        return
    ok, channels = await ensure_subscription(message.from_user.id, message.bot)
    if not ok:
        await state.update_data(pending_code=text)
        await ask_for_subscription(message, channels)
        return
    movie = db.get_random_public_movie()
    movie_id = str((movie or {}).get("id") or "").strip()
    if not movie_id:
        await message.answer("📭 Hozircha tavsiya qiladigan kino topilmadi.")
        return
    reaction = db.get_reaction_summary("movie", movie_id)
    is_favorite = db.is_favorite(message.from_user.id, "movie", movie_id)
    caption = build_random_movie_caption(movie or {}, reaction)
    await message.answer(caption, reply_markup=build_random_movie_kb(movie_id, is_favorite))


@router.message(F.text.in_({BTN_TOP_VIEWED, "Top ko'rilganlar"}))
async def top_viewed_content(message: Message) -> None:
    if not message.from_user:
        return
    ok, channels = await ensure_subscription(message.from_user.id, message.bot)
    if not ok:
        await ask_for_subscription(message, channels)
        return
    raw_items = db.list_top_viewed_content(limit=60)
    items = [row for row in raw_items if str(row.get("content_type") or "") == "movie"][:20]
    if not items:
        items = raw_items[:20]
    kb = build_search_results_kb(items)
    if not kb:
        await message.answer("🔥 TOP kinolar hozircha bo'sh.")
        return
    await message.answer("🔥 TOP kinolar", reply_markup=kb)


@router.message(F.text.in_({BTN_CREATE_AD, "E'lon berish"}))
async def create_ad_start(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return
    if not has_active_pro(message):
        await message.answer(
            "🔒 E'lon faqat PRO uchun.\n\n" + build_pro_offer_text(message.from_user.id),
            reply_markup=build_pro_purchase_kb(),
        )
        return
    await state.set_state(AdCreateState.waiting_photo)
    await message.answer(
        "🖼 Rasm yuboring.\nRasmsiz bo'lsa `/skip`.",
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
    await message.answer("📝 Sarlavha kiriting:", reply_markup=cancel_kb())


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
    await message.answer("📄 Tavsif kiriting:", reply_markup=cancel_kb())


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
    await message.answer("🔘 Tugma kerakmi?", reply_markup=build_inline_choice_kb("adbtn"))


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


@router.message(F.text.in_({BTN_CONTENT_MODE, "Media rejimi"}))
async def content_mode_menu(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    settings = db.get_bot_settings()
    await message.answer(
        build_content_mode_text(),
        reply_markup=build_content_mode_kb(settings["content_mode"]),
    )


@router.callback_query(F.data.startswith("contentmode:"))
async def content_mode_toggle(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    _, raw_mode = callback.data.split(":", 1)
    new_mode = normalize_content_mode(raw_mode)
    current_mode = db.get_bot_settings()["content_mode"]
    if new_mode == current_mode:
        await callback.answer("Rejim o'zgarmadi")
        return
    db.set_content_mode(new_mode)
    await callback.message.edit_text(
        build_content_mode_text(),
        reply_markup=build_content_mode_kb(new_mode),
    )
    await callback.answer(f"Rejim: {content_mode_label(new_mode)}")


@router.message(F.text.in_({BTN_PRO_REQUESTS, "Pro so'rovlar"}))
async def pro_requests(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    requests = db.list_pending_payment_requests(limit=15)
    if not requests:
        await message.answer("📭 PRO so'rov yo'q.")
        return
    for request in requests:
        markup = build_payment_request_review_kb(str(request.get("id") or ""))
        sent = await send_payment_request_review_to_message(message, request, markup)
        if sent and message.from_user:
            db.add_payment_request_review_message(str(request.get("id") or ""), message.from_user.id, sent.message_id)


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
    if str(request.get("status") or "") != "pending":
        await close_review_messages(callback.bot, db.get_payment_request_review_messages(request_id))
        db.clear_payment_request_review_messages(request_id)
        await callback.answer("Tayyor", show_alert=True)
        return
    user_id = int(request.get("user_tg_id") or 0)
    if action == "approve":
        if not db.resolve_payment_request(request_id, "approved", reviewed_by=callback.from_user.id):
            await close_review_messages(callback.bot, db.get_payment_request_review_messages(request_id))
            db.clear_payment_request_review_messages(request_id)
            await callback.answer("Tayyor", show_alert=True)
            return
        db.set_pro_state(user_id, True, admin_id=callback.from_user.id, note="To'lov tasdiqlandi")
        if db.get_notification_settings(user_id).get("pro_updates"):
            try:
                await callback.bot.send_message(chat_id=user_id, text="✅ PRO yoqildi.")
            except (TelegramBadRequest, TelegramForbiddenError):
                pass
        await callback.answer("✅ Tasdiq")
    else:
        if not db.resolve_payment_request(request_id, "rejected", reviewed_by=callback.from_user.id):
            await close_review_messages(callback.bot, db.get_payment_request_review_messages(request_id))
            db.clear_payment_request_review_messages(request_id)
            await callback.answer("Tayyor", show_alert=True)
            return
        if db.get_notification_settings(user_id).get("pro_updates"):
            try:
                await callback.bot.send_message(chat_id=user_id, text="❌ PRO rad etildi.")
            except (TelegramBadRequest, TelegramForbiddenError):
                pass
        await callback.answer("❌ Rad")
    await close_review_messages(callback.bot, db.get_payment_request_review_messages(request_id))
    db.clear_payment_request_review_messages(request_id)


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
async def ad_channel_delete_menu( callback: CallbackQuery) -> None:
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
        await message.answer("📭 E'lon yo'q.")
        return
    for ad in ads:
        sent = await send_ad_preview(
            message,
            title=str(ad.get("title") or ""),
            description=str(ad.get("description") or ""),
            photo_file_id=str(ad.get("photo_file_id") or ""),
            button_text=str(ad.get("button_text") or ""),
            button_url=str(ad.get("button_url") or ""),
            footer_text=f"👤 {ad.get('user_tg_id')}\n🆔 {ad.get('id')}",
            reply_markup=build_ad_review_kb(str(ad.get("id") or "")),
        )
        if sent:
            db.add_ad_review_message(str(ad.get("id") or ""), message.from_user.id, sent.message_id)


@router.callback_query(F.data.startswith("ad:channel:"))
async def ad_choose_channel(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message or not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    ad_id = callback.data.split(":", 2)[2]
    channels = db.list_ad_channels()
    ad = db.get_ad(ad_id)
    if not ad or str(ad.get("status") or "") != "pending":
        await close_review_messages(callback.bot, db.get_ad_review_messages(ad_id))
        db.clear_ad_review_messages(ad_id)
        await callback.answer("Tayyor", show_alert=True)
        return
    if not channels:
        await callback.answer("Avval kanal qo'shing", show_alert=True)
        return
    await callback.message.answer(
        "📡 Kanal tanlang:",
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
    if str(ad.get("status") or "") != "pending":
        await close_review_messages(callback.bot, db.get_ad_review_messages(ad_id))
        db.clear_ad_review_messages(ad_id)
        await callback.answer("Tayyor", show_alert=True)
        return
    channel_ref = str(channel.get("channel_ref") or "")
    try:
        await post_ad_to_channel(callback.bot, channel_ref, ad)
    except (TelegramBadRequest, TelegramForbiddenError) as exc:
        await callback.answer("Kanalga joylab bo'lmadi", show_alert=True)
        if callback.message:
            await callback.message.answer(f"❌ Joylashda xatolik: {exc}")
        return
    if not db.resolve_ad(ad_id, "posted", reviewed_by=callback.from_user.id, channel_ref=channel_ref):
        await close_review_messages(callback.bot, db.get_ad_review_messages(ad_id))
        db.clear_ad_review_messages(ad_id)
        await callback.answer("Tayyor", show_alert=True)
        return
    user_id = int(ad.get("user_tg_id") or 0)
    if db.get_notification_settings(user_id).get("ads_updates"):
        try:
            await callback.bot.send_message(
                chat_id=user_id,
                text=f"✅ E'lon joylandi: {channel.get('title') or channel_ref}",
            )
        except (TelegramBadRequest, TelegramForbiddenError):
            pass
    await close_review_messages(callback.bot, db.get_ad_review_messages(ad_id))
    db.clear_ad_review_messages(ad_id)
    await callback.answer("✅ Joylandi")


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
    if str(ad.get("status") or "") != "pending":
        await close_review_messages(callback.bot, db.get_ad_review_messages(ad_id))
        db.clear_ad_review_messages(ad_id)
        await callback.answer("Tayyor", show_alert=True)
        return
    if not db.resolve_ad(ad_id, "rejected", reviewed_by=callback.from_user.id):
        await close_review_messages(callback.bot, db.get_ad_review_messages(ad_id))
        db.clear_ad_review_messages(ad_id)
        await callback.answer("Tayyor", show_alert=True)
        return
    user_id = int(ad.get("user_tg_id") or 0)
    if db.get_notification_settings(user_id).get("ads_updates"):
        try:
            await callback.bot.send_message(chat_id=user_id, text="❌ E'lon rad etildi.")
        except (TelegramBadRequest, TelegramForbiddenError):
            pass
    await close_review_messages(callback.bot, db.get_ad_review_messages(ad_id))
    db.clear_ad_review_messages(ad_id)
    await callback.answer("❌ Rad")


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


@router.callback_query(F.data.startswith("serial_page:"))
async def serial_page(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer()
        return
    parts = callback.data.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2].isdigit():
        await callback.answer("Noto'g'ri so'rov")
        return
    serial_id = parts[1]
    page = int(parts[2])

    ok, channels = await ensure_subscription(callback.from_user.id, callback.bot)
    if not ok:
        await ask_for_subscription(callback.message, channels)
        await callback.answer("Obuna kerak")
        return

    serial = db.get_serial(serial_id)
    episodes = db.list_serial_episodes(serial_id)
    if not serial or not episodes:
        await callback.answer("Serial topilmadi", show_alert=True)
        return

    is_favorite = db.is_favorite(callback.from_user.id, "serial", serial_id)
    episode_numbers = [row["episode_number"] for row in episodes]
    try:
        await callback.message.edit_reply_markup(
            reply_markup=build_serial_episodes_kb(
                serial_id=serial_id,
                episode_numbers=episode_numbers,
                is_favorite=is_favorite,
                page=page,
            )
        )
    except TelegramBadRequest:
        pass
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
        await ask_for_subscription(callback.message, channels)
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
        media_type = str(episode.get("media_type") or "")
        file_id = str(episode.get("file_id") or "")
        if media_type == "stream" or not file_id:
            await callback.message.answer(
                caption or f"🎬 {serial.get('title') or 'Serial'}",
                reply_markup=merge_inline_keyboards(build_mini_app_open_kb(f"s_{serial_id}"), nav_kb),
                protect_content=content_should_be_protected(callback.from_user.id),
            )
        else:
            await send_stored_media(
                callback.message,
                media_type=media_type,
                file_id=file_id,
                caption=caption if caption else None,
                requester_id=callback.from_user.id,
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
    code = normalize_code(text)
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not code:
        await message.answer("Kod bo'sh bo'lmasin.")
        return
    if db.code_exists(code) or (code != text and db.code_exists(text)):
        await message.answer("⚠️ Bu kod allaqachon band. Boshqa kod yuboring.")
        return
    await state.update_data(code=code)
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
        "🏷 Metadata yuboring (format: yil|sifat|janr1,janr2|ko'rinish).\n"
        "Ko'rinish: public/pro/admin (ixtiyoriy).\n"
        "Masalan: 2024|1080p|action,drama|pro\n"
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
            "⚠️ Format noto'g'ri. To'g'ri format: yil|sifat|janr1,janr2|ko'rinish\n"
            "Masalan: 2024|1080p|action,drama|pro\n"
            "Yoki: -"
        )
        return

    await state.update_data(
        year=metadata["year"],
        quality=metadata["quality"],
        genres=metadata["genres"],
        visibility=metadata.get("visibility", "public"),
    )
    await state.set_state(AddMovieState.waiting_media)
    await message.answer(
        "📤 Endi media yuboring:\n"
        "• video / document / photo\n"
        "• yoki file_id / link matn\n"
        "• yoki sifatlar ro'yxati (360p=URL|480p=URL|720p=URL)\n"
        "• yoki JSON: {\"360p\":\"URL\",\"480p\":\"URL\",\"720p\":\"URL\"}\n\n"
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
    stream_sources = ""
    if media_type == "stream":
        stream_sources = file_id
        file_id = ""
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
        stream_sources=stream_sources,
        visibility=str(data.get("visibility") or "public"),
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
        if saved_movie:
            try:
                await maybe_post_movie_to_channel(message.bot, str(saved_movie.get("id") or ""))
            except Exception:
                logging.exception("content_post: failed to post movie to channel")
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
    code = normalize_code(text)
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not code:
        await message.answer("Kod bo'sh bo'lmasin.")
        return
    if db.code_exists(code) or (code != text and db.code_exists(text)):
        await message.answer("⚠️ Bu kod allaqachon band. Boshqa kod yuboring.")
        return
    await state.update_data(code=code)
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
        "🏷 Metadata yuboring (format: yil|sifat|janr1,janr2|ko'rinish).\n"
        "Ko'rinish: public/pro/admin (ixtiyoriy).\n"
        "Masalan: 2024|1080p|action,drama|pro\n"
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
            "⚠️ Format noto'g'ri. To'g'ri format: yil|sifat|janr1,janr2|ko'rinish\n"
            "Masalan: 2024|1080p|action,drama|pro\n"
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
        visibility=metadata.get("visibility", "public"),
    )
    if serial_id is None:
        await state.clear()
        await message.answer("⚠️ Bu kod allaqachon mavjud.", reply_markup=admin_menu_kb())
        return

    await state.update_data(
        year=metadata["year"],
        quality=metadata["quality"],
        genres=metadata["genres"],
        visibility=metadata.get("visibility", "public"),
        serial_id=serial_id,
        next_episode=1,
        episodes_added=0,
    )
    await state.set_state(AddSerialState.waiting_episode)
    await message.answer(
        "🎬 Endi 1-qismni yuboring.\n"
        "Video/document/photo yoki file_id/link yuborishingiz mumkin.\n"
        "Sifatlar uchun: 360p=URL|480p=URL|720p=URL yoki JSON.\n"
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
    stream_sources = ""
    if media_type == "stream":
        stream_sources = file_id
        file_id = ""

    created = db.add_serial_episode(serial_id, next_episode, media_type, file_id, stream_sources=stream_sources)
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
    code = normalize_code(text)
    if is_cancel_text(text):
        await state.clear()
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
        return
    if not code:
        await message.answer("Kod yuboring.")
        return
    deleted_types: list[str] = []
    if db.delete_movie(code) or (code != text and db.delete_movie(text)):
        deleted_types.append("kino")

    serial = db.get_serial_by_code(code) or ((db.get_serial_by_code(text) if code != text else None))
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
    movie = db.get_movie(code) or (db.get_movie(text) if code != text else None)
    if movie:
        stored_code = str(movie.get("code") or code).strip() or code
        current_title = str(movie.get("title") or "")
        current_description = str(movie.get("description") or "")
        await state.update_data(
            edit_type="movie",
            edit_code=stored_code,
            movie_title=current_title,
            movie_description=current_description,
            movie_media_type=str(movie.get("media_type") or ""),
            movie_file_id=str(movie.get("file_id") or ""),
            movie_stream_sources=str(movie.get("stream_sources") or ""),
            movie_visibility=str(movie.get("visibility") or "public"),
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

    serial = db.get_serial_by_code(code) or (db.get_serial_by_code(text) if code != text else None)
    if serial:
        serial_id = str(serial["id"])
        next_episode = db.get_next_serial_episode_number(serial_id)
        stored_code = str(serial.get("code") or code).strip() or code
        serial_title = str(serial.get("title") or stored_code)
        await state.update_data(
            edit_type="serial",
            serial_id=serial_id,
            serial_code=stored_code,
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
        "🏷 Yangi metadata yuboring (format: yil|sifat|janr1,janr2|ko'rinish).\n"
        "Masalan: 2024|1080p|action,drama|pro\n"
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
    old_visibility = str(data.get("movie_visibility") or "public")

    if text == "-":
        year = old_year if isinstance(old_year, int) else None
        quality = old_quality
        genres = old_genres
        visibility = old_visibility
    else:
        metadata = parse_metadata_input(text)
        if metadata is None:
            await message.answer(
                "⚠️ Format noto'g'ri. To'g'ri format: yil|sifat|janr1,janr2|ko'rinish\n"
                "Masalan: 2024|1080p|action,drama|pro\n"
                "Yoki: -"
            )
            return
        year = metadata["year"]
        quality = metadata["quality"]
        genres = metadata["genres"]
        visibility = metadata.get("visibility", "public")

    await state.update_data(
        movie_new_year=year,
        movie_new_quality=quality,
        movie_new_genres=genres,
        movie_new_visibility=visibility,
    )
    await state.set_state(EditContentState.waiting_movie_media)
    await message.answer(
        "🎞 Yangi media yuboring (video/document/photo yoki file_id/link).\n"
        "Sifatlar uchun: 360p=URL|480p=URL|720p=URL yoki JSON.\n"
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
        stream_sources = str(data.get("movie_stream_sources") or "")
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
        stream_sources = ""
        if media_type == "stream":
            stream_sources = file_id
            file_id = ""
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
    visibility = str(data.get("movie_new_visibility") or data.get("movie_visibility") or "public")
    if not title or not media_type or (not file_id and not stream_sources and media_type != "stream"):
        await state.clear()
        await message.answer("⚠️ Tahrirlash uchun ma'lumot yetarli emas.", reply_markup=admin_menu_kb())
        return

    updated = db.update_movie(
        code=code,
        title=title,
        description=description,
        media_type=media_type,
        file_id=file_id,
        stream_sources=stream_sources,
        visibility=visibility,
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
    stream_sources = ""
    if media_type == "stream":
        stream_sources = file_id
        file_id = ""

    next_episode = int(data.get("next_episode", 1))
    created = db.add_serial_episode(serial_id, next_episode, media_type, file_id, stream_sources=stream_sources)
    if not created:
        # Parallel update bo'lsa, keyingi bo'sh raqamni hisoblab yana urinib ko'ramiz.
        next_episode = db.get_next_serial_episode_number(serial_id)
        created = db.add_serial_episode(serial_id, next_episode, media_type, file_id, stream_sources=stream_sources)
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
        "📣 Xabar yuboring.\nMatn, rasm, video, gif, document, audio, voice bo'ladi.",
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
        broadcast_notice_text=extract_message_notice_text(message),
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
    notice_text = str(data.get("broadcast_notice_text") or "").strip()
    if success > 0 and notice_text:
        db.set_site_notice(notice_text, admin_id=message.from_user.id, link=get_mini_app_url())
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
        "🔎 Kino yoki serial nomini yozing",
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
    if len(text) < 1:
        await message.answer("Nom yoki kod yozing.")
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
        f"🎬 {len(results)} ta natija topildi",
        reply_markup=kb,
    )


@router.message(F.text.in_({BTN_FAVORITES, "Sevimlilarim"}))
async def list_favorites(message: Message) -> None:
    if not message.from_user:
        return
    favorites = db.list_favorites(message.from_user.id, limit=100)
    if not favorites:
        await message.answer("⭐ Saqlangan kinolar yo'q")
        return
    await message.answer(
        "⭐ Saqlangan kinolar",
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
    if current_text.startswith("⭐ sevimlilar ro'yxati") or current_text.startswith("⭐ saqlangan"):
        favorites = db.list_favorites(callback.from_user.id, limit=100)
        if favorites:
            await callback.message.edit_text(
                "⭐ Saqlangan kinolar",
                reply_markup=build_favorites_kb(favorites),
            )
        else:
            await callback.message.edit_text("⭐ Saqlangan kinolar yo'q")
        return

    try:
        if content_type == "movie":
            await refresh_movie_card_message(callback.message, content_ref, callback.from_user.id)
        else:
            serial = db.get_serial(content_ref)
            if serial:
                episodes = db.list_serial_episodes(content_ref)
                episode_numbers = [row["episode_number"] for row in episodes]
                is_favorite = db.is_favorite(callback.from_user.id, "serial", content_ref)
                await callback.message.edit_reply_markup(
                    reply_markup=build_serial_episodes_kb(
                        serial_id=content_ref,
                        episode_numbers=episode_numbers,
                        is_favorite=is_favorite,
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
    try:
        if content_type == "movie":
            await refresh_movie_card_message(callback.message, content_ref, callback.from_user.id)
            try:
                await sync_movie_channel_posts(callback.bot, content_ref)
            except Exception:
                logging.exception("content_post: failed to sync movie posts (movie_id=%s)", content_ref)
        elif content_type == "serial":
            episodes = db.list_serial_episodes(content_ref)
            await callback.message.edit_reply_markup(
                reply_markup=build_serial_episodes_kb(
                    serial_id=content_ref,
                    episode_numbers=[row["episode_number"] for row in episodes],
                    is_favorite=db.is_favorite(callback.from_user.id, "serial", content_ref),
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


@router.callback_query(F.data.startswith("preview:"))
async def preview_content_from_callback(callback: CallbackQuery, state: FSMContext) -> None:
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
        try:
            if content_type == "movie":
                await state.update_data(pending_movie_preview_id=content_ref)
            else:
                await state.update_data(pending_serial_id=content_ref)
        except Exception:
            pass
        await ask_for_subscription(callback.message, channels)
        await callback.answer("Obuna kerak")
        return

    try:
        if content_type == "movie":
            sent = await send_movie_preview_by_id(callback.message, content_ref, callback.from_user.id)
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
    await callback.answer()


@router.callback_query(F.data.startswith("open:"))
async def open_content_from_callback(callback: CallbackQuery, state: FSMContext) -> None:
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
        try:
            if content_type == "movie":
                await state.update_data(pending_movie_id=content_ref)
            else:
                await state.update_data(pending_serial_id=content_ref)
        except Exception:
            pass
        await ask_for_subscription(callback.message, channels)
        await callback.answer("Obuna kerak")
        return

    try:
        if content_type == "movie":
            sent = await send_movie_by_id(callback.message, content_ref, callback.from_user.id)
            if not sent:
                await callback.answer("Kino topilmadi", show_alert=True)
                return
            # If this was a movie preview card, refresh its caption to show updated views.
            try:
                if callback.message.reply_markup and any(
                    (btn.callback_data or "").startswith(f"react:like:movie:{content_ref}")
                    for row in callback.message.reply_markup.inline_keyboard
                    for btn in row
                ):
                    await refresh_movie_preview_message(callback.message, content_ref, callback.from_user.id)
            except Exception:
                pass
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
    if len(query) < 1 and not query.isdigit():
        await inline_query.answer(
            [],
            is_personal=True,
            cache_time=1,
            switch_pm_text="Nom yoki kod kiriting",
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
        normalized = normalize_lookup_text(query)
        tokens = [token for token in normalized.split() if token]
        digits = "".join(ch for ch in query if ch.isdigit())
        fallback_queries = []
        if digits and digits != query:
            fallback_queries.append(digits)
        fallback_queries.extend([token for token in tokens if token != normalized])
        seen_keys: set[str] = set()
        fallback_items: list[dict[str, Any]] = []
        for token in fallback_queries:
            for row in db.search_content(query=token, limit=30):
                content_type = str(row.get("content_type") or "")
                content_id = str(row.get("id") or "")
                key = f"{content_type}:{content_id}"
                if not content_type or not content_id or key in seen_keys:
                    continue
                seen_keys.add(key)
                fallback_items.append(row)
            if len(fallback_items) >= 10:
                break
        if fallback_items:
            items = fallback_items
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
        if not is_content_visible_for_user(item, inline_query.from_user.id):
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
        inline_rows = [[InlineKeyboardButton(text="📥 Botda ochish", url=deeplink)]]
        miniapp_url = get_mini_app_launch_url_with_payload(payload)
        if miniapp_url:
            inline_rows.append([InlineKeyboardButton(text="🌐 Mini App", url=miniapp_url)])
        reply_markup = InlineKeyboardMarkup(inline_keyboard=inline_rows)
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


@router.message(StateFilter(None), F.text.func(lambda value: bool(value and value.strip().lower() in LEGACY_MENU_TEXTS)))
async def legacy_menu_router(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return
    text = (message.text or "").strip().lower()
    if text in {BTN_ADMIN_PANEL.lower(), "🛠 admin panel", "🛠 panel"}:
        await open_admin_panel(message, state)
        return
    if text in {BTN_MINI_APP.lower(), "ilova"}:
        launch_url = get_mini_app_launch_url()
        if launch_url:
            await message.answer(
                "🧩 Mini ilovani ochish uchun tugmani bosing.",
                reply_markup=build_mini_app_open_kb(),
            )
        else:
            await message.answer("Ilova havolasi hali sozlanmagan.")
        return
    if text in {BTN_BACK.lower(), "⬅️ ortga", "🏠 menyu"}:
        await back_to_main(message, state)
        return
    if text in {BTN_SETTINGS.lower(), "⚙️ sozlamalar"}:
        await open_settings(message, state)
        return
    if text in {BTN_SEARCH_NAME.lower(), "🔎 nom bo'yicha qidirish", "🔎 qidirish"}:
        await search_by_name_start(message, state)
        return
    if text in {BTN_FAVORITES.lower(), "⭐ sevimlilarim", "⭐ saqlangan"}:
        await list_favorites(message)
        return
    if text in {"🔥 trend", "🔥 trending", "trending", "trend", "🔥 top kinolar", "top kinolar"}:
        await top_viewed_content(message)
        return
    if text in {BTN_NOTIFICATIONS.lower(), "🔔 bildirishnomalar", "🔔 sozlama"}:
        await notification_settings(message)
        return
    if text in {BTN_HELP.lower(), "❓ yordam"}:
        await help_menu(message)
        return
    if text in {BTN_PRO_BUY.lower(), "👑 pro olish", "👑 pro"}:
        await pro_buy(message)
        return
    if text in {BTN_PRO_STATUS.lower(), "💎 pro holatim", "💎 holat"}:
        await pro_status(message)
        return
    if text in {BTN_CREATE_AD.lower(), "📢 e'lon berish", "📢 e'lon"}:
        await create_ad_start(message, state)
        return
    if text in {BTN_MY_ADS.lower(), "🗂 e'lonlarim", "🗂 postlarim"}:
        await my_ads(message)
        return
    if text in {BTN_SUBS.lower(), "📢 majburiy obuna", "📢 obuna"}:
        await mandatory_subscriptions_menu(message)
        return
    if text in {BTN_ADD_MOVIE.lower(), "➕ kino qo'shish", "➕ kino"}:
        await add_movie_start(message, state)
        return
    if text in {BTN_ADD_SERIAL.lower(), "📺 serial qo'shish", "📺 serial"}:
        await add_serial_start(message, state)
        return
    if text in {BTN_DEL_MOVIE.lower(), "🗑 kino o'chirish", "🗑 o'chirish"}:
        await delete_movie_start(message, state)
        return
    if text in {BTN_EDIT_CONTENT.lower(), "✏️ kontent tahrirlash", "✏️ tahrir"}:
        await edit_content_start(message, state)
        return
    if text in {BTN_LIST_MOVIES.lower(), "📚 kino va serial ro'yxati", "📚 baza"}:
        await movie_list(message)
        return
    if text in {BTN_STATS.lower(), "📊 statistika", "📊 stat"}:
        await stats(message)
        return
    if text in {BTN_REQUESTS.lower(), "📥 so'rovlar", "📥 so'rov"}:
        await requests_dashboard(message)
        return
    if text in {BTN_BROADCAST.lower(), "📣 habar yuborish", "📣 xabar yuborish", "📣 xabar"}:
        await broadcast_start(message, state)
        return
    if text in {BTN_ADD_ADMIN.lower(), "👤 admin qo'shish", "👤 admin"}:
        await add_admin_start(message, state)
        return
    if text in {BTN_RANDOM_CODES.lower(), "🎲 random kod", "🎲 kod"}:
        await random_missing_codes(message)
        return
    if text in {BTN_PRO_MANAGE.lower(), "👑 pro boshqarish", "👑 pro boshqaruv"}:
        await pro_manage_start(message, state)
        return
    if text in {BTN_PRO_PRICE.lower(), "💰 pro narxi", "💰 pro narx"}:
        await pro_price_start(message, state)
        return
    if text in {BTN_PRO_DURATION.lower(), "⏳ pro muddati", "⏳ pro muddat"}:
        await pro_duration_start(message, state)
        return
    if text in {BTN_PRO_REQUESTS.lower(), "💳 pro so'rovlar", "💳 pro so'rov"}:
        await pro_requests(message)
        return
    if text in {BTN_CONTENT_MODE.lower(), "media rejimi", "🔐 media"}:
        await content_mode_menu(message)
        return
    if text in {BTN_ADS.lower(), "📰 e'lonlar", "📰 postlar"}:
        await ads_review(message)
        return
    if text in {BTN_AD_CHANNELS.lower(), "📡 e'lon kanalari", "📡 kanallar"}:
        await ad_channels_menu(message)
        return


@router.message(StateFilter(None), F.text)
async def handle_code_request(message: Message, state: FSMContext) -> None:
    if not message.from_user:
        return

    text = (message.text or "").strip()
    if not text:
        return

    protected_words = {
        BTN_ADMIN_PANEL.lower(),
        BTN_MINI_APP.lower(),
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
        BTN_RANDOM_MOVIE.lower(),
        BTN_FAVORITES.lower(),
        BTN_TOP_VIEWED.lower(),
        BTN_SETTINGS.lower(),
        BTN_NOTIFICATIONS.lower(),
        BTN_FREE_PRO.lower(),
        BTN_PRO_BUY.lower(),
        BTN_PRO_STATUS.lower(),
        BTN_HELP.lower(),
        BTN_CREATE_AD.lower(),
        BTN_MY_ADS.lower(),
        BTN_PRO_MANAGE.lower(),
        BTN_PRO_PRICE.lower(),
        BTN_PRO_DURATION.lower(),
        BTN_PRO_REQUESTS.lower(),
        BTN_CONTENT_MODE.lower(),
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
        "kino tavsiyasi",
        "tavsiya",
        "sevimlilarim",
        "top ko'rilganlar",
        "sozlamalar",
        "bildirishnomalar",
        "pro olish",
        "pro holatim",
        "yordam",
        "e'lon berish",
        "e'lonlarim",
        "pro boshqarish",
        "pro narxi",
        "pro muddati",
        "pro so'rovlar",
        "media rejimi",
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
    movie = db.get_movie(code) or (db.get_movie(text) if code != text else None)
    if not movie and code.isdigit():
        movie = db.get_movie(f"k{code}") or db.get_movie(f"K{code}")
    if movie:
        try:
            sent = await send_movie_preview_by_id(message, str(movie["id"]), message.from_user.id)
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

    serial = db.get_serial_by_code(code) or (db.get_serial_by_code(text) if code != text else None)
    if not serial:
        db.log_request(message.from_user.id, code, "not_found")
        await state.update_data(
            pending_request_query=code,
            pending_request_type="code",
        )
        await message.answer(
            "❌ Bunday kino topilmadi\n\n"
            "Xohlasangiz so'rov qoldiring.\n"
            "Kontent qo'shilganda sizga yuboramiz.",
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
    daily_task = asyncio.create_task(daily_recommendation_loop(bot))
    try:
        await dp.start_polling(bot)
    finally:
        daily_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await daily_task


if __name__ == "__main__":
    asyncio.run(main())
