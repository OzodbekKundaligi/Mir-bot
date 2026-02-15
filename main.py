import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable
from urllib.parse import urlparse

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatMemberStatus, ContentType
from aiogram.exceptions import ClientDecodeError, TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
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


class Database:
    def __init__(self, mongo_uri: str, db_name: str) -> None:
        self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        self.db = self.client[db_name]

        self.admins = self.db["admins"]
        self.users = self.db["users"]
        self.required_channels = self.db["required_channels"]
        self.movies = self.db["movies"]
        self.serials = self.db["serials"]
        self.serial_episodes = self.db["serial_episodes"]
        self.requests_log = self.db["requests_log"]

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

    def init_indexes(self) -> None:
        self.admins.create_index("tg_id", unique=True)
        self.users.create_index("tg_id", unique=True)
        self.required_channels.create_index("channel_ref", unique=True)
        self.required_channels.create_index([("is_active", ASCENDING), ("created_at", DESCENDING)])
        self.movies.create_index("code", unique=True)
        self.movies.create_index([("created_at", DESCENDING)])
        self.serials.create_index("code", unique=True)
        self.serials.create_index([("created_at", DESCENDING)])
        self.serial_episodes.create_index(
            [("serial_id", ASCENDING), ("episode_number", ASCENDING)],
            unique=True,
        )
        self.requests_log.create_index([("created_at", DESCENDING)])

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

    def add_user(self, tg_id: int, full_name: str) -> None:
        now = utc_now_iso()
        self.users.update_one(
            {"tg_id": tg_id},
            {
                "$set": {"full_name": full_name},
                "$setOnInsert": {"joined_at": now},
            },
            upsert=True,
        )

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

    def add_movie(self, movie: Movie) -> bool:
        now = utc_now_iso()
        doc = {
            "code": movie.code,
            "title": movie.title,
            "description": movie.description,
            "media_type": movie.media_type,
            "file_id": movie.file_id,
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
            {"code": 1, "title": 1, "description": 1, "media_type": 1, "file_id": 1},
        )
        return self._doc_without_object_id(doc)

    def list_movies(self, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.movies.find(
            {},
            {"code": 1, "title": 1, "created_at": 1},
        ).sort("created_at", DESCENDING).limit(limit)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def add_serial(self, code: str, title: str, description: str) -> str | None:
        now = utc_now_iso()
        doc = {
            "code": code,
            "title": title,
            "description": description,
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
            {"code": 1, "title": 1, "description": 1},
        )
        return self._doc_without_object_id(doc)

    def get_serial_by_code(self, code: str) -> dict[str, Any] | None:
        doc = self.serials.find_one(
            {"code": code},
            {"code": 1, "title": 1, "description": 1},
        )
        return self._doc_without_object_id(doc)

    def list_serial_episodes(self, serial_id: str) -> list[dict[str, Any]]:
        cursor = self.serial_episodes.find(
            {"serial_id": serial_id},
            {"episode_number": 1},
        ).sort("episode_number", ASCENDING)
        return [self._doc_without_object_id(doc) for doc in cursor if doc]

    def get_serial_episode(self, serial_id: str, episode_number: int) -> dict[str, Any] | None:
        doc = self.serial_episodes.find_one(
            {"serial_id": serial_id, "episode_number": episode_number},
            {"media_type": 1, "file_id": 1},
        )
        return self._doc_without_object_id(doc)

    def code_exists(self, code: str) -> bool:
        if self.movies.find_one({"code": code}, {"_id": 1}):
            return True
        return self.serials.find_one({"code": code}, {"_id": 1}) is not None

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
        return {
            "users": users,
            "movies": movies,
            "serials": serials,
            "serial_episodes": serial_episodes,
            "channels": channels,
            "requests": requests,
        }


class AddChannelState(StatesGroup):
    waiting_input = State()


class AddMovieState(StatesGroup):
    waiting_code = State()
    waiting_title = State()
    waiting_description = State()
    waiting_media = State()


class AddSerialState(StatesGroup):
    waiting_code = State()
    waiting_title = State()
    waiting_description = State()
    waiting_episode = State()


class DeleteMovieState(StatesGroup):
    waiting_code = State()


class AddAdminState(StatesGroup):
    waiting_tg_id = State()


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
BTN_LIST_MOVIES = "📚 Kino ro'yxati"
BTN_STATS = "📊 Statistika"
BTN_ADD_ADMIN = "👤 Admin qo'shish"
BTN_BACK = "⬅️ Ortga"
BTN_CANCEL = "❌ Bekor qilish"
BTN_SERIAL_DONE = "✅ Serialni yakunlash"


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


def main_menu_kb(is_admin: bool) -> ReplyKeyboardMarkup | ReplyKeyboardRemove:
    if is_admin:
        return ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text=BTN_ADMIN_PANEL)]],
            resize_keyboard=True,
        )
    return ReplyKeyboardRemove()


def admin_menu_kb() -> ReplyKeyboardMarkup:
    buttons = [
        [KeyboardButton(text=BTN_SUBS)],
        [KeyboardButton(text=BTN_ADD_MOVIE), KeyboardButton(text=BTN_ADD_SERIAL)],
        [KeyboardButton(text=BTN_DEL_MOVIE), KeyboardButton(text=BTN_LIST_MOVIES)],
        [KeyboardButton(text=BTN_STATS), KeyboardButton(text=BTN_ADD_ADMIN)],
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


def serial_upload_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BTN_SERIAL_DONE), KeyboardButton(text=BTN_CANCEL)]],
        resize_keyboard=True,
    )


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


def build_movie_caption(title: str | None, description: str | None) -> str:
    title_text = (title or "").strip()
    description_text = (description or "").strip()
    if title_text and description_text:
        return f"{title_text}\n\n{description_text}"
    return title_text or description_text


def build_serial_caption(title: str | None, description: str | None, episodes_count: int) -> str:
    base = build_movie_caption(title, description)
    tail = f"🎞 Qismlar soni: {episodes_count}"
    if base:
        return f"{base}\n\n{tail}"
    return tail


def build_serial_episodes_kb(serial_id: str, episode_numbers: list[int]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for number in episode_numbers:
        builder.button(text=str(number), callback_data=f"serial_ep:{serial_id}:{number}")
    builder.adjust(5)
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
) -> None:
    if media_type == "video":
        await message.answer_video(file_id, caption=caption)
    elif media_type == "document":
        await message.answer_document(file_id, caption=caption)
    elif media_type == "photo":
        await message.answer_photo(file_id, caption=caption)
    elif media_type == "animation":
        await message.answer_animation(file_id, caption=caption)
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
            caption=caption,
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
                caption=caption,
            )
        else:
            if caption:
                await message.answer(f"{caption}\n\nLink: {file_id}")
            else:
                await message.answer(f"Link: {file_id}")
    else:
        if caption:
            await message.answer(f"{caption}\n\nID: {file_id}")
        else:
            await message.answer(f"ID: {file_id}")


load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:wGVAMNxMWZgocdRVBduRDnRlJePweOay@metro.proxy.rlwy.net:36399").strip()
MONGODB_DB = os.getenv("MONGODB_DB", "kino_bot").strip() or "kino_bot"
ADMIN_IDS = parse_admin_ids(os.getenv("ADMIN_IDS", ""))

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


async def ensure_subscription(user_id: int, bot: Bot) -> tuple[bool, list[dict[str, Any]]]:
    channels = db.get_required_channels()
    if not channels:
        return True, []

    not_joined: list[dict[str, Any]] = []
    for channel in channels:
        try:
            member = await bot.get_chat_member(chat_id=channel["channel_ref"], user_id=user_id)
            if not is_member_status(member.status):
                not_joined.append(channel)
        except (TelegramBadRequest, TelegramForbiddenError):
            not_joined.append(channel)
    return len(not_joined) == 0, not_joined


async def ask_for_subscription(message: Message, channels: list[dict[str, Any]]) -> None:
    text_lines = [
        "‼️ Botdan foydalanish uchun quyida keltirilgan barcha kanallarga obuna bo'lishingiz kerak!",
        "",
        "👇 Majburiy kanallar:",
    ]
    for ch in channels:
        title = ch["title"] or ch["channel_ref"]
        text_lines.append(f"• {title}")
    await message.answer("\n".join(text_lines), reply_markup=build_subscribe_keyboard(channels))


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not message.from_user:
        return

    db.add_user(message.from_user.id, message.from_user.full_name)
    admin = db.is_admin(message.from_user.id)
    text = (
        "🎬 Assalomu alaykum, Kino Qidiruvi Botga xush kelibsiz!\n\n"
        "🔎 Kino yoki serial kodini chatga yozing.\n"
        "⚡ Bot avtomatik tekshiradi va sizga natijani yuboradi."
    )
    await message.answer(text, reply_markup=main_menu_kb(admin))


@router.callback_query(F.data == "check_sub")
async def check_subscription(callback: CallbackQuery) -> None:
    user = callback.from_user
    if not user:
        return
    ok, channels = await ensure_subscription(user.id, callback.bot)
    if ok:
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
    await message.answer("🛠 Admin panelga xush kelibsiz!", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_BACK, "Ortga"}))
async def back_to_main(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.clear()
    await message.answer("🏠 Asosiy menyu", reply_markup=main_menu_kb(True))


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
        await message.answer(
            "⚠️ Private kanal uchun join link kerak.\n"
            "Format: -1001234567890|https://t.me/+invite_link"
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

    caption = build_movie_caption(serial["title"], f"{episode_number}-qism")
    try:
        await send_stored_media(
            callback.message,
            media_type=episode["media_type"],
            file_id=episode["file_id"],
            caption=caption if caption else None,
        )
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
    await message.answer("🎬 Yangi kino kodini yuboring:", reply_markup=cancel_kb())


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

    data = await state.get_data()
    movie = Movie(
        code=data["code"],
        title=data["title"],
        description=data.get("description", ""),
        media_type=media_type,
        file_id=file_id,
    )
    created = db.add_movie(movie)
    await state.clear()
    if created:
        await message.answer("✅ Kino muvaffaqiyatli saqlandi!", reply_markup=admin_menu_kb())
    else:
        await message.answer("⚠️ Bu kod allaqachon mavjud.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_ADD_SERIAL, "Serial qo'shish"}))
async def add_serial_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(AddSerialState.waiting_code)
    await message.answer("📺 Yangi serial kodi yuboring:", reply_markup=cancel_kb())


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
    data = await state.get_data()
    code = data.get("code")
    title = data.get("title")
    if not code or not title:
        await state.clear()
        await message.answer("⚠️ Sessiya topilmadi, qaytadan boshlang.", reply_markup=admin_menu_kb())
        return
    serial_id = db.add_serial(
        code=str(code),
        title=str(title),
        description=description,
    )
    if serial_id is None:
        await state.clear()
        await message.answer("⚠️ Bu kod allaqachon mavjud.", reply_markup=admin_menu_kb())
        return

    await state.update_data(
        description=description,
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
        await state.clear()
        await message.answer(
            f"✅ Serial muvaffaqiyatli saqlandi!\n🎞 Jami qismlar: {episodes_added}",
            reply_markup=admin_menu_kb(),
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


@router.message(F.text.in_({BTN_DEL_MOVIE, "Kino o'chirish"}))
async def delete_movie_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(DeleteMovieState.waiting_code)
    await message.answer("🗑 O'chirish uchun kino kodini yuboring:", reply_markup=cancel_kb())


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
    deleted = db.delete_movie(text)
    await state.clear()
    if deleted:
        await message.answer("✅ Kino o'chirildi.", reply_markup=admin_menu_kb())
    else:
        await message.answer("❌ Bu kod bo'yicha kino topilmadi.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_LIST_MOVIES, "Kino ro'yxati"}))
async def movie_list(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    movies = db.list_movies()
    if not movies:
        await message.answer("📭 Kino bazasi hozircha bo'sh.")
        return
    lines = ["🎬 Oxirgi kinolar:"]
    for item in movies:
        lines.append(f"{item['code']} - {item['title']}")
    await message.answer("\n".join(lines))


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
        f"📥 So'rovlar: {s['requests']}"
    )
    await message.answer(text)


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


@router.message(F.text.in_({BTN_CANCEL, "Bekor qilish"}))
async def cancel_any(message: Message, state: FSMContext) -> None:
    await state.clear()
    if message.from_user and db.is_admin(message.from_user.id):
        await message.answer("❌ Bekor qilindi.", reply_markup=admin_menu_kb())
    else:
        await message.answer("❌ Bekor qilindi.", reply_markup=main_menu_kb(False))


@router.message(StateFilter(None), F.text)
async def handle_code_request(message: Message) -> None:
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
        BTN_LIST_MOVIES.lower(),
        BTN_STATS.lower(),
        BTN_ADD_ADMIN.lower(),
        BTN_BACK.lower(),
        BTN_CANCEL.lower(),
        BTN_SERIAL_DONE.lower(),
        "admin panel",
        "majburiy obuna",
        "kino qo'shish",
        "serial qo'shish",
        "kino o'chirish",
        "kino ro'yxati",
        "statistika",
        "admin qo'shish",
        "ortga",
        "serialni yakunlash",
        "bekor qilish",
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
        caption = build_movie_caption(movie["title"], movie["description"])
        try:
            await send_stored_media(
                message,
                media_type=movie["media_type"],
                file_id=movie["file_id"],
                caption=caption if caption else None,
            )
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
        await message.answer("❌ Bunday kod topilmadi. Kodni tekshirib qayta yuboring.")
        return

    episodes = db.list_serial_episodes(serial["id"])
    if not episodes:
        db.log_request(message.from_user.id, code, "serial_no_episodes")
        await message.answer("📭 Bu serialga hali qism qo'shilmagan.")
        return

    episode_numbers = [row["episode_number"] for row in episodes]
    serial_caption = build_serial_caption(
        serial["title"],
        serial["description"],
        episodes_count=len(episode_numbers),
    )
    await message.answer(
        f"{serial_caption}\n\n👇 Kerakli qismni tanlang:",
        reply_markup=build_serial_episodes_kb(serial["id"], episode_numbers),
    )
    db.log_request(message.from_user.id, code, "success")


async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
