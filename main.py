import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable
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
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv


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
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.init_tables()

    def init_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tg_id INTEGER UNIQUE NOT NULL,
                added_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tg_id INTEGER UNIQUE NOT NULL,
                full_name TEXT,
                joined_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS required_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_ref TEXT UNIQUE NOT NULL,
                title TEXT,
                join_link TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                media_type TEXT NOT NULL,
                file_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS requests_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_tg_id INTEGER NOT NULL,
                movie_code TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def seed_admins(self, admin_ids: Iterable[int]) -> None:
        now = utc_now_iso()
        for admin_id in admin_ids:
            self.conn.execute(
                "INSERT OR IGNORE INTO admins (tg_id, added_at) VALUES (?, ?)",
                (admin_id, now),
            )
        self.conn.commit()

    def is_admin(self, tg_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM admins WHERE tg_id = ? LIMIT 1",
            (tg_id,),
        ).fetchone()
        return bool(row)

    def add_admin(self, tg_id: int) -> bool:
        now = utc_now_iso()
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO admins (tg_id, added_at) VALUES (?, ?)",
            (tg_id, now),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def add_user(self, tg_id: int, full_name: str) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO users (tg_id, full_name, joined_at)
            VALUES (?, ?, ?)
            ON CONFLICT(tg_id) DO UPDATE SET full_name = excluded.full_name
            """,
            (tg_id, full_name, now),
        )
        self.conn.commit()

    def add_required_channel(
        self,
        channel_ref: str,
        title: str | None = None,
        join_link: str | None = None,
    ) -> bool:
        now = utc_now_iso()
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO required_channels (
                channel_ref, title, join_link, is_active, created_at
            ) VALUES (?, ?, ?, 1, ?)
            """,
            (channel_ref, title, join_link, now),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def remove_required_channel(self, channel_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM required_channels WHERE id = ?",
            (channel_id,),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_required_channels(self) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT id, channel_ref, title, join_link
            FROM required_channels
            WHERE is_active = 1
            ORDER BY id DESC
            """
        ).fetchall()
        return list(rows)

    def add_movie(self, movie: Movie) -> bool:
        now = utc_now_iso()
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO movies (code, title, description, media_type, file_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (movie.code, movie.title, movie.description, movie.media_type, movie.file_id, now),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_movie(self, code: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM movies WHERE code = ?",
            (code,),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_movie(self, code: str) -> sqlite3.Row | None:
        row = self.conn.execute(
            """
            SELECT code, title, description, media_type, file_id
            FROM movies
            WHERE code = ?
            LIMIT 1
            """,
            (code,),
        ).fetchone()
        return row

    def list_movies(self, limit: int = 50) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT code, title, created_at
            FROM movies
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return list(rows)

    def log_request(self, user_tg_id: int, movie_code: str, result: str) -> None:
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO requests_log (user_tg_id, movie_code, result, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_tg_id, movie_code, result, now),
        )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        users = self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        movies = self.conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        channels = self.conn.execute("SELECT COUNT(*) FROM required_channels").fetchone()[0]
        requests = self.conn.execute("SELECT COUNT(*) FROM requests_log").fetchone()[0]
        return {
            "users": users,
            "movies": movies,
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


BTN_SEND_CODE = "🎬 Kino kodini yuborish"
BTN_ADMIN_PANEL = "🛠 Admin panel"
BTN_SUBS = "📢 Majburiy obuna"
BTN_ADD_MOVIE = "➕ Kino qo'shish"
BTN_DEL_MOVIE = "🗑 Kino o'chirish"
BTN_LIST_MOVIES = "📚 Kino ro'yxati"
BTN_STATS = "📊 Statistika"
BTN_ADD_ADMIN = "👤 Admin qo'shish"
BTN_BACK = "⬅️ Ortga"
BTN_CANCEL = "❌ Bekor qilish"


def is_cancel_text(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized in {BTN_CANCEL.lower(), "bekor qilish"}


def main_menu_kb(is_admin: bool) -> ReplyKeyboardMarkup:
    buttons = [[KeyboardButton(text=BTN_SEND_CODE)]]
    if is_admin:
        buttons.append([KeyboardButton(text=BTN_ADMIN_PANEL)])
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


def admin_menu_kb() -> ReplyKeyboardMarkup:
    buttons = [
        [KeyboardButton(text=BTN_SUBS)],
        [KeyboardButton(text=BTN_ADD_MOVIE), KeyboardButton(text=BTN_DEL_MOVIE)],
        [KeyboardButton(text=BTN_LIST_MOVIES), KeyboardButton(text=BTN_STATS)],
        [KeyboardButton(text=BTN_ADD_ADMIN), KeyboardButton(text=BTN_BACK)],
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


def build_subscribe_keyboard(channels: list[sqlite3.Row]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for channel in channels:
        join_link = channel["join_link"]
        ref = channel["channel_ref"]
        title = channel["title"] or ref
        if join_link:
            builder.row(InlineKeyboardButton(text=f"📌 Obuna: {title}", url=join_link))
        elif ref.startswith("@"):
            builder.row(
                InlineKeyboardButton(
                    text=f"📌 Obuna: {title}",
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


load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
DB_PATH = os.getenv("DB_PATH", "kino_bot.db").strip()
ADMIN_IDS = parse_admin_ids(os.getenv("ADMIN_IDS", ""))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN topilmadi. .env faylga BOT_TOKEN yozing.")

if not ADMIN_IDS:
    raise RuntimeError("ADMIN_IDS bo'sh. Masalan: ADMIN_IDS=123456789")

db = Database(DB_PATH)
db.seed_admins(ADMIN_IDS)

router = Router()


def guard_admin(message: Message) -> bool:
    return bool(message.from_user and db.is_admin(message.from_user.id))


async def ensure_subscription(user_id: int, bot: Bot) -> tuple[bool, list[sqlite3.Row]]:
    channels = db.get_required_channels()
    if not channels:
        return True, []

    not_joined: list[sqlite3.Row] = []
    for channel in channels:
        try:
            member = await bot.get_chat_member(chat_id=channel["channel_ref"], user_id=user_id)
            if not is_member_status(member.status):
                not_joined.append(channel)
        except (TelegramBadRequest, TelegramForbiddenError):
            not_joined.append(channel)
    return len(not_joined) == 0, not_joined


async def ask_for_subscription(message: Message, channels: list[sqlite3.Row]) -> None:
    text_lines = ["📢 Majburiy obuna mavjud. Quyidagi kanallarga obuna bo'ling:"]
    for ch in channels:
        title = ch["title"] or ch["channel_ref"]
        text_lines.append(f"- {title}")
    await message.answer("\n".join(text_lines), reply_markup=build_subscribe_keyboard(channels))


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not message.from_user:
        return

    db.add_user(message.from_user.id, message.from_user.full_name)
    admin = db.is_admin(message.from_user.id)
    text = (
        "🎬 Kino botga xush kelibsiz.\n"
        "Kod yuboring, bot sizga kinoni chiqarib beradi."
    )
    await message.answer(text, reply_markup=main_menu_kb(admin))


@router.message(F.text.in_({BTN_SEND_CODE, "Kino kodini yuborish"}))
async def ask_code(message: Message) -> None:
    if not message.from_user:
        return
    ok, channels = await ensure_subscription(message.from_user.id, message.bot)
    if not ok:
        await ask_for_subscription(message, channels)
        return
    await message.answer("🎟 Kino kodini yuboring.")


@router.callback_query(F.data == "check_sub")
async def check_subscription(callback: CallbackQuery) -> None:
    user = callback.from_user
    if not user:
        return
    ok, channels = await ensure_subscription(user.id, callback.bot)
    if ok:
        await callback.message.answer("✅ Obuna tasdiqlandi. Endi kino kodini yuboring.")
        await callback.answer("✅ Tasdiqlandi")
    else:
        await callback.message.answer(
            "❗ Hali ham obuna to'liq emas.",
            reply_markup=build_subscribe_keyboard(channels),
        )
        await callback.answer("❗ Obuna to'liq emas")


@router.message(F.text.in_({BTN_ADMIN_PANEL, "Admin panel"}))
async def open_admin_panel(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.clear()
    await message.answer("🛠 Admin panel", reply_markup=admin_menu_kb())


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
    await message.answer("📢 Majburiy obuna boshqaruvi:", reply_markup=sub_manage_kb())


@router.callback_query(F.data == "sub_add")
async def add_sub_start(callback: CallbackQuery, state: FSMContext) -> None:
    if not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    prompt = (
        "Kanalni quyidagi formatda yuboring:\n"
        "1) @kanal_username\n"
        "2) -1001234567890|https://t.me/+invite_link\n\n"
        "Yoki kanaldan bitta postni forward qiling.\n"
        f"Bekor qilish uchun: {BTN_CANCEL}"
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
            "Noto'g'ri format.\n"
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
            "Kanal topilmadi yoki bot kanalga kira olmayapti.\n"
            "Kanalni tekshiring va botni kanalga admin qiling."
        )
        return

    try:
        me = await message.bot.get_me()
        me_member = await message.bot.get_chat_member(chat_id=channel_ref, user_id=me.id)
        if me_member.status not in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR}:
            await message.answer("Bot majburiy obuna tekshirishi uchun kanalda admin bo'lishi shart.")
            return
    except (TelegramBadRequest, TelegramForbiddenError):
        await message.answer(
            "Bot bu kanalda a'zolar obunasini tekshira olmayapti.\n"
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
            "Private kanal uchun join link kerak.\n"
            "Format: -1001234567890|https://t.me/+invite_link"
        )
        return

    created = db.add_required_channel(channel_ref=channel_ref, title=title, join_link=join_link)
    await state.clear()
    if created:
        await message.answer(
            f"Kanal qo'shildi: {title}",
            reply_markup=admin_menu_kb(),
        )
    else:
        await message.answer(
            "Bu kanal allaqachon ro'yxatda.",
            reply_markup=admin_menu_kb(),
        )


@router.callback_query(F.data == "sub_list")
async def list_subscriptions(callback: CallbackQuery) -> None:
    if not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    channels = db.get_required_channels()
    if not channels:
        await callback.message.answer("Majburiy obuna kanallari bo'sh.")
        await callback.answer()
        return

    lines = ["Majburiy obuna kanallari:"]
    for ch in channels:
        title = ch["title"] or ch["channel_ref"]
        lines.append(f"{ch['id']}. {title} ({ch['channel_ref']})")
    await callback.message.answer("\n".join(lines))
    await callback.answer()


@router.callback_query(F.data == "sub_delete_menu")
async def delete_subscriptions_menu(callback: CallbackQuery) -> None:
    if not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    channels = db.get_required_channels()
    if not channels:
        await callback.message.answer("O'chirish uchun kanal topilmadi.")
        await callback.answer()
        return

    builder = InlineKeyboardBuilder()
    for ch in channels:
        title = ch["title"] or ch["channel_ref"]
        builder.button(text=f"❌ {title}", callback_data=f"sub_del:{ch['id']}")
    builder.adjust(1)
    await callback.message.answer("O'chiriladigan kanalni tanlang:", reply_markup=builder.as_markup())
    await callback.answer()


@router.callback_query(F.data.startswith("sub_del:"))
async def delete_channel(callback: CallbackQuery) -> None:
    if not db.is_admin(callback.from_user.id):
        await callback.answer()
        return
    _, raw_id = callback.data.split(":", 1)
    try:
        channel_id = int(raw_id)
    except ValueError:
        await callback.answer("Noto'g'ri ID")
        return

    deleted = db.remove_required_channel(channel_id)
    if deleted:
        await callback.message.answer("Kanal o'chirildi.")
    else:
        await callback.message.answer("Kanal topilmadi.")
    await callback.answer()


@router.message(F.text.in_({BTN_ADD_MOVIE, "Kino qo'shish"}))
async def add_movie_start(message: Message, state: FSMContext) -> None:
    if not message.from_user or not guard_admin(message):
        return
    await state.set_state(AddMovieState.waiting_code)
    await message.answer("➕ Yangi kino kodi yuboring:", reply_markup=cancel_kb())


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
    await state.update_data(code=text)
    await state.set_state(AddMovieState.waiting_title)
    await message.answer("Kino nomini yuboring:")


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
        "Caption yuboring (foydalanuvchiga video ostida chiqadi).\n"
        "Caption kerak bo'lmasa: -"
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
        "Endi video/document/photo yuboring.\n"
        "Yoki file_id/link matn yuborishingiz ham mumkin.\n"
        "Telegram post link yuborsangiz, kanal captioni olinmaydi.\n"
        "Siz yozgan caption media ostida chiqadi."
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

    media_type = ""
    file_id = ""

    if message.content_type == ContentType.VIDEO and message.video:
        media_type = "video"
        file_id = message.video.file_id
    elif message.content_type == ContentType.DOCUMENT and message.document:
        media_type = "document"
        file_id = message.document.file_id
    elif message.content_type == ContentType.PHOTO and message.photo:
        media_type = "photo"
        file_id = message.photo[-1].file_id
    elif message.content_type == ContentType.ANIMATION and message.animation:
        media_type = "animation"
        file_id = message.animation.file_id
    elif message.text:
        text = message.text.strip()
        if text.startswith("http://") or text.startswith("https://"):
            post_data = parse_telegram_post_link(text)
            if post_data:
                media_type = "telegram_post"
                file_id = pack_post_ref(post_data[0], post_data[1])
            else:
                media_type = "link"
                file_id = text
        else:
            media_type = "file_id"
            file_id = text
    else:
        await message.answer("Noto'g'ri format. Video/document/photo yoki matn yuboring.")
        return

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
        await message.answer("✅ Kino muvaffaqiyatli saqlandi.", reply_markup=admin_menu_kb())
    else:
        await message.answer("⚠️ Bu kod allaqachon mavjud.", reply_markup=admin_menu_kb())


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
        await message.answer("⚠️ Bu kod bo'yicha kino topilmadi.", reply_markup=admin_menu_kb())


@router.message(F.text.in_({BTN_LIST_MOVIES, "Kino ro'yxati"}))
async def movie_list(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    movies = db.list_movies()
    if not movies:
        await message.answer("Kino bazasi bo'sh.")
        return
    lines = ["Oxirgi kinolar:"]
    for item in movies:
        lines.append(f"{item['code']} - {item['title']}")
    await message.answer("\n".join(lines))


@router.message(F.text.in_({BTN_STATS, "Statistika"}))
async def stats(message: Message) -> None:
    if not message.from_user or not guard_admin(message):
        return
    s = db.stats()
    text = (
        "Statistika:\n"
        f"Foydalanuvchilar: {s['users']}\n"
        f"Kinolar: {s['movies']}\n"
        f"Majburiy obuna kanallari: {s['channels']}\n"
        f"So'rovlar: {s['requests']}"
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
        BTN_DEL_MOVIE.lower(),
        BTN_LIST_MOVIES.lower(),
        BTN_STATS.lower(),
        BTN_ADD_ADMIN.lower(),
        BTN_BACK.lower(),
        BTN_SEND_CODE.lower(),
        BTN_CANCEL.lower(),
        "admin panel",
        "majburiy obuna",
        "kino qo'shish",
        "kino o'chirish",
        "kino ro'yxati",
        "statistika",
        "admin qo'shish",
        "ortga",
        "kino kodini yuborish",
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
    if not movie:
        db.log_request(message.from_user.id, code, "not_found")
        await message.answer("Bunday kod topilmadi.")
        return

    caption = build_movie_caption(movie["title"], movie["description"])
    media_caption = caption if caption else None
    media_type = movie["media_type"]
    file_id = movie["file_id"]
    try:
        if media_type == "video":
            await message.answer_video(file_id, caption=media_caption)
        elif media_type == "document":
            await message.answer_document(file_id, caption=media_caption)
        elif media_type == "photo":
            await message.answer_photo(file_id, caption=media_caption)
        elif media_type == "animation":
            await message.answer_animation(file_id, caption=media_caption)
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
                caption=media_caption,
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
                    caption=media_caption,
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
        db.log_request(message.from_user.id, code, "success")
    except (TelegramBadRequest, TelegramForbiddenError, ValueError):
        db.log_request(message.from_user.id, code, "send_error")
        await message.answer(
            "Kino yuborishda xatolik. Admin media faylni qayta yuklasin."
        )


async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
