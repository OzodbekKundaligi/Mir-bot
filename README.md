# Kino Bot (Telegram)

Bu bot kino yoki serial kod orqali media beradigan Telegram bot.

## Asosiy funksiyalar

- `/start` orqali ishga tushadi.
- Admin qo'ygan kanallarga **majburiy obuna** tekshiradi.
- Obuna tasdiqlansa foydalanuvchi kodni chatga yozadi.
- Kod topilsa kino yoki serial qismlari yuboriladi.
- Admin boshqaruvi `/admin` orqali emas, **`Admin panel`** tugmasi orqali.
- Bir nechta admin (`ADMIN_IDS`) qo'llab-quvvatlanadi.

## O'rnatish

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` ichida:

- `BOT_TOKEN` ni kiriting.
- `ADMIN_IDS` ga kamida bitta admin ID yozing (vergul bilan bir nechtasi ham mumkin).
- `MONGODB_URI` ni kiriting (masalan: `mongodb://localhost:27017`).
- `MONGODB_DB` nomini kiriting (masalan: `kino_bot`).

## Ishga tushirish

```powershell
python main.py
```

## SQLite -> MongoDB migratsiya

Agar eski `kino_bot.db` bo'lsa, MongoDB'ga ko'chirish:

```powershell
python migrate_sqlite_to_mongo.py --sqlite-path kino_bot.db --drop-existing
```

## Admin paneldan nimalar qilinadi

- `Majburiy obuna`:
  - Kanal qo'shish
  - Kanallar ro'yxati
  - Kanal o'chirish
- `Kino qo'shish`
- `Serial qo'shish`
- `Kino o'chirish`
- `Kino ro'yxati`
- `Statistika`
- `Admin qo'shish`

## Majburiy obuna kanal formatlari

1. Public kanal:
   - `@kanal_username`
2. ID + taklif link:
   - `-1001234567890|https://t.me/+invite_link`
