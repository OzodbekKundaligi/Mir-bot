# Kino Bot (Telegram)

Bu bot kino yoki serial kod orqali media beradigan Telegram bot.

## Asosiy funksiyalar

- `/start` orqali ishga tushadi.
- Admin qo'ygan kanallarga **majburiy obuna** tekshiradi.
- Obuna tasdiqlansa foydalanuvchi kodni chatga yozadi.
- Kod topilsa kino yoki serial qismlari yuboriladi.
- Topilmagan kod/nom uchun user **so'rov qoldira oladi**.
- Admin kontent qo'shganda so'rov qoldirgan userlarga **avto xabar yuboradi**.
- **Nom bo'yicha qidiruv**, **janr/yil/sifat filter**, **sevimlilar** qo'shilgan.
- **Inline mode**: `@botusername qidiruv`.
- Inline'da kod bo'yicha qidiruv, typo-ga yaqin natijalar, yuklashlar soni va kontentning o'zidan olingan preview chiqadi.
- Admin boshqaruvi `/admin` orqali emas, **`Admin panel`** tugmasi orqali.
- Bir nechta admin (`ADMIN_IDS`) qo'llab-quvvatlanadi.

## O'rnatish

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
New-Item .env -ItemType File
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

## Telegram Web App

Web App alohida papkada:

- `webapp/client` - React frontend
- `webapp/server` - FastAPI backend

Batafsil yo'riqnoma: `webapp/README.md`

Web App funksiyalari:

- Like / dislike
- Comment
- Saqlanganlar
- Yuklab olish tracking
- Tarix
- Trendlar
- O'xshash kontent tavsiyasi
- Profil statistikasi

## Docker / Procfile

Deploy uchun tayyor fayllar:

- `Procfile`
- `Procfile.bot`
- `Procfile.web`
- `Dockerfile` (bot)
- `Dockerfile.bot` (bot)
- `webapp/server/Dockerfile` (API)
- `webapp/client/Dockerfile` (frontend)
- `docker-compose.yml` (lokal full stack)

Lokal ishga tushirish:

```powershell
docker compose up --build
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
