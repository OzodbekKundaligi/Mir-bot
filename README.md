# Kino Bot (Telegram)

Bu bot kino kod orqali kino beradigan Telegram bot.

## Asosiy funksiyalar

- `/start` orqali ishga tushadi.
- Admin qo'ygan kanallarga **majburiy obuna** tekshiradi.
- Obuna tasdiqlansa foydalanuvchi kino kod yuboradi.
- Kod topilsa kino yuboriladi.
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

## Ishga tushirish

```powershell
python main.py
```

## Admin paneldan nimalar qilinadi

- `Majburiy obuna`:
  - Kanal qo'shish
  - Kanallar ro'yxati
  - Kanal o'chirish
- `Kino qo'shish`
- `Kino o'chirish`
- `Kino ro'yxati`
- `Statistika`
- `Admin qo'shish`

## Majburiy obuna kanal formatlari

1. Public kanal:
   - `@kanal_username`
2. ID + taklif link:
   - `-1001234567890|https://t.me/+invite_link`
