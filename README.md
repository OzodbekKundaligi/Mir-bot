# Kino Bot (Telegram)

Telegram kino/serial bot. Render `Web Service` sifatida ishlaydi: bitta process health server beradi, bitta process bot polling yuritadi.

## Asosiy funksiyalar

- Kod orqali kino yoki serial topish
- Nom bo'yicha qidirish
- Ko'rishlar sonini chiqarish
- Like / dislike va reyting
- `🏆 Top ko'rilganlar`
- Sevimlilar
- Topilmagan kontent uchun so'rov qoldirish
- So'rovga mos kontent qo'shilsa avto xabar yuborish
- Inline mode: `@botusername qidiruv`
- Media saqlab olish ochiq (`protect_content=False`)

## PRO funksiyalar

Loyihada faqat **bitta PRO tarif** bor.

- Admin `💰 Pro narxi` orqali yagona tarif narxini o'zgartiradi
- Admin `⏳ Pro muddati` orqali yagona tarif muddatini o'zgartiradi
- User `👑 Pro olish` orqali donat havolasiga o'tadi
- User izohga `PRO-TELEGRAM_ID` yoki Telegram ID yozadi
- `✅ To‘lov qildim` orqali payment request yuboradi
- Admin `💳 Pro so‘rovlar` bo‘limida tasdiqlaydi yoki rad etadi
- Admin `👑 Pro boshqarish` orqali PRO ni qo‘lda yoqadi yoki o‘chiradi
- Adminlar uchun PRO avtomatik **cheksiz aktiv**
- PRO user uchun e'lon berish ochiladi
- PRO user majburiy obunadan bypass qilinadi
- PRO userlarga yangi kontent notification yuboriladi

## E'lon tizimi

Faqat PRO user ishlata oladi.

- `📢 E'lon berish`
- Rasm yoki `/skip`
- Sarlavha
- Tavsif
- Inline tugma kerakmi — inline `Ha / Yo'q`
- Tugma matni
- Tugma havolasi
- Preview
- Moderator tasdig'i
- Admin kanal tanlaydi va post qiladi

Admin bo'limlari:

- `📰 E'lonlar`
- `📡 E'lon kanalari`
- Kanal qo'shish / ro'yxat / o'chirish

## Broadcast

Admin `📣 Habar yuborish` orqali quyidagilarni jo'nata oladi:

- matn
- rasm
- video
- gif / animation
- document
- audio
- voice
- emoji bilan xabar

Flow:

- xabar yuboriladi
- `Inline tugma kerakmi?` savoli chiqadi
- inline `Ha / Yo'q`
- `Ha` bo'lsa tugma matni va havola so'raladi
- preview ko'rsatiladi
- tasdiqlangach barcha userlarga yuboriladi

## Notification sozlamalari

User `🔔 Bildirishnomalar` bo'limida yoqib/o'chira oladi:

- yangi kontent
- PRO xabarlari
- e'lon holati

## O'rnatish

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` ichida:

- `BOT_TOKEN`
- `ADMIN_IDS`
- `MONGODB_URI`
- `MONGODB_DB`
- `PRO_PRICE_TEXT`
- `PRO_DURATION_DAYS`
- `PRO_PAYMENT_LINK_1`
- `PRO_PAYMENT_LINK_2`

## Ishga tushirish

Lokal:

```powershell
python main.py
```

Render `Web Service`:

- start command: `python run_all.py`
- health endpoints:
  - `/`
  - `/health`

Muhim:

- `PORT` Render tomonidan beriladi
- bot va health server birga ishga tushadi

## Docker

Tayyor fayllar:

- `Dockerfile`
- `docker-compose.yml`
- `Procfile`
- `Procfile.bot`

Lokal docker:

```powershell
docker compose up --build
```

## Admin panel

- `📢 Majburiy obuna`
- `➕ Kino qo'shish`
- `📺 Serial qo'shish`
- `🗑 Kino o'chirish`
- `📝 Kino tahrirlash`
- `📚 Kino ro'yxati`
- `📣 Habar yuborish`
- `📥 So'rovlar`
- `📊 Statistika`
- `👑 Pro boshqarish`
- `💰 Pro narxi`
- `⏳ Pro muddati`
- `💳 Pro so'rovlar`
- `📰 E'lonlar`
- `📡 E'lon kanalari`
- `🎲 Random kod`
- `👤 Admin qo'shish`

## Majburiy obuna kanal formatlari

1. Public kanal:
   - `@kanal_username`
2. ID + taklif link:
   - `-1001234567890|https://t.me/+invite_link`
