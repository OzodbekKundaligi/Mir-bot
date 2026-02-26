# Kino Web App (Telegram)

Bu papkada Telegram Web App uchun alohida frontend (`React`) va backend (`FastAPI`) bor.

## Tuzilma

- `client/` - React (Vite) ilova.
- `server/` - FastAPI API.

## Asosiy imkoniyatlar

- Majburiy obuna tekshiruvi.
- Feed + qidiruv (kod/nom/ta'rif).
- Like / dislike.
- Comment tizimi.
- Saqlanganlar (favorites).
- Ko'rish tarixi.
- Download tracking.
- Trend kontentlar.
- O'xshash kontent tavsiyalari.
- Profil statistikasi.

## 1) Backend ishga tushirish

```powershell
cd webapp/server
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` ichida bular to'g'ri bo'lishi kerak:

- `BOT_TOKEN`
- `MONGODB_URI`
- `MONGODB_DB`
- `BOT_USERNAME` (ixtiyoriy, lekin tavsiya etiladi)
- `WEBAPP_ALLOWED_ORIGINS` (masalan: `https://your-web-app-domain`)

Ishga tushirish:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 2) Frontend ishga tushirish

```powershell
cd webapp/client
npm install
npm run dev
```

`VITE_API_BASE` ni sozlash:

```powershell
$env:VITE_API_BASE='http://localhost:8000'
npm run dev
```

## 3) Bot bilan ulash (Telegram)

Asosiy bot `.env` faylga qo'shing:

- `WEBAPP_URL=https://sizning-webapp-url`

Shundan keyin bot menyusida `Web ilova` tugmasi chiqadi.

## 4) Docker (lokal)

Root papkada:

```powershell
docker compose up --build
```

Servislar:

- Bot: `Dockerfile.bot`
- API: `webapp/server/Dockerfile`
- Client: `webapp/client/Dockerfile`

## 5) Railway deploy (bitta service)

Sizda bitta service bo'lsa, shu loyiha tayyor:

1. Start command: `python run_all.py`
2. Bu bir paytda ikkalasini ishga tushiradi:
   - Bot polling (`main.py`)
   - Web API (`webapp/server/app.py`) `PORT` da
3. Frontend build fayllari `webapp/client/dist` ichidan FastAPI orqali beriladi.

Shart bo'lgan ENV:

- `BOT_TOKEN`
- `MONGODB_URI`
- `MONGODB_DB`
- `ADMIN_IDS`
- `WEBAPP_URL=https://mir-bot-production.up.railway.app/`

Konsolda tekshirish:

- `[boot] WEB API listen: 0.0.0.0:<PORT>`
- `[boot] Public URL: https://...`
- `[boot] webapi pid=...`
- `[boot] bot pid=...`

Ishlayotganini tekshirish:

- `https://mir-bot-production.up.railway.app/health` -> `{"ok": true}`
- `https://mir-bot-production.up.railway.app/` -> Web App sahifasi
