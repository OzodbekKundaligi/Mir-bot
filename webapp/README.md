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

## 5) Railway deploy

Bitta repodan 3 ta service ochish tavsiya:

1. `kino-bot-worker`
   - Dockerfile: `Dockerfile.bot`
   - Start: `python main.py`
2. `kino-web-api`
   - Dockerfile: `webapp/server/Dockerfile`
   - Public URL oladi (masalan: `https://kino-web-api.up.railway.app`)
3. `kino-web-client`
   - Dockerfile: `webapp/client/Dockerfile`
   - Build arg: `VITE_API_BASE=https://kino-web-api.up.railway.app`

Keyin bot `.env` ichida:

- `WEBAPP_URL=https://kino-web-client.up.railway.app`
