FROM node:20-alpine AS miniapp-build

WORKDIR /miniapp
COPY miniapp/package*.json ./
RUN npm ci
COPY miniapp ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
COPY --from=miniapp-build /miniapp/dist /app/miniapp/dist

CMD ["python", "run_all.py"]
