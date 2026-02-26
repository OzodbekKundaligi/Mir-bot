FROM node:20-alpine AS web-build

WORKDIR /web

COPY webapp/client/package.json /web/package.json
COPY webapp/client/package-lock.json /web/package-lock.json
RUN npm ci

COPY webapp/client /web
ARG VITE_API_BASE=
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
COPY --from=web-build /web/dist /app/webapp/client/dist

CMD ["python", "run_all.py"]
