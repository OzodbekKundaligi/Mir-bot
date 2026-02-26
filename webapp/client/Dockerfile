FROM node:20-alpine AS build

WORKDIR /app

COPY webapp/client/package.json /app/package.json
COPY webapp/client/package-lock.json /app/package-lock.json
RUN npm ci

COPY webapp/client /app
ARG VITE_API_BASE
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM node:20-alpine

WORKDIR /app
RUN npm install -g serve

COPY --from=build /app/dist /app/dist

CMD ["sh", "-c", "serve -s dist -l ${PORT:-4173}"]
