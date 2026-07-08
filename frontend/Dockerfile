# Frontend (prod): build the React app with Vite, then serve the static output with
# nginx. Multi-stage so the final image is just nginx + the compiled assets — no
# node/npm in the runtime image.

FROM node:20-alpine AS build
WORKDIR /app
# Install deps from the lockfile first (cached unless package*.json changes).
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build          # -> /app/dist

FROM nginx:1.27-alpine
# SPA config: static assets + /api proxy to the backend. Templated (see
# nginx.conf.template) — nginx:alpine's entrypoint envsubst's files under
# /etc/nginx/templates/*.template into /etc/nginx/conf.d/ at container start, using
# whatever PORT/BACKEND_URL/DNS_RESOLVER are in the environment. Defaults below match
# local docker-compose; Cloud Run overrides them (see scripts/gcloud_deploy_app.sh).
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
ENV PORT=80 \
    BACKEND_URL=http://backend:5000 \
    DNS_RESOLVER=127.0.0.11
EXPOSE 80
# nginx:alpine's base CMD already runs nginx in the foreground.
