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

# nginx-unprivileged: the same nginx, but runs as a NON-ROOT user (uid 101) out of the
# box (security hardening, §3.7) — no master process as root. It listens on 8080 by
# default (a non-root user can't bind :80) and ships the identical envsubst-on-templates
# entrypoint, so the templating below is unchanged.
FROM nginxinc/nginx-unprivileged:1.27-alpine
# SPA config: static assets + /api proxy to the backend. Templated (see
# nginx.conf.template) — the entrypoint envsubst's files under /etc/nginx/templates/
# *.template into /etc/nginx/conf.d/ at container start, using whatever
# PORT/BACKEND_URL/DNS_RESOLVER are in the environment. Defaults below match local
# docker-compose; Cloud Run overrides them (see scripts/gcloud_deploy_app.sh).
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
ENV PORT=8080 \
    BACKEND_URL=http://backend:5000 \
    DNS_RESOLVER=127.0.0.11
EXPOSE 8080
# The base image's CMD already runs nginx in the foreground as the non-root user.
