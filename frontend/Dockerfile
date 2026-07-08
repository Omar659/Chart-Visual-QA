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
# SPA config: static assets + /api proxy to the backend (see nginx.conf).
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
# nginx:alpine's base CMD already runs nginx in the foreground.
