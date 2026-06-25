# Llama Guard on CPU via Ollama, with the model BAKED INTO the image so a cold
# start never re-downloads it (key for scale-to-zero). Build: docker compose build guard.
#
# NOTE: do NOT mount a volume over /root/.ollama for this image — an empty volume
# would shadow the baked-in model and force a re-pull. (See docker-compose.yml.)
FROM ollama/ollama:latest

# Pre-pull the guard model at build time: start the server, wait until it's up,
# pull, then let the build step end (the background server stops with the shell;
# the downloaded weights persist in the image layer under /root/.ollama).
RUN ollama serve & \
    until ollama list >/dev/null 2>&1; do sleep 1; done; \
    ollama pull llama-guard3:1b

# Base image already does ENTRYPOINT ["/bin/ollama"] CMD ["serve"] -> serves :11434.
