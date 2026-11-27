# syntax=docker/dockerfile:1
ARG TARGETARCH
FROM --platform=$BUILDPLATFORM node:26-bookworm-slim AS ui
WORKDIR /work
COPY web/package*.json ./
COPY web/patches ./patches
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build && mv dist/BASE_PATH/monacoeditorwork/* dist/assets/ && rm -r dist/BASE_PATH

# Standard runtime from the same source revision, for each supported CPU architecture.
FROM ghcr.io/blakeblackshear/frigate@sha256:a1df8c3df6c7e051c60ccbf03c76755a8b5dde7a8b77aaa3a95f585ab561e37c AS runtime-amd64
FROM ghcr.io/blakeblackshear/frigate@sha256:5bce2f481c990ad8f9400eb2784c6c089d89f0e62b1a7ad403fc18bf39157695 AS runtime-arm64
FROM runtime-${TARGETARCH}
WORKDIR /opt/frigate
RUN rm -rf web frigate migrations
COPY frigate ./frigate
COPY migrations ./migrations
COPY --from=ui /work/dist ./web
COPY docker/main/rootfs/usr/local/nginx/conf/nginx.conf /usr/local/nginx/conf/nginx.conf
COPY demo /opt/drishti/demo
COPY validation /opt/drishti/validation
COPY validation/check_replay.py /opt/drishti/check_replay.py
COPY deploy/drishti/config.yml deploy/drishti/start.py /opt/drishti/
ENTRYPOINT ["python3","/opt/drishti/start.py"]
