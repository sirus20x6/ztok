# syntax=docker/dockerfile:1.7
#
# Multi-stage Dockerfile for `ztok serve`.
#
# Stage 1 (builder): Ubuntu 24.04 + upstream Zig 0.16.0 tarball. We
# build the static `ztok` binary AND the shared `libztok.so` so the
# runtime image has both available; some workflows want the standalone
# CLI (serve / validate), others want to LD_PRELOAD the .so against a
# host C/Python binding.
#
# Stage 2 (runtime): distroless/cc-debian12 — gives us glibc + libgcc_s
# + libstdc++ but no shell, no package manager, no root tools. ztok
# uses libc for the C ABI allocator (see build.zig) and pthread for
# BatchPool, both of which distroless/cc provides. Final image is
# small (~80 MiB) and has a strictly smaller attack surface than a
# full Ubuntu/Alpine.
#
# Build:
#   docker build -t ztok:1.20 .
#
# Run (HTTP server bound to all interfaces on 7890, model from host):
#   docker run --rm -p 7890:7890 -v $PWD/models:/models ztok:1.20 \
#       serve --model /models/cl100k_base.tiktoken --cl100k \
#             --host 0.0.0.0 --port 7890
#
# The default CMD prints `serve --help` so a bare `docker run ztok:1.20`
# documents itself.

FROM ubuntu:24.04 AS builder

# `--mount=type=cache` lets repeated builds skip the apt + Zig download
# when buildkit cache is warm; harmless under non-buildkit builders
# (they ignore the mount). xz-utils is needed to unpack the Zig tarball;
# ca-certificates so wget can verify ziglang.org over HTTPS.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget xz-utils ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pin Zig version. Bump when the project's minimum_zig_version bumps.
# Note: 0.16+ download layout is `zig-<arch>-<os>-<ver>.tar.xz` (the
# `linux` slug comes AFTER `<arch>`). Older releases used the inverse
# order — bump-and-forget would break the build, hence the explicit
# variable shape here.
ARG ZIG_VERSION=0.16.0
ARG ZIG_ARCH=x86_64
ARG ZIG_OS=linux

RUN wget -qO /tmp/zig.tar.xz \
        "https://ziglang.org/download/${ZIG_VERSION}/zig-${ZIG_ARCH}-${ZIG_OS}-${ZIG_VERSION}.tar.xz" && \
    mkdir -p /opt && \
    tar -xJf /tmp/zig.tar.xz -C /opt && \
    mv "/opt/zig-${ZIG_ARCH}-${ZIG_OS}-${ZIG_VERSION}" /opt/zig && \
    rm /tmp/zig.tar.xz

ENV PATH=/opt/zig:$PATH

WORKDIR /src
COPY . .

# ReleaseFast for the serving binary; ReleaseSafe is what CI runs the
# full test suite under (see .github/workflows/ci.yml).
RUN zig build -Doptimize=ReleaseFast

# --- runtime ---
FROM gcr.io/distroless/cc-debian12

# Copy only the bits a production server needs:
#   * /usr/local/bin/ztok           — the CLI driver
#   * /usr/local/lib/libztok.so     — for processes that dlopen the lib
#   * /usr/local/include/ztok.h     — C header (handy when this image
#                                     is also used as a build stage)
COPY --from=builder /src/zig-out/bin/ztok        /usr/local/bin/ztok
COPY --from=builder /src/zig-out/lib/libztok.so  /usr/local/lib/libztok.so
COPY --from=builder /src/zig-out/include/ztok.h  /usr/local/include/ztok.h

# Make sure dlopen(libztok.so) works without setting LD_LIBRARY_PATH
# at runtime. distroless's glibc honours /etc/ld.so.conf.d-style hints
# only if ldconfig has run; since we lack ldconfig in distroless we
# export LD_LIBRARY_PATH instead.
ENV LD_LIBRARY_PATH=/usr/local/lib

# Default HTTP serve port — overridable via `--port N`.
EXPOSE 7890

# Default ENTRYPOINT is the CLI; default CMD prints the top-level
# usage (which lists every subcommand including `serve`) so a bare
# `docker run ztok:1.20` is self-documenting. Override CMD with
# `serve --model ...` for real workloads (see header comment).
ENTRYPOINT ["/usr/local/bin/ztok"]
CMD ["help"]
