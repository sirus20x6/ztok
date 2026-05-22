# Build the ztok C example via pkg-config. Tells the linker where to
# find libztok.so + ztok.h from the installed ztok.pc.
#
# Run as:
#   PKG_CONFIG_PATH=<install>/lib/pkgconfig make
#
# Override CC / CFLAGS as you would for any C project.

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?=

PKG_CFLAGS := $(shell pkg-config --cflags ztok)
PKG_LIBS   := $(shell pkg-config --libs ztok)

# Pin RPATH so the resulting binary finds libztok.so at runtime without
# `LD_LIBRARY_PATH=` plumbing.
PKG_LIBDIR := $(shell pkg-config --variable=libdir ztok)
RPATH      := -Wl,-rpath,$(PKG_LIBDIR)

all: hello overlays

hello: hello.c
	$(CC) $(CFLAGS) $(PKG_CFLAGS) -o $@ $< $(PKG_LIBS) $(RPATH) $(LDFLAGS)

overlays: overlays.c
	$(CC) $(CFLAGS) $(PKG_CFLAGS) -o $@ $< $(PKG_LIBS) $(RPATH) $(LDFLAGS)

clean:
	rm -f hello overlays

.PHONY: all clean
