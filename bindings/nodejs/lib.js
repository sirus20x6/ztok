'use strict';

// libztok loader — mirrors bindings/python/ztok/_lib.py.
//
// Resolution order:
//   1. ZTOK_LIB_PATH env var (explicit override).
//   2. <pkg>/../../zig-out/lib/libztok.{so,dylib,dll} (in-tree dev path).
//   3. <pkg>/libztok.{so,dylib,dll} (package-relative for npm install).
//   4. Standard system paths: /usr/local/lib, /usr/lib, /usr/lib64,
//      /opt/homebrew/lib on macOS.
//
// Throws ZtokLibraryNotFoundError with the list of tried paths if nothing
// works. The C ABI's id buffers come back with a length-prefix header
// (see src/c_api.zig::allocIdBuf); the ONLY safe free is `ztok_ids_free`.

const fs = require('fs');
const path = require('path');
const os = require('os');

class ZtokLibraryNotFoundError extends Error {
    constructor(message) {
        super(message);
        this.name = 'ZtokLibraryNotFoundError';
    }
}

function sharedLibBasename() {
    if (process.platform === 'darwin') return 'libztok.dylib';
    if (process.platform === 'win32') return 'ztok.dll';
    return 'libztok.so';
}

function candidatePaths() {
    const base = sharedLibBasename();
    const pkgDir = __dirname;
    const paths = [];

    // 2. In-tree dev: bindings/nodejs/ -> repo root -> zig-out/lib
    paths.push(path.join(pkgDir, '..', '..', 'zig-out', 'lib', base));

    // 3. Package-relative (npm install layout)
    paths.push(path.join(pkgDir, base));
    paths.push(path.join(pkgDir, '..', base));

    // 4. Standard system paths
    if (process.platform === 'linux') {
        paths.push(path.join('/usr/local/lib', base));
        paths.push(path.join('/usr/lib', base));
        paths.push(path.join('/usr/lib64', base));
    } else if (process.platform === 'darwin') {
        paths.push(path.join('/usr/local/lib', base));
        paths.push(path.join('/opt/homebrew/lib', base));
    }
    return paths;
}

function loadLibztok(koffi) {
    // 1. Explicit env override
    const override = process.env.ZTOK_LIB_PATH;
    if (override) {
        if (!fs.existsSync(override)) {
            throw new ZtokLibraryNotFoundError(
                `ZTOK_LIB_PATH=${override} does not point to an existing file.`
            );
        }
        return koffi.load(override);
    }

    // 2-4. Standard paths
    const tried = [];
    for (const p of candidatePaths()) {
        tried.push(p);
        if (fs.existsSync(p)) {
            try {
                return koffi.load(p);
            } catch (e) {
                tried[tried.length - 1] = `${p} (dlopen failed: ${e.message})`;
            }
        }
    }

    // 5. Bare name fallback (lets the OS dynamic loader find it via
    // LD_LIBRARY_PATH / DYLD_FALLBACK_LIBRARY_PATH).
    const base = sharedLibBasename();
    tried.push(`bare name: ${base}`);
    try {
        return koffi.load(base);
    } catch (e) {
        tried[tried.length - 1] = `${base} (dlopen failed: ${e.message})`;
    }

    throw new ZtokLibraryNotFoundError(
        'Could not locate libztok. Build it with `zig build` and either:\n' +
        '  - install it system-wide (e.g. cp zig-out/lib/libztok.so /usr/local/lib/),\n' +
        '  - place it next to the ztok Node.js package, or\n' +
        '  - set ZTOK_LIB_PATH=/absolute/path/to/libztok.so.\n' +
        'Tried: ' + tried.join('; ')
    );
}

module.exports = {
    ZtokLibraryNotFoundError,
    loadLibztok,
};
