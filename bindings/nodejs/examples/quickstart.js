'use strict';

// Self-contained 8-line quickstart: load a tokenizer, encode, decode.
//
// Run from the repo root after `zig build`:
//
//   ZTOK_LIB_PATH=zig-out/lib/libztok.so \
//     node bindings/nodejs/examples/quickstart.js [path/to/tokenizer]
//
// Without an argument we synthesize a tiny .tiktoken file in the system
// temp dir so the example always exits 0 with meaningful output. Pass a
// real .tiktoken / tokenizer.json / .model / .ztm path to use it
// directly — Pipeline.fromPath auto-detects the format via the C ABI.

const fs = require('fs');
const os = require('os');
const path = require('path');

const ztok = require('..');

function synthesizeTiktoken() {
    const extras = [
        'he', 'hel', 'hell', 'hello',
        ' w', ' wo', ' wor', ' worl', ' world',
    ];
    const p = path.join(os.tmpdir(), 'ztok-quickstart.tiktoken');
    const lines = [];
    let rank = 0;
    for (let b = 0; b < 256; b++) {
        lines.push(`${Buffer.from([b]).toString('base64')} ${rank++}`);
    }
    for (const e of extras) {
        lines.push(`${Buffer.from(e, 'utf-8').toString('base64')} ${rank++}`);
    }
    fs.writeFileSync(p, lines.join('\n') + '\n');
    return p;
}

const modelPath = process.argv[2] || synthesizeTiktoken();
const pipe = ztok.Pipeline.fromPath(modelPath);
try {
    const ids = pipe.encode('hello world');
    console.log(`ztok ${ztok.version()}: ${ids.length} ids -> [${Array.from(ids)}]`);
    console.log(`decoded: ${JSON.stringify(pipe.decode(ids))}`);
} finally {
    pipe.close();
}
