'use strict';

// Tiny Express-style demo: a ~30-line tokenize-as-a-service server that
// reuses one Pipeline + one BatchPool across all requests. Commented out
// so this example doesn't add an `express` dependency to the package; if
// you want to run it, `npm install express` first.
//
// Run:
//   ZTOK_LIB_PATH=../../zig-out/lib/libztok.so \
//     node bindings/nodejs/examples/server.js cl100k_base.tiktoken

/*
const express = require('express');
const ztok = require('..');

const modelPath = process.argv[2];
if (!modelPath) {
    console.error('usage: server.js <tokenizer-file>');
    process.exit(2);
}

const pipe = ztok.Pipeline.fromPath(modelPath);
const pool = new ztok.BatchPool({ workers: 8 });

const app = express();
app.use(express.json({ limit: '4mb' }));

app.post('/encode', (req, res) => {
    const { text } = req.body || {};
    if (typeof text !== 'string') return res.status(400).json({ error: 'text required' });
    res.json({ ids: Array.from(pipe.encode(text)) });
});

app.post('/encode_batch', (req, res) => {
    const { inputs } = req.body || {};
    if (!Array.isArray(inputs)) return res.status(400).json({ error: 'inputs required' });
    const results = pipe.encodeBatch(pool, inputs).map((u) => Array.from(u));
    res.json({ results });
});

app.post('/decode', (req, res) => {
    const { ids } = req.body || {};
    if (!Array.isArray(ids)) return res.status(400).json({ error: 'ids required' });
    res.json({ text: pipe.decode(ids) });
});

app.get('/version', (_req, res) => res.json({ version: ztok.version() }));

const port = Number(process.env.PORT || 7890);
app.listen(port, () => console.log(`ztok server on :${port}`));

process.on('SIGTERM', () => { pool.close(); pipe.close(); process.exit(0); });
*/

console.log('server.js is a commented template. See the source for the body.');
console.log('Uncomment after `npm install express`.');
