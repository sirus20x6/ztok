# Token superposition implementation contract

The stable experimental contracts requested by the original implementation brief
are implemented and documented in [superposition-schemas.md](superposition-schemas.md).

The NanoGPT experiment consumes ordinary IDs and offsets through the Python
binding, audits fixed partitions against `ztok.superposition.v1`, and performs all
embedding fusion and model loss computation outside tokenizer core. Ordinary
`Pipeline.encode`, streaming, batch encoding, file formats, and existing language
bindings remain unchanged.

Current schemas:

- `ztok.superposition.v1`: fixed source-token grouping and embedding operation plan.
- `ztok.semantic_spans.v1`: contextual semantic-span exchange.
- `ztok.ccss.v1`: conservative cross-caption grouping plan.
