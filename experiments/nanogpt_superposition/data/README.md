# Prepared data

`prepare_data.py` writes deterministic `uint32` token arrays, aligned source-byte
counts, document ranges, and a versioned metadata file here. Generated corpora are
ignored by Git. Tiny Shakespeare is a plumbing corpus only; serious conclusions
must use a fixed, legally usable 100M–500M-token shard.
