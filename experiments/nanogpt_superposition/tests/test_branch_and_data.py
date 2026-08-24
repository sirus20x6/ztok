from __future__ import annotations

import hashlib

import torch
from branch_equivalence import branch_equivalence
from dataset import DeterministicBatchPrefetcher, PackedCorpus
from models.common import ModelConfig
from models.rwkv import RWKVLM
from prepare_data import SplitWriter, _iter_documents, _write_split


def test_branch_equivalence_is_exact_for_identical_candidates() -> None:
    config = ModelConfig(
        vocab_size=16, width=16, layers=1, heads=1, max_sequence_length=16
    )
    model = RWKVLM(config).eval()
    with torch.no_grad():
        model.token_embedding.weight[2].copy_(model.token_embedding.weight[1])
    result = branch_equivalence(
        model,
        torch.tensor([4, 5, 6]),
        torch.tensor([1, 2]),
        torch.tensor([0.7, 0.3]),
        torch.tensor([7, 8, 9, 10]),
        horizons=(0, 1, 4),
    )
    assert max(value.state_error for value in result.horizons) < 1e-5
    assert max(value.logit_kl for value in result.horizons) < 1e-6


def test_prepared_batches_never_cross_documents(tmp_path) -> None:
    rows = []
    for offset in (0, 20, 40):
        ids = list(range(offset, offset + 12))
        starts = list(range(12))
        ends = list(range(1, 13))
        digest = hashlib.sha256(bytes(ids)).hexdigest()
        rows.append((ids, starts, ends, 12, digest))
    _write_split(tmp_path, "train", rows)
    _write_split(tmp_path, "validation", rows[:1])
    (tmp_path / "metadata.json").write_text(
        '{"schema":"ztok.nanogpt_superposition.dataset.v1","tokenizer":{"vocabulary_size":64}}'
    )
    corpus = PackedCorpus(tmp_path, "train")
    batch = corpus.sample(
        8,
        6,
        generator=torch.Generator().manual_seed(9),
        device=torch.device("cpu"),
    )
    for row in batch.token_ids:
        assert int(row.max() - row.min()) == 5
    assert batch.source_bytes == 8 * 6


def test_prefetch_preserves_sampling_order_and_generator_state(tmp_path) -> None:
    rows = []
    for offset in (0, 20, 40):
        ids = list(range(offset, offset + 12))
        rows.append(
            (
                ids,
                list(range(12)),
                list(range(1, 13)),
                12,
                hashlib.sha256(bytes(ids)).hexdigest(),
            )
        )
    _write_split(tmp_path, "train", rows)
    (tmp_path / "metadata.json").write_text(
        '{"schema":"ztok.nanogpt_superposition.dataset.v1",'
        '"tokenizer":{"vocabulary_size":64}}'
    )
    corpus = PackedCorpus(tmp_path, "train")
    synchronous_generator = torch.Generator().manual_seed(77)
    prefetched_generator = torch.Generator().manual_seed(77)
    prefetcher = DeterministicBatchPrefetcher(
        corpus,
        batch_size=4,
        sequence_tokens=6,
        generator=prefetched_generator,
        pin_memory=False,
    )
    try:
        for _ in range(5):
            expected_random = float(
                torch.rand((), generator=synchronous_generator)
            )
            expected = corpus.sample(
                4,
                6,
                generator=synchronous_generator,
                device=torch.device("cpu"),
            )
            actual_random, actual = prefetcher.next(torch.device("cpu"))
            assert actual_random == expected_random
            assert torch.equal(actual.token_ids, expected.token_ids)
            assert torch.equal(actual.document_ids, expected.document_ids)
            assert actual.source_bytes == expected.source_bytes
        assert torch.equal(
            prefetched_generator.get_state(), synchronous_generator.get_state()
        )
    finally:
        prefetcher.close()


def test_jsonl_documents_and_streaming_writer(tmp_path) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text('{"text":"first document"}\n{"text":"second document"}\n')
    documents = list(
        _iter_documents([source], line_documents=False, jsonl_text_field="text")
    )
    assert [document.text for document in documents] == [
        b"first document",
        b"second document",
    ]
    writer = SplitWriter(tmp_path, "stream")
    for source_document in documents:
        document = source_document.text
        digest = hashlib.sha256(document).hexdigest()
        ids = list(document)
        writer.append(
            ids,
            list(range(len(ids))),
            list(range(1, len(ids) + 1)),
            len(document),
            digest,
        )
    metadata = writer.finish()
    assert metadata["documents"] == 2
    assert metadata["tokens"] == sum(len(document.text) for document in documents)
