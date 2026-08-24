"""Memory-mapped document-bounded batches with exact source-byte accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


@dataclass
class TokenBatch:
    token_ids: torch.Tensor
    source_bytes: int
    document_ids: torch.Tensor

    def to(self, device: torch.device) -> "TokenBatch":
        return TokenBatch(
            token_ids=self.token_ids.to(device=device, non_blocking=True),
            source_bytes=self.source_bytes,
            document_ids=self.document_ids,
        )


class PackedCorpus:
    def __init__(self, root: Path | str, split: str) -> None:
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("schema") != "ztok.nanogpt_superposition.dataset.v1":
            raise ValueError("unsupported prepared dataset schema")
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.split = split
        self.tokens = np.memmap(
            self.root / f"{split}.tokens.bin", dtype=np.uint32, mode="r"
        )
        self.byte_starts = np.memmap(
            self.root / f"{split}.byte_starts.bin", dtype=np.uint32, mode="r"
        )
        self.byte_ends = np.memmap(
            self.root / f"{split}.byte_ends.bin", dtype=np.uint32, mode="r"
        )
        self.documents = np.load(
            self.root / f"{split}.documents.npy", mmap_mode="r", allow_pickle=False
        )
        if not (len(self.tokens) == len(self.byte_starts) == len(self.byte_ends)):
            raise ValueError("prepared token and byte arrays are misaligned")

    def eligible_documents(self, sequence_tokens: int) -> np.ndarray:
        lengths = self.documents[:, 1] - self.documents[:, 0]
        eligible = np.flatnonzero(lengths >= sequence_tokens)
        if not len(eligible):
            raise ValueError(f"no {self.split} document has {sequence_tokens} tokens")
        return eligible

    def sample(
        self,
        batch_size: int,
        sequence_tokens: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> TokenBatch:
        eligible = self.eligible_documents(sequence_tokens)
        choices = torch.randint(
            0, len(eligible), (batch_size,), generator=generator
        ).numpy()
        output = np.empty((batch_size, sequence_tokens), dtype=np.int64)
        document_ids = np.empty(batch_size, dtype=np.int64)
        source_bytes = 0
        for row, choice in enumerate(choices):
            document_id = int(eligible[choice])
            token_start, token_end, _, _ = self.documents[document_id]
            available = int(token_end - token_start - sequence_tokens + 1)
            relative = (
                int(torch.randint(0, available, (1,), generator=generator).item())
                if available > 1
                else 0
            )
            start = int(token_start) + relative
            end = start + sequence_tokens
            output[row] = self.tokens[start:end]
            first_byte = int(self.byte_starts[start])
            last_byte = int(self.byte_ends[end - 1])
            source_bytes += max(last_byte - first_byte, 0)
            document_ids[row] = document_id
        return TokenBatch(
            token_ids=torch.from_numpy(output).to(device=device, non_blocking=True),
            source_bytes=source_bytes,
            document_ids=torch.from_numpy(document_ids),
        )


class DeterministicBatchPrefetcher:
    """Build one pinned CPU batch ahead without speculative RNG commits.

    The producer owns a clone of the sampling generator.  The caller's
    generator advances only when ``next`` consumes the completed batch, so a
    checkpoint taken with a future in flight resumes with the exact same next
    random value, documents, offsets, and tokens as synchronous sampling.
    """

    def __init__(
        self,
        corpus: PackedCorpus,
        *,
        batch_size: int,
        sequence_tokens: int,
        generator: torch.Generator,
        pin_memory: bool,
    ) -> None:
        self._corpus = corpus
        self._batch_size = int(batch_size)
        self._sequence_tokens = int(sequence_tokens)
        self._generator = generator
        self._pin_memory = bool(pin_memory)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ztok-batch")
        self._closed = False
        self._future = self._submit()

    def _submit(self) -> Future[tuple[float, TokenBatch, torch.Tensor]]:
        state = self._generator.get_state().clone()

        def produce() -> tuple[float, TokenBatch, torch.Tensor]:
            local = torch.Generator()
            local.set_state(state)
            random_value = float(torch.rand((), generator=local))
            batch = self._corpus.sample(
                self._batch_size,
                self._sequence_tokens,
                generator=local,
                device=torch.device("cpu"),
            )
            if self._pin_memory:
                batch = TokenBatch(
                    token_ids=batch.token_ids.pin_memory(),
                    source_bytes=batch.source_bytes,
                    document_ids=batch.document_ids,
                )
            return random_value, batch, local.get_state()

        return self._pool.submit(produce)

    def next(self, device: torch.device) -> tuple[float, TokenBatch]:
        if self._closed:
            raise RuntimeError("batch prefetcher is closed")
        random_value, batch, committed_state = self._future.result()
        self._generator.set_state(committed_state)
        self._future = self._submit()
        return random_value, batch.to(device)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._future.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
