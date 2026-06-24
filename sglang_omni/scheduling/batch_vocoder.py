"""Base class for batched vocoders.  (keke0315)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

PreparedItemT = TypeVar("PreparedItemT")
DecodedItemT = TypeVar("DecodedItemT")


class BatchVocoderBase(ABC, Generic[PreparedItemT, DecodedItemT]):
    @abstractmethod
    def prepare_item(self, payload: StagePayload) -> PreparedItemT:
        """Prepare a payload for decoding."""

    @abstractmethod
    def decode_batch(self, items: list[PreparedItemT]) -> list[DecodedItemT]:
        """Decode a batch of items."""

    @abstractmethod
    def store_result(
        self,
        payload: StagePayload,
        item: PreparedItemT,
        decoded: DecodedItemT,
    ) -> StagePayload:
        """Store a decoded item in its payload."""

    def batch_size_mismatch_error(self, *, actual: int, expected: int) -> Exception:
        return RuntimeError(
            f"{self.__class__.__name__}.decode_batch returned {actual} items for {expected} requests"
        )

    def compute(self, payload: StagePayload) -> StagePayload:
        return self.compute_batch([payload])[0]

    def compute_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self.prepare_item(payload) for payload in payloads]
        decoded_items = self.decode_batch(items)
        if len(decoded_items) != len(items):
            raise self.batch_size_mismatch_error(
                actual=len(decoded_items), expected=len(items)
            )
        return [
            self.store_result(payload, item, decoded)
            for payload, item, decoded in zip(payloads, items, decoded_items)
        ]

    def create_scheduler(
        self,
        *,
        max_batch_size: int,
        max_batch_wait_ms: int,
    ) -> SimpleScheduler:
        return SimpleScheduler(
            self.compute,
            batch_compute_fn=self.compute_batch,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )
