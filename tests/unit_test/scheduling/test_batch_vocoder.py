# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pytest

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.batch_vocoder import BatchVocoderBase


@dataclass(frozen=True)
class _Prepared:
    request_id: str
    value: int


@dataclass(frozen=True)
class _Decoded:
    value: int


class _FakeBatchVocoder(BatchVocoderBase[_Prepared, _Decoded]):
    def __init__(self) -> None:
        self.compute_batch_calls: list[list[str]] = []
        self.prepare_calls: list[str] = []
        self.decode_calls: list[list[_Prepared]] = []
        self.store_calls: list[str] = []
        self.call_order: list[str] = []
        self.decode_result_count: int | None = None
        self.prepare_error: Exception | None = None
        self.store_error: Exception | None = None

    def compute_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        self.compute_batch_calls.append([payload.request_id for payload in payloads])
        return super().compute_batch(payloads)

    def prepare_item(self, payload: StagePayload) -> _Prepared:
        self.prepare_calls.append(payload.request_id)
        self.call_order.append(f"prepare:{payload.request_id}")
        if self.prepare_error is not None:
            raise self.prepare_error
        return _Prepared(payload.request_id, payload.data["value"])

    def decode_batch(self, items: list[_Prepared]) -> list[_Decoded]:
        self.decode_calls.append(list(items))
        self.call_order.append("decode:" + ",".join(item.request_id for item in items))
        count = (
            len(items) if self.decode_result_count is None else self.decode_result_count
        )
        return [
            _Decoded(items[index % len(items)].value * 10) for index in range(count)
        ]

    def store_result(
        self,
        payload: StagePayload,
        item: _Prepared,
        decoded: _Decoded,
    ) -> StagePayload:
        self.store_calls.append(payload.request_id)
        self.call_order.append(f"store:{payload.request_id}")
        if self.store_error is not None:
            raise self.store_error
        assert item.request_id == payload.request_id
        payload.data = {"value": decoded.value}
        return payload


def _payload(request_id: str, value: int) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"value": value}),
        data={"value": value},
    )


def test_batch_vocoder_compute_reuses_batch_backbone() -> None:
    vocoder = _FakeBatchVocoder()
    payload = _payload("single", 3)

    result = vocoder.compute(payload)

    assert vocoder.compute_batch_calls == [["single"]]
    assert vocoder.decode_calls == [[_Prepared("single", 3)]]
    assert result is payload
    assert result.data == {"value": 30}


def test_batch_vocoder_compute_batch_preserves_order() -> None:
    vocoder = _FakeBatchVocoder()
    payloads = [_payload("first", 1), _payload("second", 2), _payload("third", 3)]

    results = vocoder.compute_batch(payloads)

    assert vocoder.prepare_calls == ["first", "second", "third"]
    assert vocoder.decode_calls == [
        [
            _Prepared("first", 1),
            _Prepared("second", 2),
            _Prepared("third", 3),
        ]
    ]
    assert vocoder.store_calls == ["first", "second", "third"]
    assert results == payloads
    assert [result.request_id for result in results] == [
        "first",
        "second",
        "third",
    ]
    assert [result.data["value"] for result in results] == [10, 20, 30]


@pytest.mark.parametrize("decoded_count", [1, 3])
def test_batch_vocoder_compute_batch_rejects_decode_count_mismatch(
    decoded_count: int,
) -> None:
    vocoder = _FakeBatchVocoder()
    vocoder.decode_result_count = decoded_count

    with pytest.raises(
        RuntimeError,
        match=rf"_FakeBatchVocoder\.decode_batch returned {decoded_count} items for 2 requests",
    ):
        vocoder.compute_batch([_payload("first", 1), _payload("second", 2)])

    assert len(vocoder.decode_calls[0]) == 2
    assert vocoder.store_calls == []


def test_batch_vocoder_compute_batch_propagates_prepare_error() -> None:
    vocoder = _FakeBatchVocoder()
    error = ValueError("prepare failed")
    vocoder.prepare_error = error

    with pytest.raises(ValueError) as exc_info:
        vocoder.compute_batch([_payload("first", 1)])

    assert exc_info.value is error
    assert vocoder.call_order == ["prepare:first"]
    assert vocoder.prepare_calls == ["first"]
    assert vocoder.decode_calls == []
    assert vocoder.store_calls == []


def test_batch_vocoder_compute_batch_propagates_store_error() -> None:
    vocoder = _FakeBatchVocoder()
    error = LookupError("store failed")
    vocoder.store_error = error

    with pytest.raises(LookupError) as exc_info:
        vocoder.compute_batch([_payload("first", 1)])

    assert exc_info.value is error
    assert vocoder.call_order == ["prepare:first", "decode:first", "store:first"]
    assert vocoder.prepare_calls == ["first"]
    assert len(vocoder.decode_calls) == 1
    assert vocoder.store_calls == ["first"]


def test_batch_vocoder_create_scheduler_sets_batch_contract() -> None:
    vocoder = _FakeBatchVocoder()

    scheduler = vocoder.create_scheduler(
        max_batch_size=2,
        max_batch_wait_ms=200,
    )

    assert scheduler._fn == vocoder.compute
    assert scheduler._batch_fn == vocoder.compute_batch
    assert scheduler._max_batch_size == 2
    assert scheduler._max_batch_wait_s == pytest.approx(0.2)
