# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang_omni.models.fun_asr import sglang_model  # noqa: E402

FunASRModel = sglang_model.FunAsrNanoForConditionalGeneration


def _make_model(max_bytes: int) -> FunASRModel:
    model = FunASRModel.__new__(FunASRModel)
    torch.nn.Module.__init__(model)
    model.audio_adaptor = torch.nn.Linear(4, 4)
    model.init_encoder_cache(max_bytes)
    return model


def _stub_encode(model: FunASRModel):
    calls = {"count": 0}

    def _fake(items):  # noqa: ANN001
        calls["count"] += 1
        return torch.ones(4)

    model._get_audio_feature_uncached = _fake  # type: ignore[assignment]
    return calls


def _item(audio_hash: int) -> SimpleNamespace:
    return SimpleNamespace(hash=audio_hash)


def test_identical_hash_encodes_once() -> None:
    model = _make_model(max_bytes=1 << 20)
    calls = _stub_encode(model)

    first = model.get_audio_feature([_item(123)])
    second = model.get_audio_feature([_item(123)])

    assert calls["count"] == 1
    assert torch.equal(first, second)


def test_disabled_cache_always_encodes() -> None:
    model = _make_model(max_bytes=0)
    calls = _stub_encode(model)

    assert model._encoder_cache is None
    model.get_audio_feature([_item(7)])
    model.get_audio_feature([_item(7)])

    assert calls["count"] == 2


def test_multi_item_batch_bypasses_cache() -> None:
    model = _make_model(max_bytes=1 << 20)
    calls = _stub_encode(model)

    model.get_audio_feature([_item(1), _item(2)])
    model.get_audio_feature([_item(1), _item(2)])

    assert calls["count"] == 2


def test_lru_evicts_when_over_budget() -> None:
    model = _make_model(max_bytes=16)
    calls = _stub_encode(model)

    model.get_audio_feature([_item(1)])
    model.get_audio_feature([_item(2)])
    model.get_audio_feature([_item(1)])

    assert calls["count"] == 3
    assert model._encoder_cache is not None
    assert model._encoder_cache.eviction_count >= 1
