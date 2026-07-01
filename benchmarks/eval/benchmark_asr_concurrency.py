# SPDX-License-Identifier: Apache-2.0
# Author:
# PoTaTo-Mika: https://github.com/PoTaTo-Mika
"""ASR concurrency-scaling benchmark on SeedTTS EN (issue #646 & #898).

Sweeps ASR transcription fan-out (concurrency) against a running ASR
SGLang Omni router and reports, for each concurrency level, the metrics tracked
in issue #646: corpus/per-sample WER, wall-clock, throughput, latency
percentiles, RTF, and per-worker routing balance. This produces the repeatable
concurrency-scaling data the issue's acceptance criteria ask for, and lets us
decide the right ASR fan-out for SeedTTS EN transcription / WER workloads.

This script transcribes the SeedTTS reference clips directly (no TTS
generation step), so it isolates ASR behavior from TTS.

Both Qwen3-ASR and Fun-ASR-Nano are supported. The per-model HTTP knobs
(max_new_tokens, whether to send temperature) live in
benchmarks.tasks.tts.make_asr_send_fn; this file is the eval checklist.

run_asr_transcription and build_asr_eval_results are the shared
transcription/scoring path; the Qwen3-ASR correctness gate
(tests/test_model/test_qwen3_asr_ci.py) imports them so the gate is just
this benchmark run plus thresholds. Both reuse the benchmark framework
abstractions (BenchmarkRunner, benchmarks.metrics).

Usage:

1. Download the test set once:

    python -m benchmarks.dataset.prepare --dataset seedtts

2. Full sweep (the benchmark starts/stops the ASR server itself):

    python -m benchmarks.eval.benchmark_asr_concurrency \
        --model-path Qwen/Qwen3-ASR-1.7B \
        --port 8000 \
        --concurrencies 1,2,4,8,16,32,64 \
        --repeats 3

    # Same sweep against Fun-ASR-Nano:
    python -m benchmarks.eval.benchmark_asr_concurrency \
        --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
        --concurrencies 1,2,4,8,16,32,64 --repeats 3

3. Against an already-running server (skip server lifecycle):

    python -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --port 8000
    python -m benchmarks.eval.benchmark_asr_concurrency \
        --use-existing-server --port 8000 \
        --concurrencies 1,2,4,8,16,32,64 --repeats 3

4. Quick local smoke on a 20-sample subset:

    python -m benchmarks.eval.benchmark_asr_concurrency \
        --use-existing-server --port 8000 \
        --max-samples 20 --concurrencies 2,32 --repeats 3

Reference Results

Reproducibility references for the FULL eval set — NOT CI thresholds.
CI runs on a subset and has its own thresholds (tests/test_model/test_qwen3_asr_ci.py).

Benchmark: SeedTTS EN  |  Dataset: seed-tts-eval, full set (EN=1088)
Hardware:  1 x NVIDIA RTX 4080 SUPER, 32 GB (single GPU, DP=1)
Last verified: 2026-06-30

Run config (aligned): dtype bf16; 3 repeats + 1 discarded warmup per level;
concurrency 1/2/4/8/16/32/64; audio duration mean 4.69s, median 4.53s,
max 8.81s, total 85.1 min. Qwen3-ASR uses max_new_tokens=128 and sends no
temperature (server bumps 0 to 0.01); Fun-ASR-Nano uses max_new_tokens=256
and sends temperature=0.0 (greedy). Both transcribe the same 1088 EN clips.

Qwen3-ASR-1.7B (Qwen/Qwen3-ASR-1.7B)

| conc | wall(s) mean | thrpt/s mean | thrpt/s best | lat_mean(s) | lat_p95(s) | rtf_mean | rtf_p95 | corpus_wer | max_wer | skipped |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1  | 180.46 | 6.03  | 6.08  | 0.165 | 0.201 | 0.0359 | 0.0460 | 0.0133 | 0.2727 | 0  |
| 2  | 106.87 | 10.19 | 10.49 | 0.196 | 0.237 | 0.0426 | 0.0550 | 0.0134 | 0.2500 | 0  |
| 4  | 67.62  | 16.09 | 16.38 | 0.248 | 0.323 | 0.0539 | 0.0741 | 0.0136 | 0.2727 | 0  |
| 8  | 39.37  | 27.64 | 27.98 | 0.288 | 0.383 | 0.0625 | 0.0854 | 0.0131 | 0.1818 | 0  |
| 16 | 26.14  | 41.62 | 41.94 | 0.383 | 0.498 | 0.0829 | 0.1142 | 0.0133 | 0.2500 | 0  |
| 32 | 19.76  | 55.07 | 55.68 | 0.577 | 0.759 | 0.1247 | 0.1678 | 0.0130 | 0.1818 | 0  |
| 64 | 19.45  | 52.79 | 52.98 | 1.187 | 1.423 | 0.2587 | 0.3471 | 0.0137 | 0.2000 | 72 |

Fun-ASR-Nano (FunAudioLLM/Fun-ASR-Nano-2512-hf)

| conc | wall(s) mean | thrpt/s mean | thrpt/s best | lat_mean(s) | lat_p95(s) | rtf_mean | rtf_p95 | corpus_wer | max_wer | skipped |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1  | 88.14 | 12.34 | — | 0.081 | 0.096 | 0.0175 | 0.0222 | 0.0171 | 0.2857 | 0 |
| 2  | 63.14 | 17.23 | — | 0.116 | 0.139 | 0.0252 | 0.0324 | 0.0171 | 0.2857 | 0 |
| 4  | 47.18 | 23.06 | — | 0.173 | 0.226 | 0.0375 | 0.0511 | 0.0171 | 0.2857 | 0 |
| 8  | 36.94 | 29.46 | — | 0.271 | 0.359 | 0.0586 | 0.0800 | 0.0171 | 0.2857 | 0 |
| 16 | 30.03 | 36.23 | — | 0.440 | 0.589 | 0.0953 | 0.1306 | 0.0171 | 0.2857 | 0 |
| 32 | 26.76 | 40.66 | — | 0.784 | 1.060 | 0.1696 | 0.2306 | 0.0171 | 0.2857 | 0 |
| 64 | 27.25 | 39.93 | — | 1.592 | 1.940 | 0.3471 | 0.4657 | 0.0171 | 0.2857 | 0 |

Qwen3-ASR at conc=64 drops 54-72 of 1088 requests per repeat (~5-7%) to
timeouts on a single GPU; WER is computed over the evaluated subset. All other
levels: 0 skipped. This is a single-GPU saturation limit, not a regression
from the refactor. CI runs at conc=32.

Headline: Qwen3-ASR-1.7B is more accurate (corpus WER 0.0130-0.0137 vs
Fun-ASR-Nano 0.0171, both stable across concurrency) and saturates higher
(55.1/s vs 40.7/s at conc=32); Fun-ASR-Nano is ~2x faster at low concurrency
(conc=1 mean latency 0.081s vs 0.165s, RTF 0.0175 vs 0.0359) with zero skipped
requests across all levels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import requests
from jiwer import process_words

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.wer import calculate_asr_speed_metrics, calculate_wer_metrics
from benchmarks.tasks.tts import (
    DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    QWEN3_ASR_MODEL_PATH,
    QWEN3_ASR_REQUEST_TIMEOUT_S,
    SampleOutput,
    make_asr_send_fn,
    normalize_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCIES = "1,2,4,8,16,32,64"


@dataclass
class AsrConcurrencyBenchmarkConfig:
    """All parameters for one ASR concurrency-sweep run (mirrors
    TtsSeedttsBenchmarkConfig in benchmark_tts_seedtts.py)."""

    model_path: str
    port: int
    host: str = "127.0.0.1"
    meta: str = DATASETS["seedtts"]
    lang: str = "en"
    max_samples: int | None = None
    concurrencies: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    repeats: int = 3
    max_new_tokens: int | None = None
    warmup: bool = False
    request_timeout_s: int = QWEN3_ASR_REQUEST_TIMEOUT_S
    disable_tqdm: bool = True
    output: str = "asr_concurrency_results.json"
    use_existing_server: bool = False
    server_timeout: int = 600
    wait_for_gpu_release: bool = True


async def run_asr_transcription(
    samples: list[SampleInput],
    *,
    host: str = "127.0.0.1",
    port: int,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    lang: str = "en",
    concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    warmup: int = 0,
    request_timeout_s: int = QWEN3_ASR_REQUEST_TIMEOUT_S,
    max_new_tokens: int | None = None,
    disable_tqdm: bool = True,
) -> tuple[list[RequestResult], float]:
    """Transcribe samples against a running ASR router at one concurrency.

    Returns (outputs, wall_clock_s) via the shared BenchmarkRunner.
    max_new_tokens=None lets make_asr_send_fn pick the backend default
    (Qwen3-ASR 128, Fun-ASR-Nano 256).
    """
    api_url = f"http://{host}:{port}/v1/audio/transcriptions"
    send_fn = make_asr_send_fn(
        model_path, api_url, lang=lang, max_new_tokens=max_new_tokens
    )
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=concurrency,
            warmup=warmup,
            disable_tqdm=disable_tqdm,
            timeout_s=request_timeout_s,
        )
    )
    outputs = await runner.run(samples, send_fn)
    return outputs, runner.wall_clock_s


def build_asr_eval_results(
    samples: list[SampleInput],
    outputs: list[RequestResult],
    wall_clock_s: float,
    lang: str,
    *,
    model_path: str = QWEN3_ASR_MODEL_PATH,
    concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
) -> dict:
    """Score transcriptions and assemble WER + speed metrics.

    Returns {"summary": wer, "speed": speed, "per_sample": [...]} with the
    exact summary.* / speed.* keys the Qwen3-ASR gate writes and the
    tune-ci-thresholds config reads. WER/speed reuse benchmarks.metrics.
    """
    result_by_id = {result.request_id: result for result in outputs}
    sample_outputs: list[SampleOutput] = []
    per_sample: list[dict] = []
    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.ref_text,
        )
        if result is None or not result.is_success:
            output.error = (result.error if result else "") or "No transcription"
        else:
            output.latency_s = result.latency_s
            output.asr_latency_s = result.latency_s
            output.audio_duration_s = result.audio_duration_s
            output.whisper_text = result.text
            output.ref_norm = normalize_text(sample.ref_text, lang)
            output.hyp_norm = normalize_text(result.text, lang)
            if output.ref_norm:
                measures = process_words(output.ref_norm, output.hyp_norm)
                output.wer = measures.wer
                output.substitutions = measures.substitutions
                output.deletions = measures.deletions
                output.insertions = measures.insertions
                output.hits = measures.hits
                output.is_success = True
            else:
                output.error = "Empty reference after normalization"
        sample_outputs.append(output)
        per_sample.append(
            {
                "id": output.sample_id,
                "is_success": output.is_success,
                "wer": output.wer if output.is_success else None,
                "ref_text": output.target_text,
                "hyp_text": output.whisper_text,
                "ref_norm": output.ref_norm,
                "hyp_norm": output.hyp_norm,
                "audio_duration_s": output.audio_duration_s,
                "latency_s": output.latency_s,
                "error": output.error,
            }
        )

    wer_summary = calculate_wer_metrics(sample_outputs, lang)
    # note (Yue Yin): gate + tune-ci-thresholds read summary.corpus_wer
    wer_summary["corpus_wer"] = wer_summary["wer_corpus"]

    asr_speed = calculate_asr_speed_metrics(sample_outputs, wall_time_s=wall_clock_s)
    # note (Yue Yin): compute_speed_metrics supplies rtf_p95 (the asr metrics omit it)
    perf = compute_speed_metrics(outputs, wall_clock_s=wall_clock_s)
    speed = {
        **asr_speed,
        "asr_model": model_path,
        "asr_concurrency": concurrency,
        "asr_rtf_p95": perf.get("rtf_p95"),
        # note (Yue Yin): plain calibration keys read by tune-ci-thresholds + gate
        "throughput_samples_per_s": asr_speed["asr_throughput_samples_per_s"],
        "latency_mean_s": asr_speed["asr_latency_mean_s"],
        "latency_median_s": asr_speed["asr_latency_median_s"],
        "latency_p95_s": asr_speed["asr_latency_p95_s"],
        "latency_p99_s": asr_speed["asr_latency_p99_s"],
        "rtf_mean": asr_speed["asr_rtf_mean"],
        "rtf_median": asr_speed["asr_rtf_median"],
        "rtf_p95": perf.get("rtf_p95"),
    }
    return {"summary": wer_summary, "speed": speed, "per_sample": per_sample}


def _fetch_worker_snapshot(host: str, port: int) -> dict | None:
    """Best-effort read of the router /workers snapshot (None if unavailable)."""
    try:
        response = requests.get(
            f"http://{host}:{port}/workers",
            timeout=10,
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _worker_delta(before: dict | None, after: dict | None) -> dict:
    """Routed/successful/failed deltas and per-worker routed balance."""
    if not before or not after:
        return {}

    def _by_id(snapshot: dict, key: str) -> dict[str, int]:
        return {
            str(w.get("display_id")): int(w.get(key, 0))
            for w in snapshot.get("workers", [])
        }

    out: dict[str, object] = {}
    for key in ("routed_requests", "successful_requests", "failed_requests"):
        before_by_id = _by_id(before, key)
        after_by_id = _by_id(after, key)
        deltas = {
            wid: after_by_id.get(wid, 0) - before_by_id.get(wid, 0)
            for wid in after_by_id
        }
        out[f"total_{key}"] = sum(deltas.values())
        if key == "routed_requests":
            out["per_worker_routed"] = deltas
    return out


async def _run_repeat(
    config: AsrConcurrencyBenchmarkConfig,
    samples: list[SampleInput],
    concurrency: int,
    repeat: int,
) -> dict:
    before = _fetch_worker_snapshot(config.host, config.port)
    outputs, wall_clock_s = await run_asr_transcription(
        samples,
        host=config.host,
        port=config.port,
        model_path=config.model_path,
        lang=config.lang,
        concurrency=concurrency,
        max_new_tokens=config.max_new_tokens,
    )
    after = _fetch_worker_snapshot(config.host, config.port)

    results = build_asr_eval_results(
        samples,
        outputs,
        wall_clock_s,
        config.lang,
        model_path=config.model_path,
        concurrency=concurrency,
    )
    summary = results["summary"]
    speed = results["speed"]
    return {
        "concurrency": concurrency,
        "repeat": repeat,
        "evaluated": summary["evaluated"],
        "total": summary["total_samples"],
        "skipped": summary["skipped"],
        "corpus_wer": summary["corpus_wer"],
        "per_sample_wer_max": summary["wer_per_sample_max"],
        "wall_clock_s": wall_clock_s,
        "throughput_samples_per_s": speed["throughput_samples_per_s"],
        "latency_mean_s": speed["latency_mean_s"],
        "latency_p95_s": speed["latency_p95_s"],
        "latency_p99_s": speed["latency_p99_s"],
        "rtf_mean": speed["rtf_mean"],
        "rtf_p95": speed["rtf_p95"],
        "worker": _worker_delta(before, after),
    }


def _aggregate(repeats: list[dict]) -> dict:
    """Mean/best/worst across repeats for the headline metrics."""

    def _stat(key: str) -> dict:
        values = [r[key] for r in repeats]
        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "concurrency": repeats[0]["concurrency"],
        "repeats": len(repeats),
        "evaluated": repeats[0]["evaluated"],
        "total": repeats[0]["total"],
        "skipped": repeats[0]["skipped"],
        "corpus_wer": _stat("corpus_wer"),
        "per_sample_wer_max": _stat("per_sample_wer_max"),
        "wall_clock_s": _stat("wall_clock_s"),
        "throughput_samples_per_s": _stat("throughput_samples_per_s"),
        "latency_mean_s": _stat("latency_mean_s"),
        "latency_p95_s": _stat("latency_p95_s"),
        "latency_p99_s": _stat("latency_p99_s"),
        "rtf_mean": _stat("rtf_mean"),
        "rtf_p95": _stat("rtf_p95"),
        "per_repeat": repeats,
    }


def _print_table(aggregates: list[dict]) -> None:
    header = (
        "| conc | reps | wall(s) mean | thrpt mean | thrpt best | "
        "lat mean(s) | lat p95(s) | rtf mean | rtf p95 | corpus WER | max WER |"
    )
    sep = "|---:" * 11 + "|"
    print("\n" + header)
    print(sep)
    for agg in aggregates:
        print(
            f"| {agg['concurrency']} | {agg['repeats']} "
            f"| {agg['wall_clock_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['max']:.3f} "
            f"| {agg['latency_mean_s']['mean']:.3f} "
            f"| {agg['latency_p95_s']['mean']:.3f} "
            f"| {agg['rtf_mean']['mean']:.4f} "
            f"| {agg['rtf_p95']['mean']:.4f} "
            f"| {agg['corpus_wer']['max']:.4f} "
            f"| {agg['per_sample_wer_max']['max']:.4f} |"
        )


async def _sweep(
    config: AsrConcurrencyBenchmarkConfig,
    samples: list[SampleInput],
) -> list[dict]:
    aggregates: list[dict] = []
    for concurrency in config.concurrencies:
        if config.warmup:
            print(f"[conc={concurrency}] warmup pass ...")
            await run_asr_transcription(
                samples,
                host=config.host,
                port=config.port,
                model_path=config.model_path,
                lang=config.lang,
                concurrency=concurrency,
                max_new_tokens=config.max_new_tokens,
            )
        repeats: list[dict] = []
        for repeat in range(1, config.repeats + 1):
            result = await _run_repeat(config, samples, concurrency, repeat)
            repeats.append(result)
            print(
                f"[conc={concurrency} rep={repeat}] "
                f"wall={result['wall_clock_s']:.3f}s "
                f"thrpt={result['throughput_samples_per_s']:.3f}/s "
                f"lat_mean={result['latency_mean_s']:.3f}s "
                f"lat_p95={result['latency_p95_s']:.3f}s "
                f"rtf_mean={result['rtf_mean']:.4f} "
                f"corpus_wer={result['corpus_wer']:.4f} "
                f"skipped={result['skipped']}"
            )
            if result["worker"].get("per_worker_routed"):
                print(f"    routed per worker: {result['worker']['per_worker_routed']}")
        aggregates.append(_aggregate(repeats))
    return aggregates


def _build_results_config(
    config: AsrConcurrencyBenchmarkConfig,
    *,
    num_samples: int,
) -> dict:
    return {
        "model_path": config.model_path,
        "host": config.host,
        "port": config.port,
        "meta": config.meta,
        "lang": config.lang,
        "max_new_tokens": config.max_new_tokens,
        "num_samples": num_samples,
        "concurrencies": list(config.concurrencies),
        "repeats": config.repeats,
        "warmup": config.warmup,
        "use_existing_server": config.use_existing_server,
    }


def benchmark(config: AsrConcurrencyBenchmarkConfig) -> dict:
    """Run the full concurrency sweep and write results JSON.

    Starts/stops the ASR server unless config.use_existing_server is set
    (mirrors benchmark_tts_seedtts.py's managed_omni_server pattern).
    Returns the persisted payload dict.
    """
    samples = load_seedtts_samples(
        config.meta, max_samples=config.max_samples, split=config.lang
    )
    print(
        f"Loaded {len(samples)} SeedTTS {config.lang} samples; "
        f"sweeping concurrency={list(config.concurrencies)} x {config.repeats} repeats "
        f"against {config.host}:{config.port} ({config.model_path})"
    )

    def _run_sweep() -> list[dict]:
        return asyncio.run(_sweep(config, samples))

    if config.use_existing_server:
        aggregates = _run_sweep()
    else:
        with managed_omni_server(
            model_path=config.model_path,
            port=config.port,
            host=config.host,
            log_file=Path(config.output).resolve().parent / "server_logs" / "asr_server.log",
            timeout=config.server_timeout,
            wait_for_gpu_release=config.wait_for_gpu_release,
        ):
            aggregates = _run_sweep()

    _print_table(aggregates)

    payload = {
        "config": _build_results_config(config, num_samples=len(samples)),
        "results": aggregates,
    }
    output_path = os.path.abspath(config.output)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote results to {output_path}")
    return payload


def _config_from_args(args: argparse.Namespace) -> AsrConcurrencyBenchmarkConfig:
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else None
    max_samples = args.max_samples if args.max_samples > 0 else None
    concurrencies = tuple(int(c) for c in args.concurrencies.split(",") if c.strip())
    return AsrConcurrencyBenchmarkConfig(
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        meta=args.meta,
        lang=args.lang,
        max_samples=max_samples,
        concurrencies=concurrencies,
        repeats=args.repeats,
        max_new_tokens=max_new_tokens,
        warmup=args.warmup,
        output=args.output,
        use_existing_server=args.use_existing_server,
        server_timeout=args.server_timeout,
        wait_for_gpu_release=not args.skip_gpu_cleanup,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port of the running ASR SGLang Omni router.",
    )
    parser.add_argument(
        "--meta",
        default=DATASETS["seedtts"],
        help="SeedTTS source (HF repo id or local meta.lst).",
    )
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples (0 = full SeedTTS set; 1088 for EN).",
    )
    parser.add_argument(
        "--concurrencies",
        default=DEFAULT_CONCURRENCIES,
        help="Comma-separated ASR concurrency levels to sweep.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--model-path",
        type=str,
        default=QWEN3_ASR_MODEL_PATH,
        help="HuggingFace model id for the ASR server. Defaults to "
        f"{QWEN3_ASR_MODEL_PATH} (Qwen3-ASR); pass FunAudioLLM/Fun-ASR-Nano-2512-hf "
        "for Fun-ASR-Nano. The backend is resolved from this path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Override max_new_tokens (0 = backend default: Qwen3-ASR 128, Fun-ASR-Nano 256).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one discarded warmup pass before timing each concurrency.",
    )
    parser.add_argument(
        "--output",
        default="asr_concurrency_results.json",
        help="Where to write the full JSON results.",
    )
    parser.add_argument(
        "--use-existing-server",
        action="store_true",
        help="Do not start or stop a server; send requests to the configured "
        "--host/--port instead. Use when a server is already running.",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Timeout in seconds to wait for server readiness.",
    )
    parser.add_argument(
        "--skip-gpu-cleanup",
        action="store_true",
        help="Do not run ensure_gpus_idle after stopping the server. Use when "
        "running multiple benchmark processes in parallel on different GPUs; "
        "combine with CUDA_VISIBLE_DEVICES per worker and clean up each GPU "
        "once after the worker finishes.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = _config_from_args(args)
    benchmark(config)


if __name__ == "__main__":
    main()
