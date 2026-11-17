#!/usr/bin/env python3
"""
Throughput of a d1 ONNX export by batch size: one request's questions over text states of mixed
lengths (the request's state repeated 1 to --max-repeats times), timed end to end
(system_one_batch: prompts, tokenizer, forwards and readouts) in two ways.

    --batch-sizes     calls of 1, 2, 4, ... requests, each call one forward of right-padded rows.
                      Every call of --max-repeats requests or more holds every length: the most
                      padding a batch can have. A request is one row when its questions fit
                      d1_set_size.
    --token-budgets   one call of all the requests under each budget, as a server would batch
                      them: d1.session sorts the rows by length into forwards of at most that
                      many padded tokens (rows x longest row).

Logs requests per second and the gain over the first batch size listed.

Usage:
    uv run d1-bench --model exports/d1-2B-ONNX --request samples/d1/ticket.json
    uv run --no-sync d1-bench --model exports/d1-2B-ONNX --request samples/d1/ticket.json \\
        --precision q4f16 --requests 256 --batch-sizes 1 4 16 64 --token-budgets 4096 16384
"""

import argparse
import json
import logging
import pathlib
import time

from liquidonnx.d1.session import BUNDLES, load
from liquidonnx.lfm3.session import default_ep

logger = logging.getLogger(__name__)


class CountedSession:
    """A session that counts its runs: the forwards of a call."""

    def __init__(self, session):
        self.session, self.runs = session, 0

    def run(self, outputs, feed):
        self.runs += 1
        return self.session.run(outputs, feed)


def best_time(run, rounds: int) -> float:
    """Seconds of the fastest of rounds calls of run, after one untimed call: the EP's kernels
    and allocations for its shapes."""
    run()
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    parser = argparse.ArgumentParser(description="Batched throughput of a d1 ONNX export")
    parser.add_argument("--model", required=True, help="Export folder")
    parser.add_argument("--request", required=True, help="Request JSON: a text state, questions")
    parser.add_argument("--precision", choices=sorted(BUNDLES), help="default: as d1-infer")
    parser.add_argument("--requests", type=int, default=64, help="Requests timed per batch size")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--token-budgets",
        type=int,
        nargs="*",
        default=[],
        help="Also time all the requests in one call under each of these token budgets",
    )
    parser.add_argument("--max-repeats", type=int, default=4, help="Longest state, in repeats")
    parser.add_argument("--rounds", type=int, default=3, help="Timed rounds; the best is kept")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (skip CUDA)")
    parser.add_argument("--no-tf32", action="store_true", help="Keep CUDA fp32 math in fp32")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    request = json.loads(pathlib.Path(args.request).read_text())
    state, questions = request["state"], request["questions"]
    if not isinstance(state, str):
        parser.error("--request needs a text state")
    requests = [
        ("\n".join([state] * (1 + i % args.max_repeats)), questions, None)
        for i in range(args.requests)
    ]
    ep = default_ep(args.cpu)
    model = load(args.model, args.precision, ep, tf32=not args.no_tf32)
    # one request of each length, one row a forward
    sample = requests[: args.max_repeats]
    tokens = [out["usage"]["input_tokens"] for out in model.system_one_batch(sample)]
    logger.info(
        f"{args.model}: {model.precision} on {ep}, {len(requests)} requests of "
        f"{len(questions)} questions, {min(tokens)} to {max(tokens)} tokens "
        f"(mean {sum(tokens) / len(tokens):.0f})"
    )

    def report(label: str, seconds: float, baseline: float | None) -> float:
        rate = len(requests) / seconds
        gain = f", {rate / baseline:.2f}x of batch {args.batch_sizes[0]}" if baseline else ""
        logger.info(
            f"{label}: {rate:8.2f} requests/s, {seconds * 1000 / len(requests):8.1f} ms each{gain}"
        )
        return rate

    decoder = model.sessions["decoder"] = CountedSession(model.sessions["decoder"])

    # === 1. Calls of a fixed number of requests, each one forward ===
    model.token_budget = 2**62  # never the limit: a call's rows are one forward
    baseline = None
    for size in args.batch_sizes:
        groups = [requests[i : i + size] for i in range(0, len(requests), size)]
        seconds = best_time(
            lambda groups=groups: [model.system_one_batch(g) for g in groups], args.rounds
        )
        rate = report(f"batch {size:3d}", seconds, baseline)
        baseline = baseline or rate

    # === 2. One call of all the requests, batched by length under a token budget ===
    for budget in args.token_budgets:
        model.token_budget = budget
        decoder.runs = 0
        seconds = best_time(lambda: model.system_one_batch(requests), args.rounds)
        forwards = decoder.runs // (args.rounds + 1)
        report(f"budget {budget:6d} ({forwards} forwards)", seconds, baseline)


if __name__ == "__main__":
    main()
