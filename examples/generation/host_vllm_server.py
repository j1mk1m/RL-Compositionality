#!/usr/bin/env python3
"""
Launch a vLLM OpenAI-compatible server with memory-focused defaults.
"""

import argparse
import os
import shlex
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Host a vLLM OpenAI API server.")
    parser.add_argument("--model", required=True, help="HF model path or model ID.")
    parser.add_argument("--host", default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--served-model-name", default=None, help="Public model name exposed by the API.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel degree.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype, e.g. bfloat16/float16.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="Fraction of each GPU memory usable by vLLM. Lower to avoid OOM.",
    )
    parser.add_argument("--max-model-len", type=int, default=None, help="Max context length on server side.")
    parser.add_argument("--max-num-seqs", type=int, default=128, help="Max in-flight sequences.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Max batched tokens per scheduling step.",
    )
    parser.add_argument("--swap-space", type=float, default=8.0, help="CPU swap space (GiB) for KV spill.")
    parser.add_argument("--disable-log-requests", action="store_true", help="Disable per-request logging.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code to transformers.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph capture.")
    parser.add_argument(
        "--extra-vllm-args",
        default="",
        help='Extra args appended verbatim, e.g. "--kv-cache-dtype fp8 --seed 42".',
    )
    return parser.parse_known_args()


def main() -> None:
    args, unknown = parse_args()

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--swap-space",
        str(args.swap_space),
    ]
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.disable_log_requests:
        cmd.append("--disable-log-requests")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.extra_vllm_args:
        cmd.extend(shlex.split(args.extra_vllm_args))
    if unknown:
        cmd.extend(unknown)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("Launching vLLM server with command:")
    print(" ".join(shlex.quote(token) for token in cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
