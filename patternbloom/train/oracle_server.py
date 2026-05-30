"""Frozen oracle service: a small instruction-tuned model exposed over HTTP."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    vllm_args = [
        "vllm",
        "serve",
        args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
        "--served-model-name", "oracle",
    ]

    sys.argv = vllm_args
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
