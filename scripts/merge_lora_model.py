#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Base model path or HF repo")
    parser.add_argument("--lora_path", required=True, help="LoRA adapter checkpoint dir")
    parser.add_argument("--output_dir", required=True, help="Where to save merged HF model")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, args.lora_path)
    merged = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()