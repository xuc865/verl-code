import os


def main():
    from vllm import LLM, SamplingParams

    model_path = os.environ.get(
        "QWEN35_MODEL_PATH",
        "/mnt/z4/solariewang/models/Qwen3.5-4B",
    )
    tensor_parallel_size = int(os.environ.get("TP", "1"))

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        language_model_only=True,
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "2048")),
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTIL", "0.70")),
    )
    outputs = llm.generate(
        ["Write a Python function add(a, b)."],
        SamplingParams(max_tokens=64),
    )
    print(outputs[0].outputs[0].text)
    print("vllm_qwen35_ok")


if __name__ == "__main__":
    main()
