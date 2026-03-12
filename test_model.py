from vllm import LLM, SamplingParams

if __name__ == '__main__':
    llm = LLM(
        model="Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    prompt = "Hello, how are you today?"
    output = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0.8))
    print(output[0].outputs[0].text)