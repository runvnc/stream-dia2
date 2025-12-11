from dia2 import Dia2, GenerationConfig, SamplingConfig

dia = Dia2.from_repo("nari-labs/Dia2-2B", device="cuda", dtype="bfloat16")
config = GenerationConfig(
    cfg_scale=2.0,
    audio=SamplingConfig(temperature=0.8, top_k=50),
    use_cuda_graph=True,
)
result = dia.generate("[S1] Hello Dia2!", config=config, output_wav="hello.wav", verbose=True)
