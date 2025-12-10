import torch
from dia2.audio.codec import MimiCodec
from transformers import MimiModel

print("Loading Mimi...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MimiModel.from_pretrained("kyutai/mimi").to(device)
codec = MimiCodec(model, device)

print("Testing streaming decode...")
# Create dummy tokens [1, 8, 100]
tokens = torch.randint(0, 2048, (1, 8, 100)).to(device)

# First chunk
chunk1 = tokens[:, :, :50]
audio1, kv1 = codec.decode_streaming(chunk1, None)
print(f"Chunk 1 audio: {audio1.shape}")
print(f"KV cache type: {type(kv1)}")
if kv1 is not None:
    print("KV cache present.")
else:
    print("KV cache is NONE!")

# Second chunk
chunk2 = tokens[:, :, 50:]
audio2, kv2 = codec.decode_streaming(chunk2, kv1)
print(f"Chunk 2 audio: {audio2.shape}")
