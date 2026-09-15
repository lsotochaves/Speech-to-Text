import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pathlib import Path


device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
)


test_audio_dir = Path("./test_audios")

test_audio_file = test_audio_dir / "Keyness.wav"

result = pipe(
    str(test_audio_file),
    chunk_length_s=30,
    stride_length_s=5,
    batch_size=4,
    return_timestamps=True,
)


output_file = Path("./txt_results") / "keyness.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write(result["text"])
