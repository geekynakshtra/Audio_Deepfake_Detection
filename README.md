Audio Deepfake Detection with Wav2Vec2

This project uses a fine-tuned [Wav2Vec2](https://huggingface.co/mo-thecreator/wav2vec2-base-finetuned) model from Hugging Face to classify audio clips as either real or spoof (fake/deepfake).

Features

- Load and preprocess audio files
- Use a pretrained Wav2Vec2 model for sequence classification
- Predict whether an audio is genuine or generated
- Output the classification label and confidence score

---

 Model

- Model Used:** [`mo-thecreator/wav2vec2-base-finetuned`](https://huggingface.co/mo-thecreator/wav2vec2-base-finetuned)
- Task: Audio deepfake detection (bonafide vs spoof)
- Framework: PyTorch + Hugging Face Transformers + torchaudio

---
