import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torchaudio

class AudioDeepfakeDetector:
    def __init__(self):
        model_name = "mo-thecreator/wav2vec2-base-finetuned"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, audio_path):
        # Load audio and resample if needed
        speech, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            speech = resampler(speech)

        # Extract features
        inputs = self.feature_extractor(speech[0], sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
        label = self.model.config.id2label[predicted_class]

        return {"label": label, "confidence": confidence}
