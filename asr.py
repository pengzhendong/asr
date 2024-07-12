# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import librosa
import sherpa_onnx
import wenet
import whisperx
from funasr import AutoModel
from modelscope import snapshot_download


class FunASR:
    def __init__(self):
        self.model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad")
        self.pattern = r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])'

    def compute(self, wav_path):
        segments = self.model.generate(wav_path)
        text = segments[0]["text"]
        while re.search(self.pattern, text):
            text = re.sub(self.pattern, r'\1\2', text)
        return text


class SherpaONNX:
    def __init__(self):
        self.sample_rate = 16000
        repo_dir = snapshot_download("pengzhendong/offline-paraformer-zh")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=f"{repo_dir}/model.onnx",
            tokens=f"{repo_dir}/tokens.txt",
            num_threads=1,
            sample_rate=self.sample_rate,
            feature_dim=80
        )

    def compute(self, wav_path):
        audio, _ = librosa.load(wav_path, sr=self.sample_rate)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text


class WeNet:
    def __init__(self):
        self.model = wenet.load_model("chinese")

    def compute(self, wav_path):
        result = self.model.transcribe(wav_path)
        return result["text"]


class WhisperX:
    def __init__(self, device="cuda"):
        assert device in ["cpu", "cuda"]
        repo_dir = snapshot_download("pengzhendong/faster-whisper-medium")
        self.model = whisperx.load_model(repo_dir, device, compute_type="float16")

    def compute(self, wav_path, batch_size=1):
        audio = whisperx.load_audio(wav_path)
        segments = self.model.transcribe(audio, batch_size=batch_size)["segments"]
        return segments[0]["text"]
