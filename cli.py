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

import click

from asr import FunASR, SherpaONNX, WeNet, WhisperX


@click.command()
@click.argument("wav_path", type=click.Path(exists=True, file_okay=True))
@click.option(
    "--engine",
    type=click.Choice(["funasr", "sherpa-onnx", "wenet", "whisperx"]),
    default="funasr",
)
def main(wav_path, engine):
    if engine == "funasr":
        model = FunASR()
    elif engine == "sherpa-onnx":
        model = SherpaONNX()
    elif engine == "wenet":
        model = WeNet()
    elif engine == "whisperx":
        model = WhisperX()

    text = model.compute(wav_path)
    print(text)


if __name__ == "__main__":
    main()
