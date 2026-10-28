"""Download a pinned, hash-checked anime-head reference model for experiments."""

import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen


REVISION = "06604feee81983792a57c21081e539c0ae229833"
URL = (f"https://huggingface.co/deepghs/anime_head_detection/resolve/{REVISION}/"
       "head_detect_v2.0_n/model.onnx")
SHA256 = "69b069d15c9b97e0d7e6db33ac1c3fd480b1be67b078922c29beacfac81ad9ac"
SIZE = 12104715


def download(path):
    path = Path(path)
    if path.exists():
        raise ValueError("目标文件已存在，请换一个 --output")
    with urlopen(URL, timeout=30) as response:
        data = response.read(SIZE + 1)
    if len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("模型文件长度或 SHA-256 校验失败，未保存文件")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载固定版本的动漫人头研究参考模型（约 12 MB）")
    parser.add_argument("--output", default="models/anime-head.onnx")
    args = parser.parse_args(argv)
    try:
        print(f"已保存：{download(args.output)}")
    except (ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
