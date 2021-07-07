# -*- coding:utf-8 -*-

import argparse
import os
import pathlib

import numpy as np
from skimage import io, img_as_float

from config import validation_rate
from trainer import train


def main(args):
    # データ読み込み
    path = pathlib.Path(f"{args.input_path}")
    all_image_paths = [item.resolve() for item in path.glob("**/*") if item.is_file()]
    all_images = np.array(
        [
            img_as_float(io.imread(path, as_gray=True))[:, :, np.newaxis]
            for path in all_image_paths
        ]
    )

    if args.input_type == "mnist":
        all_labels = [pathlib.Path(path).parent.name for path in all_image_paths]
    elif args.input_type == "chinese":
        all_labels = [
            pathlib.Path(path).name.split(".")[0].split("_")[-1]
            for path in all_image_paths
        ]
    labels = list(set(all_labels))
    label_index = {label: idx for idx, label in enumerate(labels)}
    all_labels = np.array([label_index[label] for label in all_labels])

    print(label_index)

    train(
        x_train=all_images,
        y_train=all_labels,
        label_index=label_index,
        validation_rate=validation_rate,
        output_dir=args.output_path,
        log_dir=args.log_path,
    )


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="aqualium demo")
    parser.add_argument("--input_path", default="/kqi/input/images")
    parser.add_argument("--output_path", default="/kqi/output/demo")
    parser.add_argument("--log_path", default="/kqi/output/logs")
    parser.add_argument("--input_type", default="mnist")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    main(args)
