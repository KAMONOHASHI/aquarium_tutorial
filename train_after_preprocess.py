# -*- coding:utf-8 -*-

import argparse
import glob
import os
import pickle

from trainer import train
from config import validation_rate


def main(args):
    # データ読み込み
    with open(glob.glob(f"{args.input_path}/**/train_test_datas.pkl")[0], "br") as f:
        all_images, all_labels = pickle.load(f)
    with open(glob.glob(f"{args.input_path}/**/labels_idx.pkl")[0], "br") as f:
        label_index = pickle.load(f)

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
    parser.add_argument("--input_path", default="/kqi/parent")
    parser.add_argument("--output_path", default="/kqi/output/demo")
    parser.add_argument("--log_path", default="/kqi/output/logs")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    main(args)
