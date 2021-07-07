import argparse
import os
import pathlib
import glob
import pickle

import numpy as np
import tensorflow as tf
from skimage import img_as_float, io


def main(args):
    # 推論するデータの読み込み
    path = pathlib.Path(f"{args.input_path}")
    all_image_paths = [item.resolve() for item in path.glob("**/*") if item.is_file()]
    all_images = np.array(
        [
            img_as_float(io.imread(path, as_gray=True))[:, :, np.newaxis]
            for path in all_image_paths
        ]
    )

    # ラベル情報の読み込み
    with open(
        glob.glob(f"{args.model_saved_path}/demo/label_idx.pkl")[0], "br"
    ) as f:
        label_index = pickle.load(f)

    # モデルの読み込み
    model_path = glob.glob(f"{args.model_saved_path}/demo/params")[0]
    model = tf.keras.models.load_model(model_path)

    # 予測
    y_pred = np.argmax(model.predict(all_images), axis=-1)

    # tensorboard への画像の出力
    writer = tf.summary.create_file_writer(f"{args.log_path}/images")
    for image_idx in range(len(all_image_paths)):
        predicted_label = [
            key for key, val in label_index.items() if val == y_pred[image_idx]
        ]
        title = f"{str(all_image_paths[image_idx]).split('/')[-1]}:predicted {predicted_label}"

        with writer.as_default():
            tf.summary.image(
                title, all_images[image_idx : image_idx + 1], step=0, max_outputs=1
            )


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="aqualium demo")
    parser.add_argument("--input_path", default="/kqi/input/images")
    parser.add_argument("--model_saved_path", default="/kqi/parent")
    parser.add_argument("--output_path", default="/kqi/output/demo")
    parser.add_argument("--log_path", default="/kqi/output/logs")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    main(args)
