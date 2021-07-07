import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_batch_ops import batch


def get_model(num_class) -> tf.keras.models.Sequential:
    # モデルの構築
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_class, activation="softmax"))

    # オプティマイザーと評価指標の設定
    adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=adam,
        metrics=["accuracy"],
    )

    return model


def train(
    x_train: np.array,
    y_train: np.array,
    label_index: dict,
    x_validation: np.array = None,
    y_validation: np.array = None,
    validation_rate: float = None,
    log_dir: str = "/kqi/output/log",
    output_dir: str = "/kqi/output/demo",
    batch_size: int = 64,
    early_stopping_patiens: int = 10,
):
    # validation data の作成
    if x_validation != None and y_validation != None:
        pass
    elif validation_rate != None:
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train, y_train, test_size=validation_rate, shuffle=True, random_state=42
        )
    else:
        raise ValueError("unable to create validation data")

    # data generator の宣言
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        data_format="channels_last",
    )
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

    # コールバックの設定
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patiens
    )
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True
    )
    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{output_dir}/params",
        save_weights_only=False,
        monitor="val_loss",
        save_best_only=True,
    )

    # モデルの作成
    model = get_model(len(label_index))

    # 学習実行
    model.fit_generator(
        train_generator,
        steps_per_epoch=x_train.shape[0] // batch_size,
        verbose=2,
        epochs=100,
        validation_data=(x_validation, y_validation),
        callbacks=[es, tb, cp],
    )

    # confusion matrixの作成
    y_pred = np.argmax(model.predict(x_validation), axis=-1)
    cm = confusion_matrix(y_validation, y_pred, labels=list(label_index.values()))
    cm = pd.DataFrame(cm, columns=label_index.keys(), index=label_index.keys())
    cm.to_csv(f"{output_dir}/confusion_matrix.csv")

    # tensorboard への画像の出力
    writer = tf.summary.create_file_writer(f"{log_dir}/images")

    # 各クラスについて間違えた画像のみを10枚ずつ収集してtensorboardで表示する
    wrong_pictures_idx = [
        idx for idx in range(len(y_pred)) if y_pred[idx] != y_validation[idx]
    ]

    for image_idx in wrong_pictures_idx[:20]:
        true_label = [
            key for key, val in label_index.items() if val == y_validation[image_idx]
        ]
        predicted_label = [
            key for key, val in label_index.items() if val == y_pred[image_idx]
        ]
        title = f"true {true_label}: predicted {predicted_label}"

        with writer.as_default():
            tf.summary.image(
                title, x_validation[image_idx : image_idx + 1], step=100, max_outputs=1
            )

    with open(f"{output_dir}/label_idx.pkl", "bw") as f:
        pickle.dump(label_index, f)
