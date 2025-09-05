import tensorflow as tf
import pandas as pd
import pathlib
import os

# PARAMETERS
BATCH_SIZE = 32
IMG_SIZE = (224,224)
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 12
CSV_TRAIN = "train_labels.csv"   # CSV with columns: image_path,label
CSV_VAL = "val_labels.csv"       # optional

# 1) Read CSV -> tf.data
def make_dataset_from_csv(csv_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, repeat=False):
    df = pd.read_csv(csv_path)
    image_paths = df["image_path"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    classes = sorted(list(set(labels)))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    labels_idx = [class_to_idx[l] for l in labels]
ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_idx))

    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(lambda p,l: tf.py_function(func=_load, inp=[p,l], Tout=(tf.float32, tf.int32)),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, classesds = tf.data.Dataset.from_tensor_slices((image_paths, labels_idx))

    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(lambda p,l: tf.py_function(func=_load, inp=[p,l], Tout=(tf.float32, tf.int32)),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, classes
ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_idx))

    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(lambda p,l: tf.py_function(func=_load, inp=[p,l], Tout=(tf.float32, tf.int32)),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, classes
ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_idx))

    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(lambda p,l: tf.py_function(func=_load, inp=[p,l], Tout=(tf.float32, tf.int32)),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, classes
if val_ds is not None:
    import numpy as np
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    from sklearn.metrics import classification_report, confusion_matrix
    print("Classes:", classes)
    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

# 6) Save TF SavedModel for deployment
model.save("plant_disease_saved_model")
print("Saved model and saved_model dir.")
