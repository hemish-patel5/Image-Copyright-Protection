# ============================================================
# Image Copyright Protection Using Transformer Watermarking
# Kaggle T4 / RAM-safe starter implementation
# ============================================================

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed precision for T4 GPU
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# Avoid TensorFlow taking all GPU memory at once
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

# Optional speed-up
tf.config.optimizer.set_jit(True)

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "/kaggle/input/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR"

IMG_SIZE = 128
BATCH_SIZE = 16          # Use 8 if RAM/GPU memory crashes
WM_BITS = 32             # 32-bit copyright ID
EPOCHS = 30

TRAIN_SPLIT = 0.9
AUTOTUNE = tf.data.AUTOTUNE

# Loss weights
IMAGE_LOSS_WEIGHT = 1.0
WATERMARK_LOSS_WEIGHT = 5.0
RESIDUAL_LOSS_WEIGHT = 0.2

# Watermark strength: lower = more invisible, higher = easier extraction
MAX_RESIDUAL_STRENGTH = 0.08

print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Watermark bits:", WM_BITS)

# ============================================================
# FIND IMAGE FILES
# ============================================================

valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

image_paths = []
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(valid_exts):
            image_paths.append(os.path.join(root, file))

random.shuffle(image_paths)

print("Total images found:", len(image_paths))
print("Example path:", image_paths[0] if image_paths else "No images found")

if len(image_paths) == 0:
    raise ValueError("No image files found. Check DATA_DIR.")

# ============================================================
# TF.DATA PIPELINE
# ============================================================

def load_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img

num_train = int(len(image_paths) * TRAIN_SPLIT)

train_paths = image_paths[:num_train]
val_paths = image_paths[num_train:]

train_ds = (
    tf.data.Dataset.from_tensor_slices(train_paths)
    .shuffle(min(len(train_paths), 2000), seed=SEED)
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices(val_paths)
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))

# ============================================================
# TF.DATA PIPELINE
# ============================================================

def load_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img

num_train = int(len(image_paths) * TRAIN_SPLIT)

train_paths = image_paths[:num_train]
val_paths = image_paths[num_train:]

train_ds = (
    tf.data.Dataset.from_tensor_slices(train_paths)
    .shuffle(min(len(train_paths), 2000), seed=SEED)
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices(val_paths)
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))

# ============================================================
# ATTACK LAYER
# ============================================================

class AttackLayer(layers.Layer):
    def __init__(self, img_size=128):
        super().__init__()
        self.img_size = img_size

    def gaussian_blur(self, x):
        # Simple 3x3 average blur, depthwise per channel
        kernel = tf.ones((3, 3, 3, 1), dtype=x.dtype) / 9.0
        return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")

    def call(self, x, training=False):
        if not training:
            return x

        x = tf.cast(x, tf.float32)

        # Random Gaussian noise
        if tf.random.uniform([]) < 0.50:
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.025)
            x = x + noise

        # Random brightness/contrast
        if tf.random.uniform([]) < 0.35:
            x = tf.image.random_brightness(x, max_delta=0.05)
            x = tf.image.random_contrast(x, lower=0.9, upper=1.1)

        # Random blur
        if tf.random.uniform([]) < 0.35:
            x = self.gaussian_blur(x)

        # Random crop and resize
        if tf.random.uniform([]) < 0.50:
            crop_ratio = tf.random.uniform([], 0.82, 1.0)
            crop_size = tf.cast(tf.cast(self.img_size, tf.float32) * crop_ratio, tf.int32)
            x = tf.image.resize_with_crop_or_pad(x, crop_size, crop_size)
            x = tf.image.resize(x, [self.img_size, self.img_size])

        x = tf.clip_by_value(x, 0.0, 1.0)
        return x
    
# ============================================================
# WATERMARK EMBEDDING MODEL
# ============================================================

def build_watermark_encoder(img_size=128, wm_bits=32, dim=128):
    image_input = layers.Input(shape=(img_size, img_size, 3), name="original_image")
    wm_input = layers.Input(shape=(wm_bits,), name="watermark_bits")

    # Turn watermark vector into spatial watermark map
    wm = layers.Dense((img_size // 4) * (img_size // 4) * 16, activation="gelu")(wm_input)
    wm = layers.Reshape((img_size // 4, img_size // 4, 16))(wm)

    # Image feature encoder
    x = layers.Conv2D(32, 3, padding="same", activation="gelu")(image_input)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="gelu")(x)
    x = layers.Conv2D(96, 3, strides=2, padding="same", activation="gelu")(x)

    # Combine image features and watermark features
    x = layers.Concatenate()([x, wm])
    x = layers.Conv2D(dim, 1, padding="same", activation="gelu")(x)

    # Convert feature map to tokens
    h = img_size // 4
    w = img_size // 4
    x_tokens = layers.Reshape((h * w, dim))(x)

    # Lightweight Transformer blocks
    x_tokens = TransformerBlock(dim, num_heads=4)(x_tokens)
    x_tokens = TransformerBlock(dim, num_heads=4)(x_tokens)

    x = layers.Reshape((h, w, dim))(x_tokens)

    # Decode residual watermark signal
    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="gelu")(x)

    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="gelu")(x)

    residual = layers.Conv2D(3, 3, padding="same", activation="tanh", dtype="float32")(x)
    residual = layers.Lambda(lambda r: r * MAX_RESIDUAL_STRENGTH, name="watermark_residual")(residual)

    watermarked = layers.Add(dtype="float32")([image_input, residual])
    watermarked = layers.Lambda(lambda z: tf.clip_by_value(z, 0.0, 1.0), name="watermarked_image")(watermarked)

    return keras.Model([image_input, wm_input], [watermarked, residual], name="Transformer_Watermark_Encoder")


encoder = build_watermark_encoder(IMG_SIZE, WM_BITS)
encoder.summary()

# ============================================================
# WATERMARK EXTRACTION MODEL
# ============================================================

def build_watermark_decoder(img_size=128, wm_bits=32, dim=128):
    image_input = layers.Input(shape=(img_size, img_size, 3), name="attacked_watermarked_image")

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="gelu")(image_input)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="gelu")(x)
    x = layers.Conv2D(dim, 3, strides=2, padding="same", activation="gelu")(x)

    h = img_size // 8
    w = img_size // 8

    x_tokens = layers.Reshape((h * w, dim))(x)

    x_tokens = TransformerBlock(dim, num_heads=4)(x_tokens)
    x_tokens = TransformerBlock(dim, num_heads=4)(x_tokens)

    x = layers.LayerNormalization(epsilon=1e-6)(x_tokens)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.2)(x)

    # Logits, not sigmoid, because loss uses from_logits=True
    watermark_logits = layers.Dense(wm_bits, dtype="float32", name="watermark_logits")(x)

    return keras.Model(image_input, watermark_logits, name="Transformer_Watermark_Decoder")


decoder = build_watermark_decoder(IMG_SIZE, WM_BITS)
decoder.summary()

# ============================================================
# TRAINER MODEL
# ============================================================

class WatermarkTrainer(keras.Model):
    def __init__(self, encoder, decoder, attack_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attack_layer = attack_layer

        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae = keras.losses.MeanAbsoluteError()

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.image_loss_tracker = keras.metrics.Mean(name="image_loss")
        self.wm_loss_tracker = keras.metrics.Mean(name="watermark_loss")
        self.ber_tracker = keras.metrics.Mean(name="bit_error_rate")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.image_loss_tracker,
            self.wm_loss_tracker,
            self.ber_tracker
        ]

    def train_step(self, images):
        if isinstance(images, tuple):
            images = images[0]

        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]

        # Random binary copyright watermark
        watermark = tf.cast(
            tf.random.uniform((batch_size, WM_BITS), minval=0, maxval=2, dtype=tf.int32),
            tf.float32
        )

        with tf.GradientTape() as tape:
            watermarked, residual = self.encoder([images, watermark], training=True)
            attacked = self.attack_layer(watermarked, training=True)
            logits = self.decoder(attacked, training=True)

            image_loss = self.mae(images, watermarked)
            wm_loss = self.bce(watermark, logits)
            residual_loss = tf.reduce_mean(tf.abs(residual))

            total_loss = (
                IMAGE_LOSS_WEIGHT * image_loss +
                WATERMARK_LOSS_WEIGHT * wm_loss +
                RESIDUAL_LOSS_WEIGHT * residual_loss
            )

        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        pred_bits = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
        ber = tf.reduce_mean(tf.cast(tf.not_equal(pred_bits, watermark), tf.float32))

        self.total_loss_tracker.update_state(total_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.wm_loss_tracker.update_state(wm_loss)
        self.ber_tracker.update_state(ber)

        return {
            "loss": self.total_loss_tracker.result(),
            "image_loss": self.image_loss_tracker.result(),
            "watermark_loss": self.wm_loss_tracker.result(),
            "bit_error_rate": self.ber_tracker.result()
        }

    def test_step(self, images):
        if isinstance(images, tuple):
            images = images[0]

        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]

        watermark = tf.cast(
            tf.random.uniform((batch_size, WM_BITS), minval=0, maxval=2, dtype=tf.int32),
            tf.float32
        )

        watermarked, residual = self.encoder([images, watermark], training=False)
        attacked = self.attack_layer(watermarked, training=False)
        logits = self.decoder(attacked, training=False)

        image_loss = self.mae(images, watermarked)
        wm_loss = self.bce(watermark, logits)
        residual_loss = tf.reduce_mean(tf.abs(residual))

        total_loss = (
            IMAGE_LOSS_WEIGHT * image_loss +
            WATERMARK_LOSS_WEIGHT * wm_loss +
            RESIDUAL_LOSS_WEIGHT * residual_loss
        )

        pred_bits = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
        ber = tf.reduce_mean(tf.cast(tf.not_equal(pred_bits, watermark), tf.float32))

        self.total_loss_tracker.update_state(total_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.wm_loss_tracker.update_state(wm_loss)
        self.ber_tracker.update_state(ber)

        return {
            "loss": self.total_loss_tracker.result(),
            "image_loss": self.image_loss_tracker.result(),
            "watermark_loss": self.wm_loss_tracker.result(),
            "bit_error_rate": self.ber_tracker.result()
        }


attack_layer = AttackLayer(IMG_SIZE)

trainer = WatermarkTrainer(
    encoder=encoder,
    decoder=decoder,
    attack_layer=attack_layer
)

optimizer = keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=1e-5
)

trainer.compile(optimizer=optimizer)

# ============================================================
# TRAINING
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_watermark_model.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_bit_error_rate",
        mode="min",
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_bit_error_rate",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_bit_error_rate",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
]

history = trainer.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# TRAINING CURVES
# ============================================================

def plot_curve(metric):
    plt.figure(figsize=(7, 4))
    plt.plot(history.history[metric], label=f"train_{metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_curve("loss")
plot_curve("bit_error_rate")
plot_curve("image_loss")
plot_curve("watermark_loss")

# ============================================================
# VISUAL DEMO
# ============================================================

def demo_visualization(dataset):
    images = next(iter(dataset))
    images = tf.cast(images[:4], tf.float32)
    batch_size = tf.shape(images)[0]

    watermark = tf.cast(
        tf.random.uniform((batch_size, WM_BITS), minval=0, maxval=2, dtype=tf.int32),
        tf.float32
    )

    watermarked, residual = encoder([images, watermark], training=False)
    attacked = attack_layer(watermarked, training=True)
    logits = decoder(attacked, training=False)

    pred_bits = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
    ber = tf.reduce_mean(tf.cast(tf.not_equal(pred_bits, watermark), tf.float32)).numpy()

    original = images.numpy()
    wm_img = watermarked.numpy()
    attacked_img = attacked.numpy()
    diff = np.abs(wm_img - original) * 10.0
    diff = np.clip(diff, 0, 1)

    print("Demo Bit Error Rate:", ber)
    print("Example original watermark bits: ", watermark[0].numpy().astype(int))
    print("Example extracted watermark bits:", pred_bits[0].numpy().astype(int))

    for i in range(min(4, batch_size)):
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 4, 1)
        plt.imshow(original[i])
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(wm_img[i])
        plt.title("Watermarked")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(diff[i])
        plt.title("Difference x10")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(attacked_img[i])
        plt.title("Attacked")
        plt.axis("off")

        plt.show()

demo_visualization(val_ds)

# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_model(dataset, num_batches=20):
    psnr_values = []
    ssim_values = []
    ber_values = []
    acc_values = []

    for batch_idx, images in enumerate(dataset):
        if batch_idx >= num_batches:
            break

        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]

        watermark = tf.cast(
            tf.random.uniform((batch_size, WM_BITS), minval=0, maxval=2, dtype=tf.int32),
            tf.float32
        )

        watermarked, residual = encoder([images, watermark], training=False)
        attacked = attack_layer(watermarked, training=True)
        logits = decoder(attacked, training=False)

        pred_bits = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)

        psnr = tf.reduce_mean(tf.image.psnr(images, watermarked, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(images, watermarked, max_val=1.0))
        ber = tf.reduce_mean(tf.cast(tf.not_equal(pred_bits, watermark), tf.float32))
        acc = 1.0 - ber

        psnr_values.append(psnr.numpy())
        ssim_values.append(ssim.numpy())
        ber_values.append(ber.numpy())
        acc_values.append(acc.numpy())

    results = {
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "BER": float(np.mean(ber_values)),
        "Extraction Accuracy": float(np.mean(acc_values))
    }

    return results


results = evaluate_model(val_ds, num_batches=20)

print("Evaluation Results")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

    # ============================================================
# SPECIFIC ATTACK TESTS
# ============================================================

def apply_specific_attack(x, attack_name):
    x = tf.cast(x, tf.float32)

    if attack_name == "none":
        return x

    elif attack_name == "gaussian_noise":
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.03)
        return tf.clip_by_value(x + noise, 0.0, 1.0)

    elif attack_name == "blur":
        kernel = tf.ones((3, 3, 3, 1), dtype=tf.float32) / 9.0
        return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")

    elif attack_name == "resize":
        small = tf.image.resize(x, [IMG_SIZE // 2, IMG_SIZE // 2])
        return tf.image.resize(small, [IMG_SIZE, IMG_SIZE])

    elif attack_name == "crop":
        crop_size = int(IMG_SIZE * 0.85)
        cropped = tf.image.resize_with_crop_or_pad(x, crop_size, crop_size)
        return tf.image.resize(cropped, [IMG_SIZE, IMG_SIZE])

    elif attack_name == "brightness":
        return tf.clip_by_value(tf.image.adjust_brightness(x, 0.08), 0.0, 1.0)

    else:
        raise ValueError("Unknown attack name")


def evaluate_attack(dataset, attack_name, num_batches=20):
    psnr_values = []
    ssim_values = []
    ber_values = []
    acc_values = []

    for batch_idx, images in enumerate(dataset):
        if batch_idx >= num_batches:
            break

        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]

        watermark = tf.cast(
            tf.random.uniform((batch_size, WM_BITS), minval=0, maxval=2, dtype=tf.int32),
            tf.float32
        )

        watermarked, residual = encoder([images, watermark], training=False)
        attacked = apply_specific_attack(watermarked, attack_name)
        logits = decoder(attacked, training=False)

        pred_bits = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)

        psnr = tf.reduce_mean(tf.image.psnr(images, watermarked, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(images, watermarked, max_val=1.0))
        ber = tf.reduce_mean(tf.cast(tf.not_equal(pred_bits, watermark), tf.float32))
        acc = 1.0 - ber

        psnr_values.append(psnr.numpy())
        ssim_values.append(ssim.numpy())
        ber_values.append(ber.numpy())
        acc_values.append(acc.numpy())

    return {
        "Attack": attack_name,
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "BER": float(np.mean(ber_values)),
        "Extraction Accuracy": float(np.mean(acc_values))
    }


attacks = ["none", "gaussian_noise", "blur", "resize", "crop", "brightness"]

attack_results = []
for attack in attacks:
    result = evaluate_attack(val_ds, attack, num_batches=20)
    attack_results.append(result)

for r in attack_results:
    print(r)

    # ============================================================
# REPORT TABLE
# ============================================================

import pandas as pd

results_df = pd.DataFrame(attack_results)
results_df

# ============================================================
# SAVE MODELS
# ============================================================

encoder.save("transformer_watermark_encoder.keras")
decoder.save("transformer_watermark_decoder.keras")

print("Saved encoder and decoder.")