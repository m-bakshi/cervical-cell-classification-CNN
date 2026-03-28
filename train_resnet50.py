import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight


# paths

TRAIN_DIR = r"F:\Cervival_Cell_CNN\SIPakMed_augmented\train"
VAL_DIR   = r"F:\Cervival_Cell_CNN\SIPakMed_split_dataset\val"
MODEL_OUT  = r"F:\Cervival_Cell_CNN\best_resnet50_sipakmed.h5"

# hyperparameters 
IMG_SIZE   = 224
BATCH_SIZE = 16          
NUM_CLASSES = 5

PHASE1_EPOCHS = 10       # warm-up: train head only
PHASE2_EPOCHS = 60       # fine-tune deeper layers

LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4

# data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input   # same as train
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"\n[INFO] Class indices: {train_generator.class_indices}")
print(f"[INFO] Training samples : {train_generator.samples}")
print(f"[INFO] Validation samples: {val_generator.samples}")

# class weights  (handles class imbalance)
class_labels = train_generator.classes
unique_classes = np.unique(class_labels)

weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=class_labels
)
class_weight_dict = dict(zip(unique_classes, weights))
print(f"\n[INFO] Class weights: {class_weight_dict}")

# model
def build_model(trainable_from_layer=None):
    """
    Builds ResNet50 + custom classifier head.
    trainable_from_layer: freeze layers BEFORE this index.
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # freeze all layers initially
    base_model.trainable = False

    if trainable_from_layer is not None:
        for layer in base_model.layers[trainable_from_layer:]:
            # Skip BatchNorm layers — keep them frozen (imp for ResNet)
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    # classifier head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# PHASE 1: train classifier head only  (base frozen)

print("\n" + "="*60)
print("  PHASE 1: Training classifier head (base frozen)")
print("="*60)

model = build_model(trainable_from_layer=None)  # all base frozen

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks_phase1 = [
    ModelCheckpoint(      # saves best weights
        MODEL_OUT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(    # reduces LR if validation loss plateaus
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(       # stops training if val accuracy stops improving
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=PHASE1_EPOCHS,
    callbacks=callbacks_phase1,
    class_weight=class_weight_dict
)

print(f"\n[Phase 1 done] Best val_accuracy: {max(history1.history['val_accuracy']):.4f}")


# PHASE 2: fine-tune last ResNet block + classifier head

print("\n" + "="*60)
print("  PHASE 2: Fine-tuning last conv block (layer 143+)")
print("="*60)

# reload best weights from phase 1, then unfreeze deeper layers
model = build_model(trainable_from_layer=143)

# load phase-1 best weights into the new model structure
# weights are compatible since architecture unchanged
model.load_weights(MODEL_OUT)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE2),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"[INFO] Trainable layers in Phase 2: {trainable_count}")

callbacks_phase2 = [
    ModelCheckpoint(
        MODEL_OUT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
]

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=PHASE2_EPOCHS,
    callbacks=callbacks_phase2,
    class_weight=class_weight_dict
)

print(f"\n[Phase 2 done] Best val_accuracy: {max(history2.history['val_accuracy']):.4f}")
print(f"[INFO] Best model saved to: {MODEL_OUT}")