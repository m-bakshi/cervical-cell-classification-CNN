import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input  
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# paths
MODEL_PATH = r"F:\Cervival_Cell_CNN\best_resnet50_sipakmed.h5"
TEST_DIR   = r"F:\Cervival_Cell_CNN\SIPakMed_split_dataset\test"

IMG_SIZE   = 224
BATCH_SIZE = 16

# test generator - same preprocessing as training
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input   # NOT rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False          # must be False for correct label alignment
)

class_names = list(test_generator.class_indices.keys())
print(f"[INFO] Classes: {class_names}")
print(f"[INFO] Test samples: {test_generator.samples}")

# load model
print("\n[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# evaluate
print("\n[INFO] Evaluating on test set...")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n  Test Loss    : {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")

# predictions
print("\n[INFO] Generating predictions...")
predictions = model.predict(test_generator, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes


# classification report

print("\n" + "="*60)
print("  CLASSIFICATION REPORT")
print("="*60)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)


# confusion matrix

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label",      fontsize=12)
ax.set_title(f"Confusion Matrix  (Test Accuracy: {accuracy*100:.2f}%)", fontsize=13)
plt.tight_layout()
plt.savefig(r"F:\Cervival_Cell_CNN\confusion_matrix.png", dpi=150)
plt.show()
print("\n[INFO] Confusion matrix saved.")


# training history plot
def plot_history(history_dict, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_dict['accuracy'],     label='Train Acc')
    axes[0].plot(history_dict['val_accuracy'], label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')

    axes[1].plot(history_dict['loss'],     label='Train Loss')
    axes[1].plot(history_dict['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


print("\n[DONE] Evaluation complete.")
