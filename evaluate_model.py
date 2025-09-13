import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model and prepare test data
model = load_model("models/pneumonia_model.h5", compile=False)

img_height, img_width = 224, 224
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./127.5 - 1)
test_generator = test_datagen.flow_from_directory(
    "dataset/chest_xray/test",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",  # keep categorical if you trained model this way
    shuffle=False
)

# Make predictions
y_true = test_generator.classes
y_pred_proba = model.predict(test_generator)  # shape = (num_samples, 2)

# Extract probability for Pneumonia class (assuming index 1 = Pneumonia)
pneumonia_prob = y_pred_proba[:, 1]

# Try different thresholds
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (pneumonia_prob > threshold).astype(int)  # binary prediction

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== Threshold: {threshold} ===")
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

    # Optional: plot confusion matrix for this threshold
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(test_generator.class_indices.keys()),
                yticklabels=list(test_generator.class_indices.keys()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Threshold={threshold})")
    plt.show()

