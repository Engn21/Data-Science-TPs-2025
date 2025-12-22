
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#download the dataset
# MNIST (70.000 samples) -> 784 features (28x28 pixels)
# If we say 'return_X_y=True', images (X) and labels (y) are returned separately
images, labels = fetch_openml(
    'mnist_784', return_X_y=True, as_frame=False, parser='auto'
)

# Seperate the dataset into training and test sets
# random_stane=42 if it is stable, everyone's data will be split the same way
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, random_state=42
)

# Helper filter function 

def select_with_label(images, labels, desired_labels):
    """
    desired_labels: ['0','1','2','3','4'] gibi string liste
    """
    mask = np.isin(labels, desired_labels)
    return images[mask], labels[mask]

# Select only digits 0-4 from the dataset
desired = ['0', '1', '2', '3', '4']

train_images_filtered, train_labels_filtered = select_with_label(
    train_images, train_labels, desired_labels=desired
)
test_images_filtered, test_labels_filtered = select_with_label(
    test_images, test_labels, desired_labels=desired
)

#Selects only 1000 samples per class from the filtered training set to ensure balance.
SEED = 20251020  # random seed for reproducibility
rng = np.random.default_rng(SEED)

X_parts, y_parts = [], []
for label in desired:
    idx = np.where(train_labels_filtered == label)[0]
    chosen = rng.choice(idx, size=1000, replace=False)
    X_parts.append(train_images_filtered[chosen])
    y_parts.append(train_labels_filtered[chosen])

# 5000 samples (1000 per class)
X_train = np.vstack(X_parts)
y_train = np.hstack(y_parts)

# Shuffle the training data to mix classes
perm = rng.permutation(len(y_train))
X_train, y_train = X_train[perm], y_train[perm]

# Test set remains as is (all samples of classes 0-4)
X_test, y_test = test_images_filtered, test_labels_filtered

# Normalization (0–255 → 0–1)

X_train = X_train / 255.0
X_test  = X_test  / 255.0

# 1-NN classifier implementation and evaluation
knn1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn1.fit(X_train, y_train)

# Over test data we should do predictions 
y_pred = knn1.predict(X_test)

# Accuracy and detailed results
acc = accuracy_score(y_test, y_pred) * 100
print(f"\nRandom seed used for sampling: {SEED}")
print(f"Baseline 1-NN test accuracy: {acc:.2f}%")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
