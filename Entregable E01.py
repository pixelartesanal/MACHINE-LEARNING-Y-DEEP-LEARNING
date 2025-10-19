import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
import os

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn  = np.expand_dims(x_test, -1)
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat  = keras.utils.to_categorical(y_test, num_classes)
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat  = x_test.reshape(len(x_test), -1)

pca = PCA(n_components=100)
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca  = pca.transform(x_test_flat)

def build_cnn(input_shape=(28,28,1), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

cnn = build_cnn()
cnn.summary()

callbacks = [ keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) ]
history = cnn.fit(x_train_cnn, y_train_cat, validation_split=0.1, epochs=20, batch_size=64, callbacks=callbacks, verbose=2)
cnn.save("mnist_cnn.h5")

test_loss, test_acc = cnn.evaluate(x_test_cnn, y_test_cat, verbose=0)
y_pred_cnn = np.argmax(cnn.predict(x_test_cnn), axis=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_pca, y_train)
y_pred_knn = knn.predict(x_test_pca)
acc_knn = accuracy_score(y_test, y_pred_knn)

svm = SVC(kernel='rbf', C=5, gamma='scale')
svm.fit(x_train_pca, y_train)
y_pred_svm = svm.predict(x_test_pca)
acc_svm = accuracy_score(y_test, y_pred_svm)

logreg = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
logreg.fit(x_train_pca, y_train)
y_pred_log = logreg.predict(x_test_pca)
acc_log = accuracy_score(y_test, y_pred_log)

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(x_train_pca)
clusters = kmeans.predict(x_test_pca)

cm_cnn = confusion_matrix(y_test, y_pred_cnn)
print(f"\nCNN Test accuracy: {test_acc:.4f}\n")
print("Classification report - CNN:\n", classification_report(y_test, y_pred_cnn, digits=4))
print(f"KNN (PCA 100 comps) accuracy: {acc_knn:.4f}")
print(f"SVM (PCA 100 comps) accuracy: {acc_svm:.4f}")
print(f"Logistic Regression (PCA 100 comps) accuracy: {acc_log:.4f}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('CNN - Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('CNN - Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de confusión - CNN')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()

print("\nResumen de accuracies:")
print(f"CNN:  {test_acc:.4f}")
print(f"KNN:  {acc_knn:.4f}")
print(f"SVM:  {acc_svm:.4f}")
print(f"LogR: {acc_log:.4f}")

def mostrar_ejemplos(x, y_true, y_pred, correct=True, n=9):
    idxs = np.where((y_true == y_pred) == correct)[0]
    if len(idxs) < n:
        n = len(idxs)
    idxs = np.random.choice(idxs, size=n, replace=False)
    plt.figure(figsize=(8,8))
    for i, idx in enumerate(idxs):
        plt.subplot(3,3,i+1)
        plt.imshow(x[idx].reshape(28,28), cmap='gray')
        color = 'green' if correct else 'red'
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

mostrar_ejemplos(x_test, y_test, y_pred_cnn, correct=True, n=9)
mostrar_ejemplos(x_test, y_test, y_pred_cnn, correct=False, n=9)

joblib.dump(knn, "knn_mnist_pca100.joblib")
joblib.dump(svm, "svm_mnist_pca100.joblib")
joblib.dump(logreg, "logreg_mnist_pca100.joblib")
joblib.dump(pca, "pca_100_components.joblib")
cnn.save("mnist_cnn_model.h5")