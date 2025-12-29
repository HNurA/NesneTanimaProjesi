import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

def plot_history(history):
    # Eğitim başarısını grafiğe dök
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
    plt.plot(history.history['val_accuracy'], label='Test Başarısı')
    plt.title('Model Başarısı')
    plt.legend()
    plt.show()

def print_evaluation(model, test_images, test_labels, class_names):
    # Detaylı rapor
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_labels.flatten()
    
    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Karmaşıklık Matrisi (Hangi sınıfı hangisiyle karıştırıyor?)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Karmaşıklık Matrisi (Confusion Matrix)")
    plt.show()