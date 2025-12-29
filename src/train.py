import tensorflow as tf
from data_loader import get_data, get_augmented_data_generator
from model_builder import create_cnn_model
from evaluate import plot_history, print_evaluation
import os

# Sınıf İsimleri
class_names = ['Uçak', 'Otomobil', 'Kuş', 'Kedi', 'Geyik', 'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

def main():
    print("Veri yükleniyor...")
    (train_images, train_labels), (test_images, test_labels) = get_data()
    
    print("Model oluşturuluyor...")
    model = create_cnn_model()
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print("Eğitim Başlıyor (Data Augmentation ile)...")
    # Veri artırıcı kullanarak eğitim (Daha profesyonel)
    datagen = get_augmented_data_generator(train_images, train_labels)
    
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                        epochs=15, # Epoch sayısı
                        validation_data=(test_images, test_labels))
    
    # Modeli Kaydet
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/cifar10_model.h5')
    print("Model 'models/cifar10_model.h5' olarak kaydedildi.")
    
    # Değerlendirme
    plot_history(history)
    print_evaluation(model, test_images, test_labels, class_names)

if __name__ == "__main__":
    main()