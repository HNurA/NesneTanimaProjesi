import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data():
    # Veri setini yükle
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Pre-processing: Normalizasyon (0-1 arasına çekme)
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def get_augmented_data_generator(train_images, train_labels):
    # Data Augmentation (Veri Artırma) - Pre-processing'in kralı
    # Resimleri hafifçe döndürür, kaydırır, yatay çevirir. 
    # Bu sayede model ezberlemez, gerçekten öğrenir.
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(train_images)
    return datagen