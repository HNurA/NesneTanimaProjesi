from tensorflow.keras import layers, models

def create_cnn_model():
    model = models.Sequential()
    
    # Blok 1: Özellik Çıkarımı
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization()) # Öğrenmeyi hızlandırır
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2)) # Ezberlemeyi (Overfitting) önler - Fine Tuning
    
    # Blok 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Blok 3: Sınıflandırma
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(10)) # 10 Sınıf için çıkış

    return model