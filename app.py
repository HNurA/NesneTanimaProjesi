import gradio as gr
import tensorflow as tf
import numpy as np

# Eğitilmiş modeli yükle
# Hata almamak için compile=False ekliyoruz çünkü sadece tahmin yapacağız, eğitmeyeceğiz.
model = tf.keras.models.load_model('models/cifar10_model.h5', compile=False)

class_names = ['Uçak', 'Otomobil', 'Kuş', 'Kedi', 'Geyik', 'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

def tahmin_et(img):
    if img is None:
        return None
        
    # 1. Gelen resmi TensorFlow formatına çevir
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    
    # 2. Resmi 32x32 boyutuna getir (Gradio'dan kaldırdığımız özellik buraya geldi)
    img = tf.image.resize(img, [32, 32])
    
    # 3. Modelin beklediği formata sok: (1, 32, 32, 3)
    img = tf.expand_dims(img, 0)
    
    # 4. Normalizasyon (0-255 arası değerleri 0-1 arasına çek)
    img = img / 255.0
    
    # 5. Tahmin yap
    prediction = model.predict(img).flatten()
    
    # 6. Sonuçları sözlüğe çevir
    confidences = {class_names[i]: float(tf.nn.softmax(prediction)[i]) for i in range(10)}
    return confidences

# Arayüz Ayarları (shape parametresi kaldırıldı)
interface = gr.Interface(
    fn=tahmin_et,
    inputs=gr.Image(label="Resim Yükle"), # shape parametresini sildik
    outputs=gr.Label(num_top_classes=3),
    title="Nesne Tanıma Projesi (CIFAR-10)",
    description="Örnek resimler yükleyin: Uçak, Araba, Kuş, Kedi, Geyik, Köpek, Kurbağa, At, Gemi, Kamyon."
)

if __name__ == "__main__":
    interface.launch(share=True)