import streamlit as st
import numpy as np
from PIL import Image

# Veri Oluşturma
def generate_data():
    data = []
    labels = []
    for i in range(10):  # 0-9 arasındaki rakamlar için döngü
        for _ in range(100):  # Her rakam için 100 örnek oluştur
            # Örnek veri oluştur
            image = np.random.randint(0, 2, size=(5, 5))  # Örnek bir 5x5 görüntü
            data.append(image.flatten())  # Görüntüyü düzleştirip veriye ekle
            labels.append(i)  # Etiketi ekle
    return np.array(data), np.array(labels)

# Perceptron Modeli
class Perceptron:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features + 1)  # Bias dahil ağırlıklar
        self.learning_rate = 0.1

    def predict(self, x):
        return 1 if np.dot(self.weights[1:], x) + self.weights[0] > 0 else 0

    def train(self, X, y, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                x = np.insert(X[i], 0, 1)  # Bias terimini ekleyin
                y_pred = self.predict(x)
                error = y[i] - y_pred
                self.weights[1:] += self.learning_rate * error * x[1:]  # Ağırlıkları güncelle
                self.weights[0] += self.learning_rate * error  # Bias ağırlığını güncelle

# Streamlit Uygulaması
def main():
    st.title("Sayı Tanıma Uygulaması")

    # Veri setini oluştur
    X, y = generate_data()

    # Perceptron modelini eğit
    perceptron = Perceptron(num_features=X.shape[1])
    perceptron.train(X, y, epochs=100)

    uploaded_image = st.file_uploader("Bir sayı yükleyin", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Yüklenen görüntüyü oku
        image = np.array(uploaded_image)
        st.image(image, caption='Yüklenen Görüntü', use_column_width=True)

        # Görüntüyü 5x5 boyutuna yeniden boyutlandır
        resized_image = np.array(Image.open(uploaded_image).resize((5, 5)))
        
        # Görüntüyü düzleştir
        flattened_image = resized_image.flatten()

        # Modeli kullanarak sayıyı tahmin et
        prediction = perceptron.predict(np.insert(flattened_image, 0, 1))

        st.write("Tahmin Edilen Sayı:", prediction)

if __name__ == "__main__":
    main()

