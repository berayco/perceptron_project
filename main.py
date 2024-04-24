import numpy as np
import streamlit as st
from PIL import Image

# 0'dan 9'a kadar olan sayıları temsil eden matrisler
digits = [
    np.array([[0, 1, 1, 1, 0],   # 0
              [1, 0, 0, 0, 1],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 0, 1],
              [0, 1, 1, 1, 0]]),

    np.array([[0, 0, 1, 0, 0],   # 1
              [0, 1, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 1, 1, 0]]),

    np.array([[0, 1, 1, 1, 0],   # 2
              [0, 0, 0, 1, 0],
              [0, 1, 1, 1, 0],
              [1, 0, 0, 0, 0],
              [0, 1, 1, 1, 0]]),

    np.array([[0, 1, 1, 1, 0],   # 3
              [0, 0, 0, 1, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 0, 1, 0],
              [0, 1, 1, 1, 0]]),

    np.array([[1, 0, 0, 1, 0],   # 4
              [1, 0, 0, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]),

    np.array([[1, 1, 1, 1, 0],   # 5
              [1, 0, 0, 0, 0],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 1, 0],
              [1, 1, 1, 1, 0]]),

    np.array([[0, 1, 1, 1, 0],   # 6
              [1, 0, 0, 0, 0],
              [1, 1, 1, 1, 0],
              [1, 0, 0, 1, 0],
              [0, 1, 1, 1, 0]]),

    np.array([[1, 1, 1, 1, 1],   # 7
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0]]),

    np.array([[0, 1, 1, 1, 0],   # 8
              [1, 0, 0, 1, 0],
              [0, 1, 1, 1, 0],
              [1, 0, 0, 1, 0],
              [0, 1, 1, 1, 0]]),

    np.array([[0, 1, 1, 1, 0],   # 9
              [1, 0, 0, 1, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 0, 1, 0],
              [0, 1, 1, 1, 0]])
]

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
    st.title("Digit Recognition")

    # Perceptron modelini eğit
    X = np.array(digits).reshape((10, 25))  # 10 örnek, her biri 25 özellik
    y = np.arange(10)  # Etiketler
    perceptron = Perceptron(num_features=25)  # Giriş boyutu 5x5=25
    perceptron.train(X, y, epochs=100)

    uploaded_image = st.file_uploader("Bir sayı yükleyin", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Yüklenen görüntüyü oku ve siyah-beyaz olarak dönüştür
        image = Image.open(uploaded_image).convert("L")
        st.image(image, caption='Yüklenen Görüntü', use_column_width=True)

        # Görüntüyü 5x5 boyutuna yeniden boyutlandır
        resized_image = image.resize((5, 5))
        
        # Görüntüyü numpy dizisine dönüştür
        img_array = np.array(resized_image)
        
        # Görüntüyü düzleştir
        flattened_image = img_array.flatten()
        
        # Modeli kullanarak sayıyı tahmin et
        prediction = perceptron.predict(np.insert(flattened_image, 0, 1))

        st.write("Tahmin Edilen Sayı:", prediction)

if __name__ == "__main__":
    main()


