import numpy as np
import streamlit as st
from PIL import Image
from sklearn.linear_model import Perceptron

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

# Verileri hazırla
X = np.array(digits).reshape((10, 25))  # 10 örnek, her biri 25 özellik
y = np.arange(10)  # Etiketler

# Modeli oluştur ve eğit
perceptron = Perceptron(max_iter=100)
perceptron.fit(X, y)

# Streamlit Uygulaması
def main():
    st.title("Digit Recognition")

    uploaded_image = st.file_uploader("Bir sayı yükleyin", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Yüklenen görüntüyü açın ve boyutunu 5x5'e yeniden boyutlandırın
        image = Image.open(uploaded_image).convert("L").resize((5, 5))
        st.image(image, caption='Yüklenen Görüntü', use_column_width=True)
        
        # Görüntüyü numpy dizisine dönüştür
        img_array = np.array(image)
        
        # Görüntüyü düzleştir
        flattened_image = img_array.flatten()
        
        # Modeli kullanarak sayıyı tahmin et
        prediction = perceptron.predict([flattened_image])

        st.write("Tahmin Edilen Sayı:", prediction[0])

if __name__ == "__main__":
    main()

