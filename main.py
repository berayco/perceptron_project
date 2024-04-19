import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veri setini yükleyip hazırlama
@st.cache
def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test, digits.images

# Perceptron modelini eğitme ve değerlendirme
def train_perceptron(X_train, y_train):
    model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    return model

# Görüntüyü model için uygun formata dönüştürme
def process_image(image):
    image = image.resize((8, 8), Image.LANCZOS)
    image = image.convert('L')  # Gri tonlamaya çevir
    image = np.array(image, dtype=np.float64)
    image = image.reshape(1, -1) / 16.0  # Normalize et ve düzleştir
    return image

def main():
    st.title("Digit Recognition with Perceptron")
    X_train, y_train, X_test, y_test, example_images = load_data()

    # Modeli eğit
    if st.button('Train the Perceptron Model'):
        model = train_perceptron(X_train, y_train)
        st.session_state['model'] = model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Model trained. Accuracy on test set: {accuracy * 100:.2f}%")

    # Kullanıcıdan bir görüntü yüklemesini isteyin
    uploaded_file = st.file_uploader("Upload a digit image (preferably 8x8 size)...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        processed_image = process_image(image)

        # Tahmini yap ve göster
        if st.button('Recognize Digit'):
            if 'model' in st.session_state:
                model = st.session_state['model']
                prediction = model.predict(processed_image)
                st.write(f"Predicted digit: {prediction[0]}")
            else:
                st.error("Please train the model first by clicking on 'Train the Perceptron Model'.")

if __name__ == "__main__":
    main()

