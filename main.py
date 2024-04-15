import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD

# MNIST veri setini yükleyip ön işleme
def load_data():
    # MNIST veri setini yükleyin
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalize et ve şeklini değiştir (örnek, 28x28 pikselden düz vektöre)
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    # Etiketleri kategorik formata dönüştür
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test

# Perceptron modelini oluşturma
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', input_shape=(input_shape,)))
    return model

# Modeli eğit
def train_model(model, X_train, y_train):
    model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)

# Görüntüyü model için uygun formata dönüştürme
def process_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')  # Gri tonlamaya çevir
    image = np.array(image)
    image = image.reshape(1, -1) / 255.0  # Normalize et ve düzleştir
    return image

# Streamlit uygulaması
def main():
    st.title("Digit Recognition with Single-Layer Perceptron")
    
    # Modeli ve verileri yükleyin
    X_train, y_train, X_test, y_test = load_data()
    model = build_model(X_train.shape[1], y_train.shape[1])
    
    if st.button('Train the Model'):
        # Modeli eğit
        train_model(model, X_train, y_train)
        st.success('Model trained successfully!')
    
    # Kullanıcıdan bir görüntü yüklemesini isteyin
    uploaded_file = st.file_uploader("Upload a digit image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        processed_image = process_image(image)
        
        # Tahmini yap ve göster
        if st.button('Recognize Digit'):
            prediction = model.predict(processed_image)
            st.write(f"Predicted digit: {np.argmax(prediction)}")
    
    # Opsiyonel: model performansını değerlendir
    if st.checkbox('Evaluate Model'):
        loss, accuracy = model.evaluate(X_test, y_test)
        st.write(f"Test accuracy: {accuracy}")

if __name__ == "__main__":
    main()

