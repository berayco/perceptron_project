import streamlit as st
import numpy as np

# Perceptron sınıfı
class Perceptron:
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# Streamlit uygulaması başlangıcı
def main():
    st.title('Simple Perceptron Model for Binary Classification')
    st.write("This app uses a simple perceptron to classify binary input vectors.")

    # Perceptron modelini oluşturma ve eğitim verisi ile eğitme
    perceptron = Perceptron(no_of_inputs=4)
    training_inputs = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 1]
    ])
    labels = np.array([0, 1, 1, 0])
    perceptron.train(training_inputs, labels)

    # Kullanıcı girişini alma ve tahminde bulunma
    user_input = st.text_input('Enter a binary vector (4 bits) separated by commas:')
    if user_input:
        # Girdiyi işle ve tahmin yap
        binary_vector = np.array([int(bit) for bit in user_input.split(',') if bit.strip().isdigit()])
        if len(binary_vector) == 4:
            prediction = perceptron.predict(binary_vector)
            st.success(f"The perceptron predicts this vector as: {'Class 1' if prediction == 1 else 'Class 0'}")
        else:
            st.error("Please enter a binary vector with exactly 4 bits.")

if __name__ == '__main__':
    main()
