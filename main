import streamlit as st
import numpy as np

# Perceptron sınıfını tanımlama
class Perceptron:
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# Uygulama başlangıcı
def main():
    st.title('Simple Perceptron Model')
    st.write("This is a simple demonstration of a perceptron model for binary classification.")

    # Perceptron modelinin oluşturulması ve eğitimi
    perceptron = Perceptron(no_of_inputs=16, threshold=100, learning_rate=0.1)
    training_inputs = [np.array([0]*16), np.array([1]*16)]
    labels = [0, 1]
    perceptron.train(training_inputs, labels)

    # Kullanıcı girişi
    input_vector = st.text_input("Enter your binary input vector (comma-separated):")
    if input_vector:
        input_vector = np.array([int(x.strip()) for x in input_vector.split(",")])
        prediction = perceptron.predict(input_vector)
        st.write(f"Prediction: {'Class 1' if prediction == 1 else 'Class 0'}")

if __name__ == '__main__':
    main()
