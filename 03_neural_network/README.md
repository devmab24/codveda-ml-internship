## 📊 Results Summary

### Week 3: Neural Network on MNIST

* **Model:** Feed-Forward Neural Network (Multi-Layer Perceptron, 2 hidden layers).
* **Dataset:** MNIST (70,000 grayscale images of handwritten digits, 28×28).
* **Preprocessing:** Pixel values normalized (0–1), labels one-hot encoded.

#### 🔹 Performance

* **Training Accuracy:** ~99%
* **Validation Accuracy:** ~98%
* **Test Accuracy:** ~98%

#### 🔹 Key Insights

* The model quickly converged within 10 epochs, showing both strong training and validation accuracy.
* Loss and accuracy curves confirmed **stable learning without severe overfitting**.
* The **confusion matrix** showed very few misclassifications, mostly between similar digits (e.g., 4 vs 9).
* The **classification report** confirmed balanced performance across all digit classes.

#### 🔹 Takeaway

This project demonstrates how a relatively simple **Feed-Forward Neural Network** can effectively handle image classification tasks on MNIST, achieving **near state-of-the-art accuracy** without advanced architectures. It lays a strong foundation for exploring more complex models like **Convolutional Neural Networks (CNNs)** in future tasks.