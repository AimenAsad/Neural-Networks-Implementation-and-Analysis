This multi-part project aims to achieve a thorough understanding and practical implementation of:

* **Foundational ML Algorithms:** Implementing a single-layer **Perceptron from scratch** to understand basic neural network principles.
* **Core Deep Learning Operations:** Manually implementing **2D Convolution and Correlation** to grasp their mechanics and effects on images.
* **Convolutional Neural Networks (CNNs):** Building, training, and evaluating a CNN for image classification using the CIFAR-10 dataset, including data augmentation and feature map visualization.
* **Recurrent Neural Networks (RNNs):** Implementing a **Vanilla RNN** for next-word prediction on a Shakespeare dataset, focusing on custom embedding layers and evaluating perplexity.
* **Hyperparameter Optimization:** Applying **Random Search** for hyperparameter tuning on both CNN and RNN models to identify optimal configurations.

---
## ⚙️ Question 1: Implementing Rosenblatt’s Perceptron from Scratch

This section details the manual implementation of a single-layer Perceptron to understand its core mechanics.

### 1.1 Objective

To implement a **single-layer perceptron model from scratch**, ensuring a complete understanding of forward propagation, backward propagation, and weight updates. This includes generating a dataset, visualizing it, and evaluating the model.

### 1.2 Implementation Details

* **Data Generation and Visualization:**
    * Generated a synthetic dataset containing **500 samples with two features and a binary label**.
    * Visualized the dataset by plotting data points in a two-dimensional space.
    * Split the dataset into training (80%) and testing (20%) subsets.
* **Perceptron Core Functions (Manual Implementation):**
    * **Forward Pass:** Computes the weighted sum of inputs and applies a **step activation function**. Weights and biases are stored in matrices.
    * **Backward Pass:** Calculates the error and applies backpropagation to update weights using the **perceptron learning rule**.
    * The experiment was repeated over iterations, displaying the error for each iteration to observe learning progress.
* **Visualization and Evaluation:**
    * Plotted the **decision boundary** after training completion.
    * Visualized the test dataset by plotting all test data points on the decision boundary.
    * The accompanying report discusses how well the perceptron classifies the test data.

### Relevant Files:

* `Question1-Perceptron/perceptron_implementation.py` (or Jupyter Notebook)

---

## ⚙️ Question 2: Implementing Convolution from Scratch

This section focuses on manually implementing 2D convolution and correlation operations to understand their effect on images.

### 2.1 Objective

To develop a deeper understanding of convolution operations by implementing them manually without relying on built-in deep learning functions. This involves exploring how convolution operations affect images and how different kernels influence the output.

### 2.2 Implementation Details

* **Generalized Convolution Function:**
    * Implemented a manual **2D convolution function on a grayscale image**. All loops for the core convolution operation were designed manually.
    * **Function Parameters:**
        * **Input image:** The grayscale image to be processed.
        * **Kernel:** A user-defined kernel (defaults to a random kernel if none is provided).
        * **Kernel size:** The size of the kernel matrix.
        * **Stride:** The step size for sliding the kernel.
        * **Padding:** Option to use "valid" (no padding) or "same" (zero-padding to maintain size).
        * **Mode:** Option to perform either "convolution" or "correlation".
        * All parameters except "Input Image" are optional and set by default if not provided.
* **Convolution with Specific Kernels:**
    * Applied the convolution function to a grayscale image using different kernels for various purposes:
        * **Edge detection**
        * **Image blurring**
        * **Image sharpening**
    * Outputs of these different kernels were compared and analyzed for their effects on the image.
* **Compare Convolution vs Correlation:**
    * Tested two different kernels:
        * A **symmetric kernel** (e.g., `[[0.25, 0.25], [0.25, 0.25]]`).
        * A **non-symmetric kernel** (e.g., `[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]`).
    * Compared the output results of convolution and correlation operations on the same image and analyzed the differences.
* **Visualization and Analysis:**
    * Displayed the original image.
    * Showed the output images for each of the applied kernels.
    * **Side-by-side comparisons:**
        * Manually implemented convolution vs. NumPy-based convolution.
        * Convolution vs. Correlation for the chosen kernels.
    * A detailed description of experiments in the accompanying report includes analysis of:
        * The effect of different kernels on the image.
        * The impact of kernel size, stride, and padding on the output.
        * Observations from convolution vs. correlation results.
        * Identification of kernels useful for edge detection, blurring, and sharpening.
        * Explanation of the advantages of applying multiple kernels on the same image.

### Relevant Files:

* `Question2-Convolution-From-Scratch/convolution_implementation.py` (or Jupyter Notebook)
* `Question2-Convolution-From-Scratch/images/` (for sample images)

---

## ⚙️ Question 3: Implementing a CNN for CIFAR-10

This section covers building, training, and evaluating a Convolutional Neural Network for image classification using the CIFAR-10 dataset, leveraging deep learning frameworks.

### 3.1 Objective

To build a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10 dataset**. Students will explore CNN architecture, feature extraction, and model evaluation by implementing and training a CNN model using deep learning frameworks (TensorFlow/Keras or PyTorch).

### 3.2 Dataset

* **CIFAR-10 Dataset:** Loaded from [Hugging Face - CIFAR-10 Dataset](https://huggingface.co/datasets/cifar10).
    * Consists of 60,000 images (32x32 pixels, RGB, 10 classes).

### 3.3 Implementation Details

* **Dataset Preparation:**
    * Loaded the CIFAR-10 dataset.
    * **Preprocessing:** Normalized pixel values (scaled between 0 and 1) and converted labels into one-hot encoded format.
    * Split the dataset into training (80%) and testing (20%) subsets.
* **CNN Classifier Implementation:**
    * Built a CNN using a deep learning framework (TensorFlow/Keras or PyTorch).
    * **CNN Architecture includes:**
        * **Convolutional Layers:** For feature extraction.
        * **ReLU Activation Function:** To introduce non-linearity.
        * **Pooling Layers (Max/Average):** To reduce spatial dimensions.
        * **Fully Connected Layers:** To learn complex patterns for classification.
        * **Softmax Output Layer:** To generate class probabilities.
    * Trained the model using an appropriate optimizer (e.g., Adam, SGD).
* **Evaluate and Compare Model Performance:**
    * Evaluated model **accuracy** on test data.
    * Performed **data augmentation** (e.g., flipping, rotation) and analyzed its impact.
    * **Compared:**
        * Model trained without augmentation.
        * Model trained with augmentation.
    * **Visualized** the loss and accuracy curves.
* **Feature Map Visualization:**
    * Extracted and visualized feature maps from different layers to show how early layers capture edges, while deeper layers capture high-level features.
* **Ablation Study: Impact of Hyperparameters on Accuracy:**
    * Conducted an ablation study by modifying and observing the impact of the following four hyperparameters on accuracy:
        * **Learning Rate:** Experimented with at least three different learning rates (e.g., 0.001, 0.01, 0.1) to analyze their effect on model convergence and accuracy.
        * **Batch Size:** Trained the model with different batch sizes (e.g., 16, 32, 64) and compared their influence on training time and performance.
        * **Number of Convolutional Filters:** Varied the number of filters in the convolutional layers (e.g., 16, 32, 64) and observed the effect on feature extraction and accuracy.
        * **Number of Layers:** Modified the number of convolutional layers in the model (e.g., 3, 5, 7 layers) and compared how deeper models perform compared to shallower ones.

### 3.4 Evaluation and Comparison of Model Performance

* **Performance Metrics:**
    * Assessed and compared CNN models using: **Accuracy, Precision, Recall, F1-Score, Confusion Matrix**.
* **Comparison of Models:**
    * Trained and evaluated two models:
        1.  Model without Data Augmentation
        2.  Model with Data Augmentation (e.g., flipping, rotation, shifting)
    * Computed the above metrics for both models and presented results in a tabular format (e.g., `Table 1: Performance Metrics Comparison of CNN Models`).

| Model                | Accuracy | Precision | Recall | F1-Score |
| :------------------- | :------- | :-------- | :----- | :------- |
| Without Augmentation | [Value]  | [Value]   | [Value] | [Value]  |
| With Augmentation    | [Value]  | [Value]   | [Value] | [Value]  |

* **Confusion Matrix Visualization:**
    * Generated and visualized the confusion matrix as a heatmap for both models to analyze classification performance and misclassified categories.
* **Loss and Accuracy Curves:**
    * Plotted training and validation loss over epochs to analyze convergence.
    * Plotted training and validation accuracy over epochs to evaluate overfitting or underfitting.
    * Compared curves for both models and discussed key observations.

### Relevant Files:

* `Question3-CNN-CIFAR10/cnn_cifar10.py` (or Jupyter Notebook)
* `Question3-CNN-CIFAR10/visualizations/` (for plots like Loss/Accuracy curves, Feature Maps, Confusion Matrices)

---

## ⚙️ Question 4: Implementing a Vanilla RNN for Next-Word Prediction

This research-oriented task involves training a Vanilla RNN for next-word prediction on Shakespeare text, with a focus on custom-trained word embeddings.

### 4.1 Objective

To train a **Vanilla Recurrent Neural Network (RNN)** on a Shakespeare text dataset from Hugging Face for next-word prediction. Instead of using pre-trained embeddings (like Word2Vec or GloVe), the task focuses on training custom word embeddings using an **Embedding Layer** in TensorFlow or PyTorch.

### 4.2 Dataset

* **Shakespeare Text Dataset:** Used a publicly available dataset from [Hugging Face - Shakespeare Dataset](https://huggingface.co/datasets/shakespeare).

### 4.3 Implementation Steps

* **Load and Preprocess the Dataset:**
    * Loaded the Shakespeare dataset.
    * Tokenized words to create a vocabulary.
    * Split the dataset into training (80%) and testing (20%) subsets.
* **Implement the Vanilla RNN Model:**
    * Implemented a **custom RNN cell** (no LSTMs or GRUs) using Python Classes for these layers.
    * Used a **trainable Embedding Layer** (TensorFlow/Keras or PyTorch) to learn word representations, setting a suitable embedding size.
    * The model was designed to process word sequences and predict the next word.
    * Used **Cross-Entropy Loss** and an appropriate optimizer (e.g., Adam).
* **Train the Model and Monitor Performance:**
    * Trained the model using **Backpropagation Through Time (BPTT)**.
    * Monitored training loss and validation loss across epochs.
    * Saved the trained model.
* **Generate Text Predictions:**
    * Provided a seed phrase (e.g., “To be or not to”).
    * The model generated the next word iteratively, forming a complete sentence of at least 10 words.
* **Evaluate Model Performance:**
    * Computed and reported the following metrics:
        * **Perplexity** (Measures model uncertainty).
        * **Word-level accuracy**.
        * Loss curve visualization.
    * Compared learned embeddings with randomly initialized ones.
* **Ablation Studies (Pre-trained Embeddings):**
    * Trained a separate RNN model using **pre-trained word embeddings** (Word2Vec or GloVe).
    * Compared its performance with the model trained using randomly initialized embeddings based on:
        * Perplexity
        * Word-level accuracy
        * Loss curve visualization
    * Plotted a **confusion matrix** showing misclassified words.
    * Analyzed and discussed the impact of using pre-trained embeddings on the model’s performance.

### 4.4 Expected Output

* **Comparison Table:** A table comparing word-level accuracy and perplexity for different embedding types (Random vs. Learned).

| Embedding Type    | Word-Level Accuracy | Perplexity |
| :---------------- | :------------------ | :--------- |
| Random Embeddings | [Value]             | [Value]    |
| Learned Embeddings | [Value]             | [Value]    |

* **Generated Text Sequences:** Examples of text sequences generated from the model.

### Relevant Files:

* `Question4-RNN-NextWordPrediction/rnn_shakespeare.py` (or Jupyter Notebook)
* `Question4-RNN-NextWordPrediction/visualizations/` (for plots and images)

---

## ⚙️ Question 5: Hyperparameter Search for CNN and RNN

This final section focuses on applying random search for hyperparameter optimization on the previously implemented CNN and RNN models.

### 5.1 Objective

To implement **hyperparameter search** for both the CNN and RNN models using the **Random Search technique**. The goal is to define a set of hyperparameters to search over, train multiple models with different combinations, select the best-performing configuration based on validation accuracy, and then test these best configurations on the respective test datasets.

### 5.2 Implementation Details

* **Hyperparameter Search Technique:**
    * Used **RandomizedSearchCV from Scikit-Learn** or a custom random sampling approach.
* **Hyperparameters Explored:**
    * Learning rate
    * Number of layers
    * Number of neurons (for RNN) or filters (for CNN)
    * Batch size
    * Optimizer (e.g., Adam, SGD, RMSprop)
    * Activation functions (e.g., ReLU, Tanh, Sigmoid)
    * Dropout rate
    * Kernel size (for CNN)
    * Stride (for CNN)
    * Weight initialization method (e.g., Xavier, He Normal)
* **Model Training and Selection:**
    * Trained multiple models with various hyperparameter combinations.
    * Selected the **best-performing configuration** based on validation accuracy.
* **Final Evaluation:**
    * Tested the **best hyperparameter configurations** for both CNN and RNN models on the test datasets used in their respective previous questions.
    * Compared their performance in terms of the evaluation metrics employed in Questions 3 and 4 (e.g., Accuracy, Precision, Recall, F1-Score for CNN; Perplexity, Word-level accuracy for RNN).
