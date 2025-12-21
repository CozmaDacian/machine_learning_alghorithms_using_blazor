Custom Machine Learning Library with Blazor Interface
A comprehensive machine learning application built entirely from scratch in C#, featuring a web-based user interface developed with Blazor. This project demonstrates the fundamental mathematics behind popular algorithms without relying on external machine learning libraries like TensorFlow, PyTorch, or ML.NET.
<img width="1550" height="831" alt="image" src="https://github.com/user-attachments/assets/98dc4359-17a6-474f-8c67-9cd1a55bd5e6" />

Key Technical Highlights
Pure C# Implementation: All mathematical operations, matrix manipulations, and optimization algorithms are implemented using standard C# collections and arrays.

Interactive Web Interface: A Blazor-based UI allows users to upload datasets, configure hyperparameters, and visualize training results in real-time.

Custom Math Engine: Includes manual implementations of derivatives for backpropagation and statistical probability density functions.

Implemented Algorithms
1. Neural Network (Deep Learning)
A fully connected feedforward neural network built from the ground up.

Architecture: configurable layers and neuron counts.

Training: Implements Backpropagation algorithm for weight updates.

Loss Functions: Supports multiple loss functions (e.g., Mean Squared Error, Cross-Entropy) to evaluate model performance during training.

2. Naive Bayes Classifier
A probabilistic classifier based on Bayes' theorem, optimized for continuous data.

Distribution: Uses a Gaussian (Normal) Distribution to calculate the likelihood of features.

Logic: Calculates class probabilities by assuming feature independence, making it highly efficient for high-dimensional datasets.

3. K-Nearest Neighbors (KNN)
A non-parametric, lazy learning algorithm used for classification and regression.

Distance Metric: Specifically implements the Manhattan Distance (L1 Norm) to calculate similarity between data points:



<img width="1547" height="761" alt="image" src="https://github.com/user-attachments/assets/eb610a7b-546a-497f-8fdd-141c0a208072" />
