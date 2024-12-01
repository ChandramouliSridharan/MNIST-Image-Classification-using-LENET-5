# MNIST Image Classification Using LeNet-5

## Project Overview  
This project demonstrates image classification on the MNIST dataset using the LeNet-5 architecture implemented in PyTorch without using Keras. 
The MNIST dataset consists of grayscale images of handwritten digits ranging from 0 to 9. The primary goal is to classify these images into their respective digit classes.

## Dataset Description  
The dataset contains two main parts:
1. **Training Dataset**: `mnist_train.csv`  
2. **Test Dataset**: `mnist_test.csv`

Each image is represented by 784 pixels (28x28) flattened into a single row.  
- **Labels**: The digit (0-9) corresponding to the image.

### Data Preprocessing
1. **Loading the Dataset**: The dataset is loaded into Pandas DataFrames.  
2. **Reshaping**: Images are reshaped to (1, 28, 28) for PyTorch's expected input format.  
3. **Normalization**: The pixel values are scaled to a range of [0, 1] by dividing by 255.  
4. **Padding**: Images are padded to (1, 32, 32) to align with the LeNet-5 input requirements.

### Data Splitting
- **Training Set**: 80% of the training data.  
- **Validation Set**: 20% of the training data.  
- **Test Set**: Separate dataset for final evaluation.

## LeNet-5 Architecture
The LeNet-5 architecture consists of the following layers:
1. **Convolutional Layer 1**: 6 filters, kernel size 5x5, activation: Tanh.  
2. **Average Pooling Layer 1**: Kernel size 2x2, stride 2.  
3. **Convolutional Layer 2**: 16 filters, kernel size 5x5, activation: Tanh.  
4. **Average Pooling Layer 2**: Kernel size 2x2, stride 2.  
5. **Convolutional Layer 3**: 120 filters, kernel size 4x4, activation: Tanh.  
6. **Fully Connected Layer 1**: 120 → 84, activation: Tanh.  
7. **Fully Connected Layer 2**: 84 → 10, activation: Softmax.

### Model Initialization
- **Loss Function**: CrossEntropyLoss  
- **Optimizer**: Adam with learning rate 0.001  

## Training and Validation
The model was trained for 10 epochs with a batch size of 64.  
### Accuracy Calculation Function
The accuracy was calculated as the percentage of correctly predicted labels over the total samples.

### Training Results
During training, both accuracy and loss were tracked for each epoch.

Epoch 1/10 --- Train Accuracy: 94.98% --- Validation Accuracy: 94.93%
Epoch 2/10 --- Train Accuracy: 96.52% --- Validation Accuracy: 96.36%
Epoch 3/10 --- Train Accuracy: 97.72% --- Validation Accuracy: 97.35%
Epoch 4/10 --- Train Accuracy: 97.90% --- Validation Accuracy: 97.48%
Epoch 5/10 --- Train Accuracy: 98.40% --- Validation Accuracy: 97.91%
Epoch 6/10 --- Train Accuracy: 98.58% --- Validation Accuracy: 98.01%
Epoch 7/10 --- Train Accuracy: 98.48% --- Validation Accuracy: 97.74%
Epoch 8/10 --- Train Accuracy: 98.59% --- Validation Accuracy: 97.99%
Epoch 9/10 --- Train Accuracy: 98.95% --- Validation Accuracy: 98.34%
Epoch 10/10 --- Train Accuracy: 99.06% --- Validation Accuracy: 98.23%

## Testing the Model  
The model was evaluated on the test set to determine the final accuracy:  
**Test Accuracy**: `98.41%`

## Conclusion  
- The LeNet-5 model successfully classified MNIST digits with high accuracy.  
- The final test accuracy demonstrates the robustness of the model.  
