# MNIST-Classification
인공신경망과 딥러닝 

Assignment #2

24510101 김상훈 
shkim@ds.seoultech.ac.kr

## Lenet-5 
![Lenet-5](./images/lenet5.png)

### LeNet-5 Model Architecture Details

#### First Convolutional Layer (`conv1`):
- **Input Channels:** 1 (grayscale image)
- **Output Channels:** 6
- **Kernel Size:** 5x5
- **Padding:** 2
- **Parameters:** \( (5 \times 5 \times 1 \times 6) + 6 \) (weights + biases)
- **Total:** 150 + 6 = 156

#### First Pooling Layer (`pool1`):
- **Type:** Max Pooling
- **Kernel Size:** 2x2
- **Stride:** 2
- **Parameters:** 0 (pooling layers do not have trainable parameters)

#### Second Convolutional Layer (`conv2`):
- **Input Channels:** 6
- **Output Channels:** 16
- **Kernel Size:** 5x5
- **Parameters:** \( (5 \times 5 \times 6 \times 16) + 16 \)
- **Total:** 2,400 + 16 = 2,416

#### Second Pooling Layer (`pool2`):
- **Type:** Max Pooling
- **Kernel Size:** 2x2
- **Stride:** 2
- **Parameters:** 0

#### First Fully Connected Layer (`fc1`):
- **Input Features:** 16 \times 5 \times 5 = 400
- **Output Features:** 120
- **Parameters:** \(400 \times 120 + 120\)
- **Total:** 48,000 + 120 = 48,120

#### Second Fully Connected Layer (`fc2`):
- **Input Features:** 120
- **Output Features:** 84
- **Parameters:** \(120 \times 84 + 84\)
- **Total:** 10,080 + 84 = 10,164

#### Third Fully Connected Layer (`fc3`):
- **Input Features:** 84
- **Output Features:** 10
- **Parameters:** \(84 \times 10 + 10\)
- **Total:** 840 + 10 = 850

### Total Parameters for LeNet-5:
- **Total:** 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706

![Lenet-5](./images/lenet5_result.png)

## Custom MLP

![Lenet-5](./images/custom_mlp.png)

### CustomMLP Model Architecture

The CustomMLP model consists of several fully connected layers, as described:

#### First Fully Connected Layer (`fc1`):
- **Input Features:** 784 (flattened 28x28 image)
- **Output Features:** 64
- **Parameters:** Calculation of weights and biases:
  - **Weights:** \(784 \times 64 = 50,176\)
  - **Biases:** 64
  - **Total for fc1:** \(50,176 + 64 = 50,240\)

#### Second Fully Connected Layer (`fc2`):
- **Input Features:** 64
- **Output Features:** 64
- **Parameters:** Calculation of weights and biases:
  - **Weights:** \(64 \times 64 = 4,096\)
  - **Biases:** 64
  - **Total for fc2:** \(4,096 + 64 = 4,160\)

#### Third Fully Connected Layer (`fc3`):
- **Input Features:** 64
- **Output Features:** 64
- **Parameters:** Same as fc2
  - **Total for fc3:** 4,160

#### Fourth Fully Connected Layer (`fc4`):
- **Input Features:** 64
- **Output Features:** 32
- **Parameters:**
  - **Weights:** \(64 \times 32 = 2,048\)
  - **Biases:** 32
  - **Total for fc4:** \(2,048 + 32 = 2,080\)

#### Fifth Fully Connected Layer (`fc5`):
- **Input Features:** 32
- **Output Features:** 16
- **Parameters:**
  - **Weights:** \(32 \times 16 = 512\)
  - **Biases:** 16
  - **Total for fc5:** \(512 + 16 = 528\)

#### Sixth Fully Connected Layer (`fc6`):
- **Input Features:** 16
- **Output Features:** 10
- **Parameters:**
  - **Weights:** \(16 \times 10 = 160\)
  - **Biases:** 10
  - **Total for fc6:** \(160 + 10 = 170\)

### Total Parameters Calculation
Adding all the parameters from each layer, we get:

- **fc1:** 50,240
- **fc2:** 4,160
- **fc3:** 4,160
- **fc4:** 2,080
- **fc5:** 528
- **fc6:** 170
- **Total Parameters for CustomMLP:** 50,240 + 4,160 + 4,160 + 2,080 + 528 + 170 = 61,338


![Lenet-5](./images/custom_mlp_result.png)

## Lenet-5_Regularized

![Lenet-5](./images/lenet5_regularized.png)

![Lenet-5](./images/lenet5_regularized_result.png)

## Results

![Lenet-5](./images/full_result.png)

