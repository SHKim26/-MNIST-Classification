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

![Lenet-5](./images/custom_mlp_result.png)

## Lenet-5_Regularized

![Lenet-5](./images/lenet5_regularized.png)

![Lenet-5](./images/lenet5_regularized_result.png)

## Results

![Lenet-5](./images/full_result.png)

