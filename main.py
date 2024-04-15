import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dataset
from model import LeNet5, CustomMLP
import matplotlib.pyplot as plt

def train(model, dataloader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

def test(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

def plot_results(results):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Testing Results for All Models')

    for name, data in results.items():
        trn_losses, tst_losses, trn_accuracies, tst_accuracies = data
        axs[0, 0].plot(trn_losses, label=f"{name} Train Loss")
        axs[0, 1].plot(tst_losses, label=f"{name} Test Loss")
        axs[1, 0].plot(trn_accuracies, label=f"{name} Train Accuracy")
        axs[1, 1].plot(tst_accuracies, label=f"{name} Test Accuracy")

    axs[0, 0].set_ylabel('Loss')
    axs[0, 1].set_ylabel('Loss')
    axs[1, 0].set_ylabel('Accuracy (%)')
    axs[1, 1].set_ylabel('Accuracy (%)')

    for ax in axs.flat:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = dataset.MNIST('../data/train.tar')
    test_dataset = dataset.MNIST('../data/test.tar')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    models = {
        "LeNet5": LeNet5().to(device),
        "CustomMLP": CustomMLP().to(device),
        "LeNet5_Regularized": LeNet5().to(device)
    }

    models["LeNet5_Regularized"].conv1 = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), 
        nn.BatchNorm2d(6)
    )
    models["LeNet5_Regularized"].conv2 = nn.Sequential(
        nn.Conv2d(6, 16, 5), 
        nn.BatchNorm2d(16)
    )
    models["LeNet5_Regularized"].fc1 = nn.Sequential(
        nn.Linear(16 * 5 * 5, 120),
        nn.Dropout(0.2)
    )
    models["LeNet5_Regularized"].fc2 = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(120, 84) 
    )
    models["LeNet5_Regularized"].fc3 = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(84, 10)
    )
    criterion = nn.CrossEntropyLoss()
    results = {}

    for name, model in models.items():
        print(f"{name} has {count_parameters(model):,} trainable parameters.")
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        trn_losses, tst_losses, trn_accuracies, tst_accuracies = [], [], [], []

        best_test_loss = float('inf')  
        no_improve_epochs = 0  

        for epoch in range(15):  
            train_loss, train_accuracy = train(model, train_loader, device, criterion, optimizer)
            test_loss, test_accuracy = test(model, test_loader, device, criterion)

            trn_losses.append(train_loss)
            tst_losses.append(test_loss)
            trn_accuracies.append(train_accuracy)
            tst_accuracies.append(test_accuracy)

            print(f'Epoch {epoch + 1}, {name}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.2f}%, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.2f}%')

            # 조기 종료 조건 확인
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improve_epochs = 0  
            else:
                no_improve_epochs += 1 
                if no_improve_epochs >= 3:
                    print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in test loss.")
                    break 

        results[name] = (trn_losses, tst_losses, trn_accuracies, tst_accuracies)

    plot_results(results)
    
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """    
    
if __name__ == '__main__':
    main()
