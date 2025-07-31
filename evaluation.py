import logging
import os
import time
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluation:
    def __init__(self, model, device, test_dataset, config):
        self.model = model
        self.device = device
        self.test_dataset = test_dataset
        self.config = config
        self.metrics = {}

        # Initialize metrics
        self.correct_predictions = 0
        self.total_predictions = 0
        self.confusion_matrix = np.zeros((config.num_classes, config.num_classes), dtype=int)

    def evaluate(self):
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=os.cpu_count())

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                self.total_predictions += labels.size(0)
                self.correct_predictions += (predicted == labels).sum().item()

                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    self.confusion_matrix[t.long(), p.long()] += 1

        accuracy = self.correct_predictions / self.total_predictions
        logging.info(f'Accuracy: {accuracy:.4f}')
        self.metrics['accuracy'] = accuracy

        precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        logging.info(f'Precision: {precision}')
        self.metrics['precision'] = precision

        recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        logging.info(f'Recall: {recall}')
        self.metrics['recall'] = recall

        f1_score = 2 * (precision * recall) / (precision + recall)
        logging.info(f'F1 Score: {f1_score}')
        self.metrics['f1_score'] = f1_score

    def compute_metrics(self):
        self.evaluate()
        return self.metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, config.checkpoint_name))
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    evaluator = Evaluation(model, device, test_dataset, config)
    metrics = evaluator.compute_metrics()

    logging.info('Evaluation results:')
    for metric, value in metrics.items():
        logging.info(f'{metric}: {value}')

if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    main()
    end_time = time.time()
    logging.info(f'Evaluation completed in {end_time - start_time:.2f} seconds.')

class Config:
    def __init__(self):
        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_name = 'model_checkpoint.pt'
        self.batch_size = 64
        self.num_classes = 10

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.config.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x