import logging
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    'DATA_PATH': 'data',
    'MODEL_PATH': 'models',
    'LOG_PATH': 'logs',
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LEARNING_RATE': 0.001,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from CSV files"""
        try:
            train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
            test_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'))
            return train_df, test_df
        except FileNotFoundError:
            logger.error('Data file not found')
            raise

    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess data by scaling and encoding categorical variables"""
        try:
            scaler = StandardScaler()
            train_df[['feature1', 'feature2']] = scaler.fit_transform(train_df[['feature1', 'feature2']])
            test_df[['feature1', 'feature2']] = scaler.transform(test_df[['feature1', 'feature2']])
            return train_df, test_df
        except NotFittedError:
            logger.error('Scaler not fitted')
            raise

    def split_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets"""
        try:
            X_train, X_val, y_train, y_val = train_test_split(train_df.drop('target', axis=1), train_df['target'], test_size=0.2, random_state=42)
            return X_train, X_val, y_train, y_val
        except ValueError:
            logger.error('Invalid data shape')
            raise

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Trainer:
    def __init__(self, model: Model, device: torch.device):
        self.model = model
        self.device = device

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Train the model"""
        try:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG['LEARNING_RATE'])
            criterion = nn.BCELoss()
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                for batch in train_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels.view(-1, 1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    correct = 0
                    for batch in val_loader:
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels.view(-1, 1))
                        val_loss += loss.item()
                        pred = (outputs > 0.5).float()
                        correct += (pred == labels.view(-1, 1)).sum().item()
                    accuracy = correct / len(val_loader.dataset)
                    logger.info(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Val Accuracy: {accuracy:.4f}')
        except Exception as e:
            logger.error(f'Training failed: {str(e)}')
            raise

class Evaluator:
    def __init__(self, model: Model, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(self, test_loader: DataLoader):
        """Evaluate the model"""
        try:
            self.model.eval()
            with torch.no_grad():
                correct = 0
                for batch in test_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    pred = (outputs > 0.5).float()
                    correct += (pred == labels.view(-1, 1)).sum().item()
                accuracy = correct / len(test_loader.dataset)
                logger.info(f'Test Accuracy: {accuracy:.4f}')
                return accuracy
        except Exception as e:
            logger.error(f'Evaluation failed: {str(e)}')
            raise

def main():
    # Load data
    data_processor = DataProcessor(CONFIG['DATA_PATH'])
    train_df, test_df = data_processor.load_data()

    # Preprocess data
    train_df, test_df = data_processor.preprocess_data(train_df, test_df)

    # Split data
    X_train, X_val, y_train, y_val = data_processor.split_data(train_df, test_df)

    # Create dataset and data loader
    class CustomDataset(Dataset):
        def __init__(self, X: pd.DataFrame, y: pd.Series):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return self.X.iloc[index], self.y.iloc[index]

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    # Create model and trainer
    model = Model()
    device = CONFIG['DEVICE']
    model.to(device)
    trainer = Trainer(model, device)

    # Train model
    trainer.train(train_loader, val_loader, CONFIG['EPOCHS'])

    # Evaluate model
    evaluator = Evaluator(model, device)
    accuracy = evaluator.evaluate(test_loader)

if __name__ == '__main__':
    main()