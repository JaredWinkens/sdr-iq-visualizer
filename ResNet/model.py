import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ============================================
# OPTION 1: ResNet Architecture (RECOMMENDED)
# ============================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RadioSignalResNet(nn.Module):
    """
    ResNet for Radio Signal Classification
    Input: (batch_size, 2, 128) - IQ samples
    Output: (batch_size, num_classes)
    """
    def __init__(self, num_classes=11):
        super(RadioSignalResNet, self).__init__()
        
        # Reshape IQ data: (2, 128) -> (1, 2, 128)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Adaptive pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch, 2, 128)
        x = x.unsqueeze(1)  # (batch, 1, 2, 128)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# ============================================
# OPTION 2: 1D CNN (Faster, Simpler)
# ============================================

class RadioSignalCNN1D(nn.Module):
    """
    1D CNN for Radio Signal Classification
    Works directly on flattened IQ time-series
    """
    def __init__(self, num_classes=11):
        super(RadioSignalCNN1D, self).__init__()
        
        # Input: (batch, 2, 128) -> flatten to (batch, 1, 256)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(256 * 32, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Flatten IQ: (batch, 2, 128) -> (batch, 1, 256)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================
# OPTION 3: CLDNN (CNN + LSTM + DNN)
# ============================================

class RadioSignalCLDNN(nn.Module):
    """
    CLDNN for Radio Signal Classification
    Combines CNN for feature extraction and LSTM for temporal patterns
    """
    def __init__(self, num_classes=11):
        super(RadioSignalCLDNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((1, 2))
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((1, 2))
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=128*2, hidden_size=128, 
                           num_layers=2, batch_first=True, dropout=0.3)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 2, 128)
        x = x.unsqueeze(1)  # (batch, 1, 2, 128)
        
        # CNN feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Reshape for LSTM: (batch, seq_len, features)
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, x.size(1), -1)
        
        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last time step
        
        # Dense classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================
# Training Setup
# ============================================

class RadioSignalTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.argmax(dim=1) if len(labels.shape) > 1 else labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(dim=1) if len(labels.shape) > 1 else labels).sum().item()
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.argmax(dim=1) if len(labels.shape) > 1 else labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.argmax(dim=1) if len(labels.shape) > 1 else labels).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=100):
        best_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"✓ Saved new best model with accuracy: {best_acc:.2f}%")
            print("-" * 60)

def create_model(architecture='resnet', num_classes=11):
    """
    Factory function to create models
    
    Args:
        architecture: 'resnet', 'cnn1d', or 'cldnn'
        num_classes: Number of modulation types to classify
    """
    if architecture == 'resnet':
        return RadioSignalResNet(num_classes)
    elif architecture == 'cnn1d':
        return RadioSignalCNN1D(num_classes)
    elif architecture == 'cldnn':
        return RadioSignalCLDNN(num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model('resnet', num_classes=11)
    
    # Test with dummy data
    batch_size = 32
    dummy_input = torch.randn(batch_size, 2, 128)  # IQ samples
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")