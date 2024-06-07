import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from torch.optim import SGD 
from PIL import Image
import os


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, class_label, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_label = class_label
        self.images_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images_list[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_label
        
        return image, label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten(1, -1)
        self.fc = nn.Linear(in_features=3*3*10, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dir1 = "./GTSRB_subset_2/class1"
dir2 = "./GTSRB_subset_2/class2"
data_class1 = CustomImageDataset(dir1, 0, transform)
data_class2 = CustomImageDataset(dir2, 1, transform)
all_data = ConcatDataset([data_class1, data_class2])
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(all_data, [0.8, 0.2], generator=generator)

train_dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = CNN()
print(model)

num_of_epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.2)

for e in range(num_of_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in train_dataset:
        optimizer.zero_grad()

        images, labels = batch
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {e+1}/{num_of_epochs}, Loss: {epoch_loss / len(train_dataset)}")

model.eval()

test_loss = 0.0
correct = 0
total = 0

# Disable gradient computation during testing
with torch.no_grad():
    for batch in test_dataset:
        images, labels = batch
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / len(test_dataset)

accuracy = correct / total

print(f"Test Loss: {avg_test_loss}, Accuracy: {accuracy}")