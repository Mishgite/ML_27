import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Конфигурация
DATA_DIR = './'
BATCH_SIZE = 128 if torch.cuda.is_available() else 32
IMG_SIZE = 192
EPOCHS = 50
NUM_CLASSES = 20
LR = 0.002
ACCUM_STEPS = 2

# Оптимизированные трансформации
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class OptimizedDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        img_id = self.df.iloc[idx]['img_id']
        for ext in ['.jpg', '.png']:
            img_path = os.path.join(self.img_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                break

        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        self.cache[idx] = (image, label)
        return image, label


class TurboCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main():
    # Инициализация
    torch.backends.cudnn.benchmark = True if device.type == 'cuda' else False

    # Загрузка данных
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_answers.csv'))
    categories_df = pd.read_csv(os.path.join(DATA_DIR, 'activity_categories.csv'))

    # Преобразование меток
    category_map = {row['id']: i for i, row in categories_df.iterrows()}
    train_df['label'] = train_df['target_feature'].map(category_map)

    # Стратифицированное разделение
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        stratify=train_df['label'],
        random_state=42
    )

    # Датасеты
    train_set = OptimizedDataset(
        os.path.join(DATA_DIR, 'img_train'),
        train_df.iloc[train_idx],
        train_transform
    )

    val_set = OptimizedDataset(
        os.path.join(DATA_DIR, 'img_train'),
        train_df.iloc[val_idx],
        val_transform
    )

    # DataLoader'ы
    num_workers = 4 if device.type == 'cuda' else 2
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Инициализация модели
    model = TurboCNN().to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # Настройка оптимизации
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    if device.type == 'cuda':
        optimizer = optim.AdamW(model.parameters(), lr=LR, fused=True)

    # Настройка mixed precision
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 3,
        steps_per_epoch=len(train_loader) // ACCUM_STEPS,
        epochs=EPOCHS
    )

    # Обучение
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        # Валидация
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = correct / len(val_set)
        print(f"Epoch {epoch + 1}: Val Acc = {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    # Предсказание
    test_files = sorted(
        [f for f in os.listdir(os.path.join(DATA_DIR, 'img_test'))
         if f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0]))

    test_df = pd.DataFrame({
        'img_id': [int(f.split('.')[0]) for f in test_files],
        'label': 0
    })

    test_set = OptimizedDataset(
        os.path.join(DATA_DIR, 'img_test'),
        test_df,
        val_transform
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            predictions.extend(outputs.argmax(1).cpu().numpy())

    submission = pd.DataFrame({
        'id': test_df['img_id'],
        'target_feature': predictions
    }).sort_values('id')

    submission.to_csv('submission.csv', index=False)
    print("Результаты сохранены в submission.csv")


if __name__ == '__main__':
    main()
