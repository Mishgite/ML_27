import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
BATCH_SIZE = 64 if torch.cuda.is_available() else 32
IMG_SIZE = 112
EPOCHS = 100
PATIENCE = 7
NUM_CLASSES = 20
LR = 0.0005
ACCUM_STEPS = 2

# Улучшенные аугментации для RNN
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


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


class DeepVisionRNN(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN часть
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x56x56

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128x28x28

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # 256x7x7
        )

        # LSTM часть
        self.rnn = nn.LSTM(
            input_size=256,  # Исправлено: теперь соответствует размеру признаков на временной шаг
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # 2*512 из бидирекционного LSTM
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        # CNN
        x = self.cnn(x)  # [B, 256, 7, 7]

        # Преобразование в последовательность
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # [B, 7, 7, 256]
        x = x.reshape(batch_size, 7 * 7, 256)  # [B, 49, 256]

        # LSTM
        x, _ = self.rnn(x)  # [B, 49, 1024]

        # Усреднение по временным шагам
        x = x.mean(dim=1)

        return self.classifier(x)


def main():
    # Инициализация
    torch.backends.cudnn.benchmark = True if device.type == 'cuda' else False

    # Загрузка данных
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_answers.csv'))
    categories_df = pd.read_csv(os.path.join(DATA_DIR, 'activity_categories.csv'))
    category_map = {row['id']: i for i, row in categories_df.iterrows()}
    train_df['label'] = train_df['target_feature'].map(category_map)

    # Стратифицированное разделение
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        stratify=train_df['label'],
        random_state=42
    )

    # Датасеты и DataLoader'ы
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

    num_workers = 4 if device.type == 'cuda' else 2
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Инициализация модели
    model = DeepVisionRNN().to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # Оптимизация
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=2,
        factor=0.5
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scaler = torch.amp.GradScaler(
        init_scale=2. ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=(device.type == 'cuda')
    )
    # Обучение
    best_acc = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        # Тренировочная эпоха
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixed precision training
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS

        # Валидация
        model.eval()
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = correct / len(val_set)
        avg_val_loss = val_loss / len(val_set)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}:")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Ранняя остановка и планировщик
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model_rnn.pth')
            print(f"New best accuracy: {best_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping after {PATIENCE} epochs without improvement")
                break

    # Предсказание на тестовых данных
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

    model.load_state_dict(torch.load('best_model_rnn.pth', map_location=device, weights_only=True))
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(images)
            predictions.extend(outputs.argmax(1).cpu().numpy())

    submission = pd.DataFrame({
        'id': test_df['img_id'],
        'target_feature': predictions
    }).sort_values('id')

    submission.to_csv('submission_rnn.csv', index=False)
    print("Результаты сохранены в submission_rnn.csv")


if __name__ == '__main__':
    main()
