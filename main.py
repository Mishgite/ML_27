import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# Конфигурация
DATA_DIR = './'
BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 20
NUM_CLASSES = 20

# Ожидаемые категории
EXPECTED_CATEGORIES = [
    'sports', 'inactivity quiet/light', 'miscellaneous', 'occupation', 'water activities',
    'home activities', 'lawn and garden', 'religious activities', 'winter activities',
    'conditioning exercise', 'bicycling', 'fishing and hunting', 'dancing', 'walking',
    'running', 'self care', 'home repair', 'volunteer activities', 'music playing', 'transportation'
]

# Проверка файлов
required_files = [
    os.path.join(DATA_DIR, 'train_answers.csv'),
    os.path.join(DATA_DIR, 'activity_categories.csv'),
    os.path.join(DATA_DIR, 'img_train'),
    os.path.join(DATA_DIR, 'img_test')
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Отсутствует файл: {path}")

# Трансформации данных
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ActivityDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Получаем имя файла из столбца img_id
        img_id = self.df.iloc[idx]['img_id']
        img_name = f"{img_id}.jpg"  # Добавляем расширение
        label = self.df.iloc[idx]['label']
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


class CNNModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def main():
    # Загрузка данных
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_answers.csv'))
    categories_df = pd.read_csv(os.path.join(DATA_DIR, 'activity_categories.csv'))

    # Проверка структуры данных
    print("Столбцы train_answers.csv:", train_df.columns.tolist())
    print("Первые 5 строк train_answers.csv:\n", train_df.head())

    # Создание маппинга категорий
    category_map = {}
    for _, row in categories_df.iterrows():
        try:
            category_map[row['id']] = EXPECTED_CATEGORIES.index(row['category'])
        except ValueError as e:
            raise ValueError(f"Неизвестная категория: {row['category']}") from e

    # Преобразование меток
    train_df['label'] = train_df['target_feature'].map(category_map)

    # Проверка меток
    print("\nСтатистика меток:")
    print("Минимум:", train_df['label'].min())
    print("Максимум:", train_df['label'].max())
    print("Уникальные значения:", sorted(train_df['label'].unique()))

    if train_df['label'].isna().any():
        raise ValueError("Обнаружены NaN в метках")
    if (train_df['label'].min() < 0) or (train_df['label'].max() >= NUM_CLASSES):
        raise ValueError(f"Метки должны быть в диапазоне 0-{NUM_CLASSES - 1}")

    # Датасеты и загрузчики
    full_dataset = ActivityDataset(
        img_dir=os.path.join(DATA_DIR, 'img_train'),
        df=train_df,
        transform=train_transform
    )

    train_size = int(0.8 * len(full_dataset))
    train_set, val_set = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # Обучение
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    # Предсказание
    test_files = sorted(
        [f for f in os.listdir(os.path.join(DATA_DIR, 'img_test')) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(x.split('.')[0]))

    test_df = pd.DataFrame({
        'img_id': [int(f.split('.')[0]) for f in test_files],
        'image': test_files
    })
    test_df['label'] = 0  # Фиктивные метки

    test_dataset = ActivityDataset(
        img_dir=os.path.join(DATA_DIR, 'img_test'),
        df=test_df,
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Безопасная загрузка модели
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # Сохранение результатов
    submission = pd.DataFrame({
        'id': test_df['img_id'],
        'target_feature': predictions
    }).sort_values('id')

    submission.to_csv('submission.csv', index=False)
    print("Результаты сохранены в submission.csv")


if __name__ == '__main__':
    main()
