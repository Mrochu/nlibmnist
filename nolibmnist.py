from PIL import Image
import os

base_path = "/Users/saint/Documents/GitHub/nlibmnist"
train_path = "/Users/saint/Documents/GitHub/nlibmnist/trainingSet"

train_dataset = load_dataset(train_path)
print(f"훈련 데이터셋 크기: {len(train_dataset)}개")

def load_dataset(data_dir):
    dataset = []
    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        label = int(label_name)  # 폴더명이 라벨
        for filename in os.listdir(label_path):
            if filename.endswith('.jpg'):
                img_path = os.path.join(label_path, filename)
                with Image.open(img_path) as img:
                    img = img.convert('L')        # 흑백 변환
                    img = img.resize((28, 28))   # 28x28 픽셀로 조정
                    pixels = list(img.getdata()) # 784개 픽셀
                    pixels = [p / 255.0 for p in pixels]  # 정규화
                    dataset.append((pixels, label))  # zip 형태로 저장
    return dataset

