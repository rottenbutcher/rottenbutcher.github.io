import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim

# 1. 기본 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 하이퍼파라미터
epochs = 10
batch_size = 64
learning_rate = 0.001

# 2. 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 데이터셋 분할 부분 ---

# 전체 훈련 데이터셋(60,000개)을 우선 모두 불러옵니다.
full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                  download=True, transform=transform)

# 훈련셋과 검증셋의 크기를 정의합니다.
train_size = 50000
val_size = 10000 # 60000 - 50000

# random_split을 사용해 전체 훈련셋을 훈련셋과 검증셋으로 나눕니다.
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 공식 테스트 데이터셋을 불러옵니다.
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)

# 각 데이터셋에 대한 DataLoader를 생성합니다.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 분할 완료 ---

print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"검증 데이터셋 크기: {len(val_dataset)}")
print(f"테스트 데이터셋 크기: {len(test_dataset)}")

# 이제 train_loader, val_loader, test_loader를 사용하여
# 각각 훈련, 검증, 테스트를 진행할 수 있습니다.

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫 번째 합성곱 블록
        # 입력 채널: 1 (흑백), 출력 채널: 16, 커널 크기: 5, 패딩: 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 이미지 크기를 절반으로 줄임
        )
        # 두 번째 합성곱 블록
        # 입력 채널: 16, 출력 채널: 32, 커널 크기: 5, 패딩: 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 이미지 크기를 다시 절반으로 줄임
        )
        # 완전 연결 계층 (분류기)
        # 7*7*32는 두 번의 풀링을 거친 후의 피처 맵 크기입니다.
        # 최종 출력은 10개 (숫자 0~9) 입니다.
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # 1차원 벡터로 펼치기
        out = self.fc(out)
        return out

# 모델 인스턴스 생성 및 GPU로 이동
model = CNN().to(device)
print(model)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 훈련 루프
for epoch in range(epochs):
    model.train() # 모델을 훈련 모드로 설정
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # 데이터를 GPU로 이동
        images = images.to(device)
        labels = labels.to(device)

        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Finished Training')

# 평가 루프
model.eval() # 모델을 평가 모드로 설정
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 훈련된 모델의 가중치 저장
torch.save(model.state_dict(), 'mnist_cnn.pth')
print('Model saved to mnist_cnn.pth')

# 훈련된 모델의 가중치 저장
torch.save(model.state_dict(), 'mnist_cnn.pth')
print('Model saved to mnist_cnn.pth')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from ipycanvas import Canvas, hold_canvas
from ipywidgets import Button, VBox, Label
import io

# 1. CNN 모델 클래스 정의 (동일)
class CNN(nn.Module):
    # ... (이전과 동일한 모델 코드) ...
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x); out = self.layer2(out)
        out = out.reshape(out.size(0), -1); out = self.fc(out)
        return out

# 2. 훈련된 모델 로드 (동일)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.eval()

# 3. 이미지 전처리 함수
def preprocess_image(image_data):
    # [수정 1] ImageOps.invert 코드를 제거합니다.
    # Create a PIL Image from the raw pixel data
    # [FIX] Create a PIL Image directly from the numpy array data
    img = Image.fromarray(image_data, 'RGBA').convert('L')

    transform = transforms.Compose([
        transforms.Resize((28, 28)), transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)

# 4. 실시간 드로잉 캔버스 및 위젯 생성
canvas = Canvas(width=280, height=280, layout={'border': '2px solid lightgray'})
canvas.sync_image_data = True # Not needed if we get data via get_image_data()

# [수정 2] 배경을 검게 칠하고, 펜 색을 흰색으로 설정
with hold_canvas(canvas):
    canvas.fill_style = 'black' # Set background to black
    canvas.fill_rect(0, 0, canvas.width, canvas.height)
    canvas.stroke_style = 'white' # Set drawing color to white
    canvas.line_width = 13

drawing = False
# ... (마우스 이벤트 핸들러는 이전과 동일) ...
def on_mouse_down(x, y):
    global drawing
    drawing = True
    canvas.move_to(x, y)

def on_mouse_move(x, y):
    if drawing:
        canvas.line_to(x, y)
        canvas.stroke()

def on_mouse_up(x, y):
    global drawing
    drawing = False

canvas.on_mouse_down(on_mouse_down)
canvas.on_mouse_move(on_mouse_move)
canvas.on_mouse_up(on_mouse_up)

# 버튼 및 레이블 생성
classify_btn = Button(description="인식하기")
clear_btn = Button(description="지우기")
result_label = Label(value="흰색으로 숫자를 그려주세요")

# 버튼 클릭 이벤트 핸들러

def classify_handwriting(b):
    # Get the pixel data from the canvas
    image_data = canvas.get_image_data()

    # --- ERROR FIX ---
    # Before (ambiguous): if not image_data:
    # After (specific): Check if the numpy array's size is 0
    if image_data.size == 0:
        result_label.value = "Canvas is empty. Please draw a digit."
        return
    # --- END FIX ---

    # The rest of the function remains the same
    # [FIX] Pass the numpy array to preprocess_image
    img_tensor = preprocess_image(image_data).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        prediction = predicted.item()

    result_label.value = f'Model prediction: {prediction}'

# Make sure to re-assign the button click event
classify_btn.on_click(classify_handwriting)

def clear_canvas(b):
    # [수정 3] 캔버스를 지울 때도 검은색으로 다시 칠합니다.
    with hold_canvas(canvas):
        canvas.clear()
        canvas.fill_style = 'black'
        canvas.fill_rect(0, 0, canvas.width, canvas.height)
    result_label.value = "흰색으로 숫자를 그려주세요"

classify_btn.on_click(classify_handwriting)
clear_btn.on_click(clear_canvas)

# 위젯들을 화면에 표시
display(VBox([canvas, classify_btn, clear_btn, result_label]))
