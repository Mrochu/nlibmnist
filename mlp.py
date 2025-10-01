import csv
import math
import random

# --- 1. 데이터 로딩 ---
all_data = []
data_path = '/Users/saint/Documents/GitHub/nlibmnist/mnist_dataset.csv' # 여기에 실제 mnist.csv 파일의 전체 경로를 입력해야 합니다.

try:
    with open(data_path,'r') as file:
        reader = csv.reader(file)
        next(reader, None) # 헤더 건너뛰기
        for row in reader:
            all_data.append([int(i) for i in row])
except FileNotFoundError:
    print(f"오류: '{data_path}' 파일을 찾을 수 없습니다.")

# --- 2. 데이터 분리 (훈련/테스트) ---
if all_data:
    random.shuffle(all_data)
    split_index = int(len(all_data) * 0.8)
    train_set = all_data[:split_index]
    test_set = all_data[split_index:]
    print(f"데이터 로딩 및 분리 완료. (전체: {len(all_data)}, 훈련: {len(train_set)}, 테스트: {len(test_set)})")
else:
    print("데이터가 로드되지 않았습니다.")
    train_set, test_set = [], []

# --- 3. MLP 구현 ---
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 가중치 초기화 (입력층 -> 은닉층, 은닉층 -> 출력층)
        self.weights_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.weights_hidden_output = [[random.uniform(-0.5, 0.5) for _ in range(self.output_size)] for _ in range(self.hidden_size)]

        # 편향 초기화 (은닉층, 출력층)
        self.bias_hidden = [0.0 for _ in range(self.hidden_size)]
        self.bias_output = [0.0 for _ in range(self.output_size)]

        print(f"MLP 모델 생성 완료: {input_size} -> {hidden_size} -> {output_size}")

    def _sigmoid(self, x):
        """안정적인 시그모이드 함수"""
        return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))

    def _sigmoid_derivative(self, x):
        """시그모이드 함수의 미분값을 계산합니다. x는 시그모이드 함수를 통과한 값(output)입니다."""
        return x * (1 - x)

    def _calculate_layer_output(self, inputs, weights, biases):
        """한 개 층의 출력을 계산하는 헬퍼 함수. 가중합과 활성화 함수 출력을 모두 반환합니다."""
        num_inputs = len(inputs)
        num_outputs = len(biases)
        
        weighted_sums = [0.0] * num_outputs
        layer_outputs = [0.0] * num_outputs

        for i in range(num_outputs):
            weighted_sum = 0.0
            for j in range(num_inputs):
                weighted_sum += inputs[j] * weights[j][i]
            weighted_sum += biases[i]
            
            weighted_sums[i] = weighted_sum # 활성화 함수 통과 전 값 저장
            layer_outputs[i] = self._sigmoid(weighted_sum) # 활성화 함수 통과 후 값 저장
            
        return layer_outputs, weighted_sums

    def feedforward(self, inputs):
        """순전파를 통해 입력에 대한 예측을 계산하고, 역전파에 필요한 중간 값들을 반환합니다."""
        # 입력층 -> 은닉층
        hidden_outputs, hidden_weighted_sums = self._calculate_layer_output(inputs, self.weights_input_hidden, self.bias_hidden)

        # 은닉층 -> 출력층
        final_outputs, final_weighted_sums = self._calculate_layer_output(hidden_outputs, self.weights_hidden_output, self.bias_output)
        
        return hidden_outputs, final_outputs, hidden_weighted_sums, final_weighted_sums

    def evaluate(self, data):
        """테스트 데이터로 모델의 정확도를 평가합니다."""
        correct_predictions = 0
        # evaluate 함수에서는 역전파에 필요한 중간 값들이 필요 없으므로,
        # feedforward의 반환 값 중 final_outputs만 사용합니다.
        for row in data:
            label = row[0]
            pixels = [p / 255.0 for p in row[1:]] # 픽셀 값을 0~1 사이로 정규화
            
            # feedforward가 여러 값을 반환하므로, 필요한 값만 받습니다.
            _, outputs, _, _ = self.feedforward(pixels)
            predicted_label = outputs.index(max(outputs))
            
            if predicted_label == label:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(data)
        return accuracy

    def train(self, training_data, epochs, learning_rate):
        """
        신경망을 훈련시킵니다.
        
        Args:
            training_data (list): 훈련 데이터셋. 각 요소는 [레이블, 픽셀1, ...] 형태.
            epochs (int): 전체 훈련 데이터를 반복할 횟수.
            learning_rate (float): 학습률. 가중치 업데이트의 크기를 조절합니다.
        """
        print("\n--- 학습 시작 ---")
        for epoch in range(epochs):
            random.shuffle(training_data) # 각 에포크마다 데이터를 섞어줍니다.

            for row in training_data:
                label = row[0]
                pixels = [p / 255.0 for p in row[1:]] # 픽셀 값을 0~1 사이로 정규화

                # 1. 순전파 (Forward Pass) - 역전파에 필요한 모든 중간 값들을 받습니다.
                hidden_outputs, final_outputs, hidden_weighted_sums, final_weighted_sums = self.feedforward(pixels)

                # 실제 정답을 원-핫 인코딩으로 변환
                target = [0.0] * self.output_size
                target[label] = 1.0

                # 2. 역전파 (Backward Pass)

                # 2-1. 출력층 오류 (delta_output) 계산
                # (target - final_output) * sigmoid_derivative(final_output_value)
                delta_output = [0.0] * self.output_size
                for i in range(self.output_size):
                    error = target[i] - final_outputs[i]
                    delta_output[i] = error * self._sigmoid_derivative(final_outputs[i])

                # 2-2. 은닉층 오류 (delta_hidden) 계산
                # sum(delta_output[k] * weights_hidden_output[j][k]) * sigmoid_derivative(hidden_output_value)
                delta_hidden = [0.0] * self.hidden_size
                for i in range(self.hidden_size):
                    error = 0.0
                    for j in range(self.output_size):
                        error += delta_output[j] * self.weights_hidden_output[i][j] # 은닉층 -> 출력층 가중치 사용
                    delta_hidden[i] = error * self._sigmoid_derivative(hidden_outputs[i])

                # 2-3. 가중치 및 편향 업데이트

                # 은닉층 -> 출력층 가중치 업데이트
                for i in range(self.hidden_size):
                    for j in range(self.output_size):
                        # delta_output[j]는 출력층 j 뉴런의 오류, hidden_outputs[i]는 은닉층 i 뉴런의 활성화 값
                        self.weights_hidden_output[i][j] += learning_rate * delta_output[j] * hidden_outputs[i]
                
                # 출력층 편향 업데이트
                for i in range(self.output_size):
                    self.bias_output[i] += learning_rate * delta_output[i]

                # 입력층 -> 은닉층 가중치 업데이트
                for i in range(self.input_size):
                    for j in range(self.hidden_size):
                        # delta_hidden[j]는 은닉층 j 뉴런의 오류, pixels[i]는 입력층 i 뉴런의 활성화 값
                        self.weights_input_hidden[i][j] += learning_rate * delta_hidden[j] * pixels[i]

                # 은닉층 편향 업데이트
                for i in range(self.hidden_size):
                    self.bias_hidden[i] += learning_rate * delta_hidden[i]
            
            # 각 에포크 종료 후 정확도 평가
            accuracy = self.evaluate(test_set)
            print(f"Epoch {epoch + 1}/{epochs} - 정확도: {accuracy * 100:.2f}%")
        print("--- 학습 완료 ---")

# --- 4. 모델 생성 및 학습 실행 ---
if train_set:
    INPUT_SIZE = 784
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10
    mnist_mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    print("\n학습 시작 전, 무작위 가중치 상태의 모델 성능을 평가합니다...")
    initial_accuracy = mnist_mlp.evaluate(test_set)
    print(f" -> 초기 정확도: {initial_accuracy * 100:.2f}%")

    # 학습 시작
    
    EPOCHS = 10 # 학습 반복 횟수
    LEARNING_RATE = 0.1 # 학습률
    mnist_mlp.train(train_set, EPOCHS, LEARNING_RATE)

    # 학습 후 최종 정확도 평가
    print("\n--- 학습 후 최종 정확도 평가 ---")
    final_accuracy = mnist_mlp.evaluate(test_set)
    print(f" -> 최종 정확도: {final_accuracy * 100:.2f}%")
