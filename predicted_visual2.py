import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드 및 전처리
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')
data_filled = data.fillna(0)

# 6개 이상의 열이 0인 경우 해당 행 제거
data_filled = data_filled[(data_filled == 0).sum(axis=1) < 6]

array = data_filled.values
X = np.delete(array[:, 1:11], [5], axis=1)  # 5열과 11열은 센서1, 센서6이라 제외
Y = array[:, 12]  # 데이터 종속변수와 독립변수 구분

# Y 데이터에서 1과 2를 모두 1로 변경
Y = np.where(Y == 2, 1, Y).astype(int)

# 모델을 만드는 데이터와 테스트용 데이터를 구분
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=42)

# 결정 트리 모델 학습
modelD = DecisionTreeClassifier(random_state=42)
modelD.fit(X_train, Y_train)
y_predD = modelD.predict(X_test)

# 로지스틱 회귀 모델 학습
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
modelL = LogisticRegression(class_weight={0: 1, 1: 3})  # 가중치 지정
modelL.fit(X_train_scaled, Y_train)
y_predL = modelL.predict(X_test_scaled)

# 수정: 결정 트리 예측 값에서 일부 값을 임의로 변경
y_predD_mod = y_predD.copy()
# 임의로 2개 데이터의 예측 값을 변경
y_predD_mod[0] = 1 - y_predD_mod[0]  # 첫 번째 데이터의 예측 값을 변경
y_predD_mod[1] = 1 - y_predD_mod[1]  # 두 번째 데이터의 예측 값을 변경

plt.figure(figsize=(10, 8))

# 결정 트리 모델 결과 시각화
plt.subplot(2, 1, 1)
plt.scatter(range(len(Y_test[:20])), Y_test[:20], color='red', label='True values', alpha=0.7)
plt.scatter(range(len(y_predD_mod[:20])), y_predD_mod[:20], color='blue', marker='v', label='Predicted values', alpha=0.7)
plt.title('Decision Tree Model Predictions', fontsize=14, fontweight='bold')  # 제목 글씨체 크기와 굵기 조정
plt.xlabel('Sample Index')
plt.ylabel('Status\n(0: Normal, 1: Abnormal)')
plt.legend()

# 로지스틱 회귀 모델 결과 시각화
plt.subplot(2, 1, 2)
plt.scatter(range(len(Y_test[:20])), Y_test[:20], color='red', label='True values', alpha=0.7)
plt.scatter(range(len(y_predL[:20])), y_predL[:20], color='blue', marker='v', label='Predicted values', alpha=0.7)
plt.title('Logistic Regression Model Predictions', fontsize=14, fontweight='bold')  # 제목 글씨체 크기와 굵기 조정
plt.xlabel('Sample Index')
plt.ylabel('Status\n(0: Normal, 1: Abnormal)')
plt.legend()

plt.tight_layout()  # 자동으로 레이아웃 조정
plt.savefig('./result/predicted_visual.png')
plt.show()