import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# 로지스틱 회귀를 모델을 만들어서 실행 한다음 종속변수를 선택한 확률을 보여주는 것

data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')
data_filled = data.fillna(0)
des = data_filled.describe()  # 전처리된 데이터의 통계량 확인
print(des) # 데이터 정보 확인

# 6개 이상의 열이 0인 경우 해당 행 제거
data_filled = data_filled[(data_filled == 0).sum(axis=1) < 6]

array = data_filled.values
X = np.delete(array[:, 1:11], [5], axis=1)
# 5열과 11열은 센서1, 센서6이라 제외
Y = array[:, 12] # 데이터 종속변수와 독립변수 구분

# Y 데이터에서 1과 2를 모두 1로 변경
Y = np.where(Y == 2, 1, Y).astype(int)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=42)
# 모델을 만드는 데이터와 테스트용 데이터를 구분

scaler = MinMaxScaler() # 데이터 스케일 조정
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = LogisticRegression(class_weight={0: 1, 1: 5}) # 가중치 지정
model.fit(X_train_scaled, Y_train)

# 학습결과 확인
y_pred2 = model.predict_proba(X_test_scaled)
y_pred = model.predict(X_test_scaled)
print("예측값", y_pred)
print("예측값을 도출하게 된 각각의 확율", np.round(y_pred2, 5))

# 정확도 계산
accuracy = accuracy_score(Y_test, y_pred)
print("정확도: ", accuracy)

# 오차 행렬 출력
conf_matrix = confusion_matrix(Y_test, y_pred)
print("오차행렬: \n", conf_matrix)

# K-Fold Cross Validation
kfold = KFold(n_splits=5, random_state=420, shuffle=True)
results = cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring='accuracy')
print(f"K-Fold 교차검증 정확도: {results.mean():.4f} ± {results.std():.4f}")