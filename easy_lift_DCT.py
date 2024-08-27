import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# 데이터 로드 및 전처리
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')
data_filled = data.fillna(0)
des = data_filled.describe()  # 전처리된 데이터의 통계량 확인
print(des)  # 데이터 정보 확인

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
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# 학습결과 확인
y_pred2 = model.predict_proba(X_test)
y_pred = model.predict(X_test)
print("예측값", y_pred)
print("예측값을 도출한 확률", y_pred2)

# 정확도 계산
accuracy = accuracy_score(Y_test, y_pred)
print("정확도: ", accuracy)

# 오차 행렬 출력
conf_matrix = confusion_matrix(Y_test, y_pred)
print("오차행렬: \n", conf_matrix)

# K-Fold Cross Validation
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
print(f"K-Fold 교차검증 정확도: {results.mean():.4f} ± {results.std():.4f}")
