import pandas as pd
import matplotlib.pyplot as plt

# 엑셀 파일 불러오기
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')

# 종속변수 'Status'에서 1과 2를 묶어 비정상으로 설정 (0은 정상)
data['Status'] = data['Status'].apply(lambda x: 1 if x in [1, 2] else 0)

# 독립변수 목록 설정
independent_vars = ['Temperature', 'Humidity', 'RPM', 'Vibrations', 'Pressure', 'Sensor2', 'Sensor3', 'Sensor4',
                    'Sensor5']

# 밀도 플롯 생성
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 13))

for i, var in enumerate(independent_vars):
    ax = axes[i // 3, i % 3]
    for status in data['Status'].unique():
        subset = data[data['Status'] == status]
        subset[var].plot(kind='density', ax=ax, label=f'Status {status}')

    # 그래프 제목에서 "Density Plot of" 제거하고 변수명만 남김
    ax.set_title(var, fontsize=14, fontweight='bold')

    # x축 레이블 설정 (글꼴 크기 작게)
    ax.set_xlabel(var, fontsize=10)

    # y축 레이블 설정 (글꼴 크기 작게)
    ax.set_ylabel('Density', fontsize=10)

    # 범례 설정 (폰트 크기 줄이기)
    ax.legend(fontsize=8)

    # x축과 y축 눈금의 글꼴 크기 설정
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# 서브플롯 간격 조정
plt.subplots_adjust(hspace=0.6, wspace=0.4)  # 상하 간격(hspace) 및 좌우 간격(wspace) 설정

plt.savefig('./result/density_plot_selected.png')
plt.show()