[CIC-Darknet 2020](https://www.unb.ca/cic/datasets/darknet2020.html) 데이터셋을 이용한 트래픽 분류 프로젝트입니다.

<br>

### 1. 데이터셋 구성

데이터셋은 83개의 피처와 4개의 트래픽 유형과 8가지 어플리케이션 카테고리로  라벨링 돼 있습니다. 피처 설명은 [여기로](https://www.kaggle.com/datasets/peterfriedrich1/cicdarknet2020-internet-traffic)

<img src="https://user-images.githubusercontent.com/76269316/169469127-6406eaf3-99f9-487d-bf4d-8d61f65f7a50.png" alt="스크린샷 2022-05-20 오후 3 45 08" style="zoom:67%;" />

4개의 트래픽 유형 중 Non-Tor, Non-VPN에 해당하는 어플리케이션은 Benign Traffic이고, Tor-VPN에 해당하는 어플리케이션은 Darknet Traffic입니다.

같은 데이터셋을 사용한 다른 논문들은 Benign, Darknet을 분류하는 다크넷 트래픽 탐지 과정과 어플리케이션을 분류하는 트래픽 분류 과정을 나눠서 진행했는데 Benign 트래픽이건, Darknet 트래픽이건 실제로는 모두 같이 흐르기 때문에 이를 한 번에 분류하기로 하였습니다.

같은 데이터셋을 사용한 다른 논문들

- [DarkDetect: Darknet Traffic Detection and Categorization Using Modified Convolution-Long Short-Term Memory](https://ieeexplore.ieee.org/document/9514531)
- [Darknet Traffic Classification using Machine Learning Techniques](https://ieeexplore.ieee.org/document/9493386)

따라서 저는 다음과 같이 8개의 어플리케이션 카테고리를 16개의 어플리케이션 카테고리로 변환한 뒤 분류를 진행했습니다.

<img src="https://user-images.githubusercontent.com/76269316/169467808-c0e7cffd-a0dd-44e5-8625-7ff1e89564ad.png" alt="스크린샷 2022-05-20 오후 3 36 05" style="zoom: 67%;" />

<br>

### 2. 데이터 전처리

라벨링을 변환한 뒤, 불필요하다고 판단한 피처를 삭제했습니다. 

```python
drop_columns = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']
traffic_df.rename(columns={'Label.1':'target'}, inplace=True)
traffic_df.drop(drop_columns, axis=1, inplace=True)
```

Flow ID의 경우 각 샘플의 고유 ID를 나타내는 피처인데, 분류를 진행하는데 있어서 불필요하다고 판단돼 제거했고, 마찬가지로 Source IP 주소, Destination IP 주소, 트래픽을 보낸 시간인 Timestamp 피처를 제거했습니다.

또 데이터에 있는 Null값과 Infinity 값을 제거했습니다.

```python
traffic_df = traffic_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
```

<br>

추가적으로 데이터 분포의 왜곡으로 인한 성능 저하를 막기 위해 로그 변환을 적용했습니다.

```python
columns = []

columns += list(traffic_df.select_dtypes(['float16']).columns)
columns += list(traffic_df.select_dtypes(['float32']).columns)
columns += list(traffic_df.select_dtypes(['float64']).columns)


for column in columns:
    if column == 'target':
        continue
    traffic_df[column] = np.log1p(traffic_df[column])
```

<br>

이후 LabelEncoder를 사용하여 16개의 클래스를 인코딩 했습니다.

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

traffic_df['target'] = encoder.fit_transform(traffic_df['target'])
```

<img src="https://user-images.githubusercontent.com/76269316/169633073-04973332-20bf-4611-bbd1-ee144435faa0.png" alt="image" style="zoom:50%;" />

<br>

클래스 불균형이 심하기 때문에, 각 클래스별 분포를 유지하도록 훈련, 검증, 테스트 데이터셋을 나눴습니다.

```python
from sklearn.model_selection import train_test_split

y_traffic_df = traffic_df['target']
X_traffic_df = traffic_df.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_traffic_df, y_traffic_df, stratify=y_traffic_df, test_size=0.4, random_state=11)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

![image](https://user-images.githubusercontent.com/76269316/169633164-450001bc-1188-40d0-b3fb-398c6560b3fb.png)

<br>

```python
X_vali, X_test, y_vali, y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5, random_state=11)
print(X_vali.shape, y_vali.shape)
print(X_test.shape, y_test.shape)
```

![image](https://user-images.githubusercontent.com/76269316/169633180-3694cd7e-678f-4fc3-84bf-950d2f8305cf.png)

<br>

### 3. 모델 성능 확인 (Valid) - 1

LightGBM을 사용하여 전처리만 거친 데이터셋을 사용해 검증 데이터셋을 예측해봤습니다.

LightGBM을 사용한 이유는 균형 트리 분할(Level Wise Tree Growth) 방식을 사용하는 XGBoost보다 학습 시간 및 예측 수행 시간이 더 빠르고, 메모리 사용량이 적다는 장점이 있어 분류 모델로 채택하였습니다.

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

lgbm_wrapper = LGBMClassifier(random_state=11, n_estimators=500, boost_from_average=False)
lgbm_wrapper.fit(X_train, y_train)
lgbm_pred = lgbm_wrapper.predict(X_vali)

print('{0} 정확도:{1:.4f}'.format(lgbm_wrapper.__class__.__name__, accuracy_score(y_vali, lgbm_pred)))
print(confusion_matrix(y_vali, lgbm_pred), '\n')
print(classification_report(y_vali, lgbm_pred, target_names=label_name), '\n')
```

![image](https://user-images.githubusercontent.com/76269316/169633294-20682596-46bd-4f4e-868e-741c22ce6eeb.png)

Scikit-Learn의 classification report를 사용해 출력한 결과인데 클래스가 많음에도 대체적으로 준수한 성능을 확인할 수 있습니다.

<br>

### 4. 피처 선택

피처가 79개로 너무 많아 학습 시간도 오래걸릴 뿐더러, 많은 피처 수는 과적합을 유발할 수 있기 때문에 feature selection을 사용하여 피처를 선택해 사용했습니다.

Scikit-Learn의 SelectFromModel은 피처 중요도를 계산해 중요도가 임곗값보다 높은 피처를 선택해 주는 메소드입니다. 임곗값으로는 평균을 사용했으며, 총 29개의 피처가 선택됐습니다.

```python
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

select = SelectFromModel(LGBMClassifier(), threshold="mean")
select.fit(X_train, y_train)
```



![feature importance](https://user-images.githubusercontent.com/76269316/169633578-d40f0763-9154-42ec-a994-79241e7120b7.png)

<br>

### 5. 모델 성능 확인 (Valid) - 2

위에서 선택한 29개의 피처만 가지고 모델을 테스트 해봤습니다.

```python
drop_columns = [2, 5, 8, 9, 10, 12, 13, 14, 15, 18, 19, 23, 24, 25, 28, 32, 33, 34, 35, 37, 40, 41, 42, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 69, 70, 71, 72, 73, 74, 76]
traffic_df.drop(traffic_df.columns[drop_columns], axis=1 ,inplace=True)
```

![image](https://user-images.githubusercontent.com/76269316/169633680-50464d08-1897-4e07-99b1-0578126f2f42.png)

<br>

![image](https://user-images.githubusercontent.com/76269316/169633804-6f2ec71b-82ed-4e75-9653-91804c9903a1.png)

79개의 피처를 사용할 때보다 성능이 약간 떨어졌지만 피처를 50개나 줄인 것을 생각하면  나쁘지는 않은 것 같습니다.

<br>

### 6. 하이퍼 파라미터 튜닝

Optuna는 하이퍼 파라미터 최적화 프레임워크로 파라미터의 범위를 지정해주면 측정 지표에 따라 하이퍼 파라미터를 변경하면서 최적의 파라미터를 찾아줍니다. 측정 지표를 F1 스코어로 설정해 하이퍼 파라미터를 튜닝했습니다.

사용한 하이퍼 파라미터는 다음과 같습니다.

<img src="https://user-images.githubusercontent.com/76269316/169633842-bdabf8c3-cb4a-4a6f-876e-ff9e692c6449.png" alt="optuna" style="zoom:50%;" />

<br>

### 7. 모델 성능 확인 (Valid) - 3

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

lgbm_wrapper = LGBMClassifier(random_state=11, max_depth=10, learning_rate=0.046690609202960004, n_estimators=1000, colsample_bytree=0.5449634467189419, colsample_bylevel=0.6170260924043028, colsample_bynode=0.5982217573905297, reg_lambda=0.02395878521182272, reg_alpha=0.6772588890679594, subsample=0.85, min_child_weight=2, gamma=0.32949337881908053, boost_from_average=False)
lgbm_wrapper.fit(X_train, y_train)
lgbm_pred = lgbm_wrapper.predict(X_vali)

print('{0} 정확도:{1:.4f}'.format(lgbm_wrapper.__class__.__name__, accuracy_score(y_vali, lgbm_pred)))
print(confusion_matrix(y_vali, lgbm_pred), '\n')
print(classification_report(y_vali, lgbm_pred, target_names=label_name), '\n')
```

![image](https://user-images.githubusercontent.com/76269316/169633926-b26e37a8-be6c-4945-be91-1b16be459c97.png)

<br>

하이퍼 파라미터를 튜닝한 LightGBM 모델과 하지 않은 모델의 성능을 비교해봤습니다.

![image](https://user-images.githubusercontent.com/76269316/169633966-d7b4899f-c31d-4a03-975e-c7860beccd7e.png)

조금 더 잘 예측한 클래스도 있지만, 성능이 떨어진 클래스도 있습니다. 그래도 대체적으로 성능이 약간씩 향상 됐습니다.

<br>

### 8. 모델 성능 

최종적으로 테스트 데이터셋을 사용하여 모델의 성능을 비교해봤습니다.

![f1 score](https://user-images.githubusercontent.com/76269316/169634013-244bdd3b-175a-4048-9072-072ef2bb05b7.png)

79개의 피처를 사용했을 때와 약간의 성능 차이는 있지만, 비슷한 분류 성능을 보이는 것을 확인할 수 있었습니다.

<br>

### 9. 결론

일부 클래스들의 경우 F1 Score가 상대적으로 낮은 것을 확인할 수 있는데, 이는 클래스 분포가 균일하지 않고 불균형한 클래스 불균형 문제(Class Imbalance Problem) 및 피처 값이 유사하여 서로 다른 클래스가 같은 피처 공간(Feature Space)에 혼재돼 발생하는 클래스 중첩 문제(Class Overlap Problem)가 발생하기 때문입니다.

➕ [ROCT: Radius-based Class Overlap Cleaning Technique to Alleviate the Class Overlap Problem in Software Defect Prediction](https://ieeexplore.ieee.org/document/9529661) 논문에 따르면 클래스 불균형 문제가 항상 모델 성능을 하락시키지는 않는다고 합니다.

![image](https://user-images.githubusercontent.com/76269316/169634399-cef06dd2-758a-4d19-a620-c25527a8e668.png)

위 그림은 논문에서 해당 논문에서 가져온 예시인데, 왼쪽 이미지에서  Non-defective 클래스와 Defective 클래스의 분포가 불균형하지만 두 클래스는 피처 공간 내에서 완벽하게 분리돼 있기 때문에 이 경우에는 모델의 성능을 하락시키지 않는다고 합니다.

하지만 오른쪽 그림처럼 클래스 중첩 문제는 두 클래스가 같은 피처 공간에 섞여있기 때문에 두 클래스를 분리하는 것이 어렵다고 합니다.

<br>

아래 그림은 t-SNE(t-Distributed Stochastic Neighbor Embedding) 기법을 사용하여 Benign_Voip, Darknet_Voip 클래스를 2차원으로 축소한 뒤 시각화한 모습입니다.

![t-sne](https://user-images.githubusercontent.com/76269316/169634105-b86ffdba-f140-418c-b9db-6d8fad900f3e.png)

두 클래스는 어플리케이션 종류가 같기 때문에 피처 값도 유사한 값을 갖게 됩니다. 그럴 경우 위와 같이 클래스 중첩 문제가 발생해 두 클래스를 구분하는 것을 어렵게 만듧니다.

사실 이 클래스 중첩 문제를 해결하는데 초점을 두고 연구를 진행했는데, Radius-based Class Overlap Cleaning Technique이나 Conditional GAN을 이용한 오버샘플링 기법등 여러 기법을 적용해봤지만 성능 향상이 없어서 이 정도로 마무리하고 학부 논문을 작성했습니다.

추후 클래스 중첩 문제를 해결해 모든 클래스를 잘 분류하도록 하는 것이 목표입니다.
