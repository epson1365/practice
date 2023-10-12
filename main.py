#필요한 라이브러리를 불러옵니다.
import numpy as np #행렬 계산
import pandas as pd #데이터 처리 계산
import matplotlib.pyplot as plt #그림

from sklearn.model_selection import train_test_split #훈련,테스트 분리
from sklearn.naive_bayes import GaussianNB #머신러닝 방법 중 하나
from sklearn.preprocessing import LabelEncoder,OneHotEncoder #LabelEncoder:문자를 숫자로, OneHotEncoder:숫자들을 분류하기 위해 하나만 1을 넣고 나머지는 0을 넣어 분류하는 방법
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QTextEdit, QLabel, QGridLayout

from sklearn.metrics import confusion_matrix #실제와 예측을 표로 나타낸 것
from sklearn.metrics import accuracy_score, f1_score #정밀도, 재현률

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning) #DeprecationWarning: 함수가 바뀌었는데 옛날 함수를 사용하면 발생하는 오류 무시

#dataset = pd.read_csv('/content/drive/My Drive/파일경로/파일명명')
#dataset = pd.read_csv('/content/drive/My Drive/2-2 학기/방학특강/kddcup.data_10_percent_corrected.csv')
dataset = pd.read_csv(r'C:\Users\Jinwoo\Desktop\3학년2학기\캡스톤\archive_kdd-cup99\archive_kdd-cup99\kddcup.data\kddcup.csv')

#처음 다섯개 데이터를 조회
dataset.head()

#마지막 Label 칼럼 안에 담긴 고유의 카테고리 값을 확인
dataset['normal.'].unique() #중첩되지 않고 사용되는 종류 확인

#nomal을 제외한 모든 카테고리를 attack으로 변경
dataset['normal.'] = dataset['normal.'].replace(['buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.',
       'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.',
       'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.',
       'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
       'spy.', 'rootkit.'],'attack')
dataset['normal.'].unique()

#x = 마지막 label 열을 제외한 모든 열을 특징으로 사용
#y = 마지막 label 열을 카테고리로 사용

x = dataset.iloc[:,:-1].values #iloc위치, .values: 값만 넣어준다.
y = dataset.iloc[:,41].values

print(x.shape,y.shape)

uniq1 = dataset.tcp.unique()
uniq2 = dataset.http.unique()
uniq3 = dataset.SF.unique()

print(uniq1,'\n',uniq2,'\n',uniq3)
print(uniq1.size,'\n',uniq2.size,'\n',uniq3.size)

#tcp(프로토콜), http(서비스), SF(플래그) 열을 One-Hot Encoding
from sklearn.compose import ColumnTransformer #특정 칼럼만 바꾸는 방법

labelencoder_x_1 = LabelEncoder() #LabelEncoder = 문자를 숫자로 변환
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()

x[:,1] = labelencoder_x_1.fit_transform(x[:,1]) #분포 되어있는 통계에 마춰서 문자에서 숫자로 변환
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
x[:,3] = labelencoder_x_3.fit_transform(x[:,3])

#원핫인코딩으로 칼럼수 증가 3 > 80 따라서 총 칼럼의 수 41 > 118
onehotencoder_1 = ColumnTransformer([("tcp",OneHotEncoder(),[1])],remainder='passthrough') #[1]: 시작 위치 지정
onehotencoder_2 = ColumnTransformer([("http",OneHotEncoder(),[4])],remainder='passthrough')
onehotencoder_3 = ColumnTransformer([("SF",OneHotEncoder(),[70])],remainder='passthrough')

x = np.array(onehotencoder_1.fit_transform(x))
x = np.array(onehotencoder_2.fit_transform(x))
x = np.array(onehotencoder_3.fit_transform(x))

print(x.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)

classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
print(y_pred.shape)
print(y_test.shape)

clf = RandomForestClassifier()
clf.fit(x_train,y_train)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        grid = QGridLayout()

        self.btn = QPushButton('분석 시작', self)
        self.btn.clicked.connect(self.analyze)

        self.text_edit = QTextEdit()

        grid.addWidget(self.btn, 0, 0)
        grid.addWidget(self.text_edit, 1, 0, 5, 0)

        self.setLayout(grid)
        self.setWindowTitle('딥러닝 패킷 분석')
        self.show()

    def analyze(self):
        # 여기에서 분석 코드를 실행합니다.
        # 예를 들어:
        dataset = load_and_preprocess_data()
        x_train, x_test, y_train, y_test = split_data(dataset)
        classifier = train_model(x_train, y_train)
        y_pred = classifier.predict(x_test)
        self.text_edit.setText(f"예측 결과:\n{y_pred}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
