## 농장물 생장 및 수확량 예측 (FarmConnect 기술지원)
본 문서는 Ubuntu 18.04를 기준으로 토마토 생장/수확 예측 프로그램에 대한 환경 구축 및 실행 방법에 대해 서술한다.

## 환경 구축

[Anaconda](https://www.anaconda.com/products/individual) 다운로드
Anaconda 다운로드 후 아래의 명령어들을 실행
~~~
$ sha256sum [your_anaconda_file]
$ bash [your_anaconda_file]
$ source ~/.bashrc
$ conda create --name [your_conda_env_name] python=3.7
$ conda activate [your_conda_env_name]
$ conda install -c conda-forge tensorflow-gpu==2.2.0 pandas numpy numba matplotlib scikit-learn openpyxl xgboost xlrd==1.2.0 tqdm plotly plotly_express joblib
~~~

## 프로그램 다운로드
~~~
$ git clone https://github.com/ETRI-EdgeAnalytics/smartfarm.git
$ cd smartfarm
~~~

## 생장 예측
~~~
$ python main.py --config_path [your_rf_config.json]
~~~

## 수확 예측
~~~
$ python main.py --config_path [your_multiencoder_config.json]
~~~

## 예제

- 생장 예측
~~~
$ python main.py --config_path ./configs/rf.json
~~~
- 수확 예측
~~~
$ python main.py --config_path ./configs/myeong.json # 환경 + 생육 데이터
$ python main.py --config_path ./configs/env_config.json # 환경 데이터
~~~
