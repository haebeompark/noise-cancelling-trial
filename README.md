# noise-cancelling-trial

## 개요
  ### base
  
  - noise가 가미된 음원파일로부터 목소리를 추출
  - deep Learning을 사용할 것임 : 아마 Logistic regression, 필요하다면 다른방법으로 할 예정
  
  
  ### advance
  
  - 여러 목소리가 있다면 목소리별로 화자를 분리, 배열로 목소리를 개별적으로 추출
  - 웹사이트로서 서비스 : tutorial한 ruby-on-rails를 사용하거나 python flask서버 사용예정, 추후 사용자의 동의하에 Input 데이터 수집-> 제대로 분리가 되었나요? 예 / 아니요
  - GAN사용
  
## 사전준비
  ### 데이터 셋 확보
  [무료 noise+voice set 3900개](https://iqtlabs.github.io/voices/) ->[깃허브](https://github.com/Lab41/VOiCES-subset)에서 -> https://app.box.com/s/55wzvodkxiwllncnh1bkk147zbxm4l68
  
  ### 전처리
  잡음제거등의 소리관련 프로그램을 이용할 수 있고, 일정길이마다 잘라서 처리 할 수 있다.

## 계획
  ### 데이터 전처리
  - 길이가 서로 다른 데이터들 때문에 wav를 일정길이마다 잘라 처리한다. -> trainSet은 잡음이 추가된 음원과 제거된 음원으로 제공 되어야함.
  
  ----
  신경망을 본격적으로 구축하기 이전에 wav파일을 읽고 쓸 수 있도록 한다. 신경망의 원할한 학습을 위해 단위시간 cutting을 해줌.
  
  ### 신경망
  - 원시적인 logistic regression을 사용해본다. -> 예상되는 문제점 : 학습이 매우 느릴것으로 예상된다. input과 output의 크기가 커 overfitting이 우려됨.
  - resnet18 ~ 200 -> 예상되는 문제점 : 속도, 잡음이 제대로 제거되지 않을 수 있고 무엇보다 output의 수가 많은경우 적절치 않다.
  - GAN : resnet은 판별자, 위조자로서 logistic 회귀를 사용 - 위조자는 noised파일에서 잡음을 없앤 파일을 위조한다. 판별자는 위조된 음원을 판별한다.
  
  ----
  우선 위의 신경망을 전부 순서대로 구축하고 data를 넣어 시험해보는 것이 계획
  
  ### 시각화
  - 머신러닝의 꽃 시각화 - 음원, 잘라낸 음원의 파동을 시각화 하며, 시간에 따른 에너지 크기별 색칠을 할 계획. 
  
## 참고자료, 참고한 블로그들
https://goudacheese.gitlab.io/project/project1/
http://keunwoochoi.blogspot.com/2016/03/2.html
https://m.blog.naver.com/PostView.nhn?blogId=takion7&logNo=221660937266&proxyReferer=http:%2F%2Fm.blog.naver.com%2F - 데이터 전처리관련
https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html - wav파일읽기 numpy method
https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array
https://stackoverflow.com/questions/51079048/scipy-io-wavfile-write-no-sound
https://banana-media-lab.tistory.com/entry/Librosa-python-library%EB%A1%9C-%EC%9D%8C%EC%84%B1%ED%8C%8C%EC%9D%BC-%EB%B6%84%EC%84%9D%ED%95%98%EA%B8%B0
https://greeksharifa.github.io/%ED%8C%8C%EC%9D%B4%EC%8D%AC/2019/12/16/numba/ - Numba
https://gurujung.github.io/dev/numba_user_jit/ -Numba

https://yonghyuc.wordpress.com/2019/08/06/pytorch-cuda-gpu-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/ - pyTorch
https://mc.ai/pytorch%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%95%98%EA%B8%B0-cnn/ - pyTorch


## 실행
main.py를 실행해서 terminal에서 아래 문자열을 입력하며 진행한다.
### create Simple data

```
createDataSet train 1000
```
0~1값을 갖는 train set X가 1000개 생성되고 X의 Sin값으로 Y를 생성한다.

```
createDataSet test 100
```
test set으로서 생성한다.

### autoBuild
```
autoBuild
```
learning rate와 hiddenLayer수를 적절히 설정하도록 하는 builder
무작위성 탐색과 느린반복 때문에 실행할 때 마다 다른값을 가지며 시간이 오래걸린다.
-> 이미 최적의 lr과 hiddenLayer를 찾았으면 코드를 수정해서 그 값으로 Build하면 됨.

>> 너무느려서 개선필요함

### loadDataSet
```
loadDataSet train 1000
```
wav파일을 sampling한 것을 X, vectorization을 한 값을 Y로 trainSet을 생성한다.

```
loadDataSet test 100
```
testSet으로서 load

### model
```
model train
```
build된 신경망으로 학습진행
정확도를 내놓는다.

```
model test
```
학습된 신경망으로부터 테스트
testSet의 정확도를 내놓는다.