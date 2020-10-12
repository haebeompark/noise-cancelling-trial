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
