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
