# Okey_spellCheck
Existing error correction algorithms of smartphone keyboard, 
which is widely used in smartphone use, require complicated Korean grammar. 
Also, the analytical approach is difficult to determine the ambiguity of the context.
Due to the nature of the language, the developer must constantly modify things that change over time.
We apply a machine learning algorithm to easily modify the language change by building one model.
Based on the existing smartphone keyboard, this project is complemented and added to the three major parts of typing text correction and correction,
word automatic recommendation function, automatic spacing function.

다양한 연령대의 사용자가 스마트폰의 메신저 애플리케이션 및 SNS 서비스 등을 이용하게 되었음에도
오타 수정과 띄어쓰기 등에서 느끼는 사용자들의 불편함은 여전히 크다.
수 많은 키보드 관련 애플리케이션들은 정확도 미미, 사용 불편 등의 이유로 이용이 많지 않다.

오타 수정이나 띄어쓰기 교정기의 제작에 있어서도
문법 지식이 선행되어야 하며, 유지 보수 또한 어려운 편이다.

이러한 단점들을 해결하고자 본 프로젝트에서는 딥러닝을 접목한 단어 자동 완성 및 오타 수정과 
통계적 확률 기반 모델을 사용한 자동 띄어쓰기 기술을 사용하여
데이터를 기반으로 하는 
안드로이드 용 스마트 키보드를 개발하는 것을 목표로 한다.

https://youtu.be/g3tAj8hETl4



## dic_modify.csv
  훈련 데이터 생성 결과 파일

## hangul.py
  한글 음운/ 음절 저장

    char_arr   한글 음운 리스트
    char_dic   한글 음절 인덱싱 사전

## makeNoisy.py
  오타 생성기

  기존 단어를 이용하여 키보드 오타를 자동 생성한다.
  오타 삽입과 교체 기준은 키보드의 위치이다.
  각 자판의 주변 위치를 list에 저장하고, random 하게 교체 혹은 삽입한다.

    split_data(string)  단어를 음운으로 분해한다.
    make_noisy(string)  단어에 대한 오타 데이터를 제작한다.
    make_train_data()   단어리스트로 훈련데이터를 제작한다. 
                        훈련 데이터 형식은 다음과 같다.
                        [오타단어길이, 정답단어길이, 오타글자인덱싱, 정답글자인덱싱]

## makeTrie.py
  Trie 형태로 기존 단어들을 저장한다. 실 사용시 더 높은 정확성을 위해, 이 파일에서 제작한 json 파일에서 단어가 있는 지 검사한 후 없는 경우에만 오타 교정기를 실행한다.

## makeWord.py
  분해된 자소를 단어로 합친다. 한글 두벌 키보드 입력을 바탕으로 오토마타를 제작하였으며 오타 제작 시 필요하다.
  
    combine_word(word) 단어를 통합한다

## recoverWord.py
  Indexing 형태 <-> 한글. 오타 교정기를 돌리기 전과 후에 실행시킨다.
  
    recover_word(list) tensorflow 작업 결과물을 단어로 변환
    convert_num(string) 단어를 tensorflow input data로 변환

## spell_check_tensorflow.py
  sequence to sequence 기반의 deep learning Korean spell check model
  
  
  ### class SmallConfig
      hyperparameter setting
    
  ### class Seq2SeqModel
      __init__(self, batch_size=config.batch_size,epoch=config.epoch,
                   bidirectional=True,
                   attention=False)
      _init_placeholders(self)  입력 데이터를 저장할 공간을 초기화 한다.
      _init_embeddings(self)  embedding matrix를생성해서 훈련할 값을 벡터 임베딩한다
      
      encoder : 소스언어 정보를 압축한다
      _init_simple_encoder(self)  순방향 훈련 encoder(bidirectional이 아닌 경우)
      _init_bidirectional_encoder(self) 단어의 역방향으로도 훈련하기 위한 bidirectional encoder
      
      decoder : encoder의 압축 정보를 받아 결과로 변환한다
      _init_decoder(self)
      _init_decoder_train_connectors(self)
      
      _init_optimizer(self)  adam optimizer를사용해 손실을 최소화한다
      
      feed_dict에 입력할 형태
      make_train_inputs(self, inputs_length_, targets_length_, inputs_, targets_ )
      make_inference_inputs(self, inputs_length_, inputs_)
      
      read_data(self, file_name)
      read_data_batch(self,tensors)
      
   train_on_copy_task_(session, model,
                        len_x,len_y,x,y,
                        initial_step = 0,
                       verbose=True)
