"""
키보드 오타를 삽입, 교체, 삭제의 형식으로 자동 생성한다
오타 삽입과 교체 기준은 키보드의 위치이다.
각 자판의 주변 위치를 list에 저장하고, random 하게 교체 혹은 삽입한다.

split_data(string)
    string: 단어
    return 단어를 음운으로 분해한 리스트

make_noisy(string)
    string: 단어
    return 단어에 대한 오타리스트,정답리스트 튜플
            오타가 정답단어보다 길어진 경우 정답단어에 0으로 패딩작업을 수행

make_train_data()
    word_list_dir에 있는 단어 리스트를
    save_file에 [오타단어길이, 정답단어길이, 오타글자인덱싱, 정답글자인덱싱] 형식으로 저장한다.
"""

import makeWord as m
import csv
import os
import hangul as hg

# 파일 위치
word_list_dir = 'C:/Users/kimhyeji/Desktop/데이터'
save_file = 'C:/Users/kimhyeji/PycharmProjects/tfTest/dic_modify_.csv'


#한글 음소 분할을 위한 변수 설정
UNICODE_N, CHOSUNG_N, JUNGSUNG_N = 44032, 588, 28
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

#각 자판 주위의 자모음들을 저장하는 사전
KEYBOARD = {}

#예측값과 목표값이 같은 데이터 생성 갯수
#학습시, 옳은 단어를 입력했을 경우, 같은 결과가 나와야 하기 때문
same_word = 1

#각 자판 주위의 자모음들을 저장한다.
def keyboard_order():
    #겹자모음을 이루는 자모음들
    K_double = [{'ㄳ': ['ㄱ', 'ㅅ']}, {'ㄵ': ['ㄴ', 'ㅈ']}, {'ㄶ': ['ㄴ', 'ㅎ']}, {'ㄺ': ['ㄹ', 'ㄱ']}, {'ㄻ': ['ㄹ', 'ㅁ']},
                {'ㄼ': ['ㄹ', 'ㅂ']}, {'ㄽ': ['ㄹ', 'ㅅ']}, {'ㄿ': ['ㄹ', 'ㅍ']},{'ㄾ':['ㄹ','ㅌ']}, {'ㅀ':['ㄹ','ㅎ']},{'ㅄ': ['ㅂ', 'ㅅ']},
                {'ㅘ': ['ㅗ', 'ㅏ']}, {'ㅙ': ['ㅗ', 'ㅐ']}, {'ㅚ': ['ㅗ', 'ㅣ']}, {'ㅝ': ['ㅜ', 'ㅓ']}, {'ㅞ': ['ㅜ', 'ㅔ']},
                {'ㅟ': ['ㅜ', 'ㅣ']}, {'ㅢ': ['ㅡ', 'ㅣ']}]

    #한번의 클릭으로 겹자모음을 이루는 경우에는 오타가 날 수 있는 경우를 다 적어주었다.
    K_double_one = [{'ㄲ': ['ㄸ', 'ㅆ', 'ㄹ']}, {'ㄸ': ['ㄲ', 'ㅉ', 'ㅇ']}, {'ㅃ': ['ㅉ', 'ㅁ', 'ㄴ']}, {'ㅆ': ['ㄲ', 'ㅛ', 'ㅎ']},
                {'ㅉ': ['ㅃ', 'ㄸ', 'ㄴ']},
                {'ㅒ': ['ㅖ', 'ㅑ', 'ㅣ']}, {'ㅖ': ['ㅒ','ㅣ']}]
    #키보드 위치의 자모음
    K = ['ㅂ','ㅈ','ㄷ','ㄱ','ㅅ','ㅛ','ㅕ','ㅑ','ㅐ',
         'ㅁ','ㄴ','ㅇ','ㄹ','ㅎ','ㅗ','ㅓ','ㅏ','ㅣ',
         'ㅋ','ㅌ','ㅊ','ㅍ','ㅠ','ㅜ','ㅡ']
    K_len = len(K)

    # ㅂ : 0, ㅈ : 1 ... 인덱싱 사전
    K_dic = {n:i for i,n in enumerate(K)}
    #키보드 범위를 지정
    comp = lambda a: (a >= 0 and a < len(K))

    #알고리즘 적용을 위해 오른쪽 가장자리의 ㅔ는 예외처리함
    KEYBOARD['ㅔ'] = ['ㅐ','ㅣ']

    #단모음, 단자음
    for c in K:
        KEYBOARD[c] = []
        num = K_dic[c]

        #자판의 주변에 위치한 자판의 모음을 KEYBOARD에 저장한다.
        for locate in [-9,-8,-1,+1,+8,+9,+10]:

            #키보드 가장자리 예외처리
            if ((c in ['ㅐ','ㅣ']) and (locate in [-8,1,10])) or \
                    ((c in ['ㅁ','ㅋ','ㅂ']) and (locate in [-1, 8])):
                continue

            if(comp(num + locate)): KEYBOARD[c].append(K[num + locate])

    #쌍모음, 쌍자음
    # c =  {'ㄳ': ['ㄱ', 'ㅅ']} ...
    for c in K_double:
        #쌍자모음
        #'ㄳ'
        key = list(c.keys())[0]

        KEYBOARD[key] = []

        #쌍자모음의 분해
        #['ㄱ','ㅅ']
        values = list(c.values())

        if(len(values[0]) == 2):
            KEYBOARD[key] += [[i , values[0][1]] for i in KEYBOARD[values[0][0]]]
            KEYBOARD[key] += [[values[0][0], i ] for i in KEYBOARD[values[0][1]]]

    for c in K_double_one:
        key = list(c.keys())[0]
        values = list(c.values())
        KEYBOARD[key] = []

        KEYBOARD[key] += values[0]


def split_word(word):
    """
    단어를 분리한다.
    :param word: 단어
    :return: [ㄷ,ㅏ,ㄴ,ㅇ,ㅓ]
    """
    char = list(word)
    
    #입력 단어를 분리해서 저장
    split_list = []

    # 분리했을 때, 한 글자가 되는 범위
    # ㄱㅏㅂㅏㅇ -> [0,1,4]
    split_index = [0]
    
    for c in char:
        char_code = ord(c) - UNICODE_N
        #초성 분리
        chosung = int(char_code / CHOSUNG_N)
        split_list.append(CHOSUNG[chosung])

        #종성 분리
        jungsung =  int((char_code - (CHOSUNG_N * chosung)) / JUNGSUNG_N)
        split_list.append(JUNGSUNG[jungsung])

        #종성 분리
        jongsung = int((char_code - (CHOSUNG_N * chosung) - (JUNGSUNG_N * jungsung)))
        #종성이 없는 경우는 더해주지 않는다.
        if(JONGSUNG[jongsung] != ' '):
            split_list.append(JONGSUNG[jongsung])
            
        split_index.append(len(split_list) -1)
    return split_list, split_index

def make_noisy(w):
    """
    기준음운의 주변 키보드 음운으로
    가능한 모든 오타를 제작한다.
    초성,중성,종성 및 각 글자는 랜덤으로 선택한다.
    """

    split_list, split_index = split_word(w)

    #분해한 문자의 길이
    word_len = len(split_list)
    error_word_list = []
    target_word_list = []

    len_w = len(w)
    if(len(w) == 1):
        len_w = 2
        split_index[1] = 1

    #글자의 수
    for i in range(len_w - 1):
        #target_word에 0을 삽입하기 위한 범위 지정
        for n in range(split_index[i],split_index[i+1]):
            error_split= []

            #삭제
            #감사 -> 감ㅏ
            target_word = w
            error_split= split_list[0:n] + split_list[n+1:]
            error_word = m.combine_word(error_split)

            #while (len(error_word) > len(target_word)):
            #    target_word = target_word[0:i] + '0' + target_word[i:]
            error_word_list.append(error_word)
            target_word_list.append(target_word)

            #자리교체
            #감사 -> 갓마
            if(n != 0):
                target_word = w
                if(n != 0): error_split= split_list[0:n-1] + list(split_list[n]) + list(split_list[n-1]) + split_list[n+1:]
                error_word = m.combine_word(error_split)

                #while(len(error_word) > len(target_word)):
                #    target_word = target_word[0:i] + '0' + target_word[i:]
                error_word_list.append(error_word)
                target_word_list.append(target_word)

            for near_key in (KEYBOARD[split_list[n]]):
            # 교체
            #감사 -> 감하
                target_word = w
                error_split= split_list[0:n] + list(near_key) + split_list[n + 1:]
                error_word = m.combine_word(error_split)

                #while(len(error_word) > len(target_word)):
                #    target_word = target_word[0:i] + '0' + target_word[i:]
                error_word_list.append(error_word)
                target_word_list.append(target_word)
            #추가
            #감사 -> 감ㅎ사
                target_word = w
                if(len(near_key) > 1):
                    error_split= split_list[0:n+1] + list(near_key[i%2]) + split_list[n+1:]
                else:
                    error_split= split_list[0:n+1] + list(near_key) + split_list[n+1:]
                error_word = m.combine_word(error_split)

                #while(len(error_word) > len(target_word)):
                #    target_word = target_word[0:i] + '0' + target_word[i:]

                error_word_list.append(error_word)
                target_word_list.append(target_word)

    #오타에 목표단어를 그대로 추가
    for i in range(same_word):
        error_word_list.append(w)
        target_word_list.append(w)

    return error_word_list, target_word_list

def make_train_data():
    """
    단어 리스트에서 오타를 생성해,
    [오타길이, 정답길이, 오타인덱싱, 정답인덱싱]
    목록을 반환한다.
    """

    # 작업 위치를 변경한다.
    os.chdir(word_list_dir)

    #오타 생성 시 필요한 키보드 배열
    keyboard_order()
    #print(KEYBOARD)

    # target list
    word_list = []
    # input list
    word_error_list = []


    with open('dic.csv', 'r') as rf, open(save_file,'w',newline = "\n",encoding='utf-8') as wf:
        #모든 한글 음절 인덱싱 사전
        #ㄱ : 1 , ㄲ : 2 ...
        index_dic = hg.char_dic

        r = csv.reader(rf)
        w = csv.writer(wf)

        len_all_data = 0
        len_data = 0
        for row in r:
            word = []

            if(row is None): continue
            word = row[0]
            if(int(row[1]) < 50): break
            if(len(word) > 7): continue

            try:
                errors , targets = make_noisy(word)

                len_data += 1
                len_all_data += len(targets)
                word_list += [[int(index_dic[t]) for t in target] for target in targets]
                word_error_list += [[int(index_dic[e]) for e in error] for error in errors]
            except:
                print("error")
                print(word)
        """
        #단어의 최대 길이 구하기
        max = 0
        for i in range(len(word_list)):
            if (max < len(word_error_list[i])): m
            ax = len(word_error_list[i])
            if (max < len(word_list[i])): max = len(word_list[i])
        """
        max=7
        #데이터 생성, 저장
        #error 단어 길이, target 단어 길이, error 단어, target 단어
        for i in range(len(word_list)):
            if(len(word_error_list[i]) > 7):
                print("long")
                continue
            lists = [len(word_error_list[i]), len(word_list[i])]
            lists += word_error_list[i]
            for _ in range(max - len(word_error_list[i])): lists.append(0)
            lists += word_list[i]
            for _ in range(max - len(word_list[i])): lists.append(0)
            w.writerow(lists)

        #총 오타 수
        print(len_all_data)
        print(len_data)


#학습 시 , 반드시 주석처리해주어야함
#반! 드! 시
make_train_data()
