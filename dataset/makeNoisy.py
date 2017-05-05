#키보드 오타를 삽입, 교체, 삭제의 형식으로 자동 생성한다
#오타 삽입과 교체 기준은 키보드의 위치이다.
#각 자판의 주변 위치를 list에 저장하고, random 하게 교체 혹은 삽입한다.
import makeWord as m
import csv
import os
import random

#파일 위치
file_location = 'C:/Users/kimhyeji/Desktop/현대문어_원시_말뭉치'
#작업 위치를 변경한다.
os.chdir(file_location)


#한글 음소 분할을 위한 변수 설정
UNICODE_N, CHOSUNG_N, JUNGSUNG_N = 44032, 588, 28
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

#각 자판 주위의 자모음들을 저장하는 사전
KEYBOARD = {}

#각 자판 주위의 자모음들을 저장한다.
def keyboard_order():
    #겹자모음을 이루는 자모음들
    K_double = [{'ㄳ': ['ㄱ', 'ㅅ']}, {'ㄵ': ['ㄴ', 'ㅈ']}, {'ㄶ': ['ㄴ', 'ㅎ']}, {'ㄺ': ['ㄹ', 'ㄱ']}, {'ㄻ': ['ㄹ', 'ㅁ']},
                {'ㄼ': ['ㄹ', 'ㅂ']}, {'ㄽ': ['ㄹ', 'ㅅ']}, {'ㄿ': ['ㄹ', 'ㅍ']}, {'ㅄ': ['ㅂ', 'ㅅ']},
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


#10개의 오타를 제작한다.
#10/총 글자수 각 글자마다 동일한 갯수의 오타 제작
#초성,중성,종성 및 각 글자는 랜덤으로 선택한다.
def make_noisy(w):
    char = list(w)
    #입력 단어를 분리해서 저장
    word_split = []

    # 분리했을 때, 한 글자가 되는 범위
    # ㄱㅏㅂㅏㅇ -> [0,1,4]
    split_index = [0]
    for c in char:
        char_code = ord(c) - UNICODE_N
        #초성 분리
        chosung = int(char_code / CHOSUNG_N)
        word_split.append(CHOSUNG[chosung])

        #종성 분리
        jungsung =  int((char_code - (CHOSUNG_N * chosung)) / JUNGSUNG_N)
        word_split.append(JUNGSUNG[jungsung])

        #종성 분리
        jongsung = int((char_code - (CHOSUNG_N * chosung) - (JUNGSUNG_N * jungsung)))
        #종성이 없는 경우는 더해주지 않는다.
        if(JONGSUNG[jongsung] != ' '):
            word_split.append(JONGSUNG[jongsung])
        split_index.append(len(word_split) -1)

    #분해한 문자의 길이
    word_len = len(word_split)
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
            target_word = w
            error_split= word_split[0:n] + word_split[n+1:]
            error_word = m.combine_word(error_split)
            while (len(error_word) > len(target_word)):
                target_word = target_word[0:i] + '0' + target_word[i:]
            error_word_list.append(error_word)
            target_word_list.append(target_word)

            #자리교체
            if(n != 0):
                target_word = w
                if(n != 0): error_split= word_split[0:n-1] + list(word_split[n]) + list(word_split[n-1]) + word_split[n+1:]
                error_word = m.combine_word(error_split)

                while(len(error_word) > len(target_word)):
                    target_word = target_word[0:i] + '0' + target_word[i:]
                error_word_list.append(error_word)
                target_word_list.append(target_word)


            for near_key in (KEYBOARD[word_split[n]]):
            # 교체
                target_word = w
                error_split= word_split[0:n] + list(near_key) + word_split[n + 1:]
                error_word = m.combine_word(error_split)

                while(len(error_word) > len(target_word)):
                    target_word = target_word[0:i] + '0' + target_word[i:]
                error_word_list.append(error_word)
                target_word_list.append(target_word)
            #추가
                target_word = w
                if(len(near_key) > 1):
                    error_split= word_split[0:n+1] + list(near_key[i%2]) + word_split[n+1:]
                else:
                    error_split= word_split[0:n+1] + list(near_key) + word_split[n+1:]
                error_word = m.combine_word(error_split)

                while(len(error_word) > len(target_word)):
                    target_word = target_word[0:i] + '0' + target_word[i:]

                error_word_list.append(error_word)
                target_word_list.append(target_word)

    #정답단어 추가
    for i in range(5):
        error_word_list.append(w)
        target_word_list.append(w)

    return error_word_list, target_word_list
keyboard_order()
#print(KEYBOARD)

# target list
word_list = []

# input list
word_error_list = []

with open('dic.csv', 'r') as rf, open('dic_modify.csv','w',newline = "\n") as wf:
    r = csv.reader(rf)
    w = csv.writer(wf)

    for row in r:
        word = []

        if(r is None): pass
        word = row[0]
        error , target = make_noisy(word)
        word_list += target
        word_error_list += error
    w.writerow(word_list)
    w.writerow(word_error_list)
