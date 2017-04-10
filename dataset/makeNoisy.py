#키보드 오타를 삽입, 교체, 삭제의 형식으로 자동 생성한다
#오타 삽입과 교체 기준은 키보드의 위치이다.
#각 자판의 주변 위치를 list에 저장하고, random 하게 교체 혹은 삽입한다.

import csv
import os

#파일 위치
file_location = 'C:/Users/kimhyeji/Desktop/현대문어_원시_말뭉치'
#작업 위치를 변경한다.
os.chdir(file_location)


#한글 음소 분할을 위한 변수 설정
UNICODE_N, CHOSUNG_N, JUNGSUNG_N = 44032, 588, 28
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

KEYBOARD = {}

#각 자판 주위의 자음들을 저장한다.
def keyboard_order_jaum():
    #쌍자음을 이루는 자음들
    K_double = [{'ㄲ': ['ㄱ']}, {'ㄸ': ['ㄷ']}, {'ㅃ': ['ㅂ']}, {'ㅆ': ['ㅅ']}, {'ㅉ': ['ㅈ']},
                {'ㄳ': ['ㄱ', 'ㅅ']}, {'ㄵ': ['ㄴ', 'ㅈ']}, {'ㄶ': ['ㄴ', 'ㅎ']}, {'ㄺ': ['ㄹ', 'ㄱ']}, {'ㄻ': ['ㄹ', 'ㅁ']},
                {'ㄼ': ['ㄹ', 'ㅂ']}, {'ㄽ': ['ㄹ', 'ㅅ']}, {'ㄿ': ['ㄹ', 'ㅍ']}, {'ㅄ': ['ㅂ', 'ㅅ']},
                {'ㅒ': ['ㅐ']}, {'ㅖ': ['ㅔ']},
                {'ㅘ': ['ㅗ', 'ㅏ']}, {'ㅙ': ['ㅗ', 'ㅐ']}, {'ㅚ': ['ㅗ', 'ㅣ']}, {'ㅝ': ['ㅜ', 'ㅓ']}, {'ㅞ': ['ㅜ', 'ㅔ']},
                {'ㅟ': ['ㅜ', 'ㅣ']}, {'ㅢ': ['ㅡ', 'ㅣ']}]
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
    for c in K_double:
        key = list(c.keys())[0]
        KEYBOARD[key] = []
        values = list(c.values())
        for v in values[0]:
            KEYBOARD[key] += KEYBOARD[v]

#10개의 오타를 제작한다.
#총 글자/10 각 글자마다 동일한 갯수의 오타 제작
#초성,중성,종성 및 각 글자는 랜덤으로 선택한다.
def make_noisy(w):
    char = list(w)
    for c in char:
        word_split = []
        char_code = ord(c) - UNICODE_N
        chosung = int(char_code / CHOSUNG_N)
        word_split.append(CHOSUNG[chosung])
        jungsung =  int((char_code - (CHOSUNG_N * chosung)) / JUNGSUNG_N)
        word_split.append(JUNGSUNG[jungsung])
        jongsung = int((char_code - (CHOSUNG_N * chosung) - (JUNGSUNG_N * jungsung)))
        word_split.append(JONGSUNG[jongsung])
        print("".join(word_split))
    return word_split

"""

with open('dic.csv', 'r') as rf, open('dic_modify','w',newline = "\n") as wf:
    r = csv.reader(rf)
    w = csv.writer(wf)

    for row in r:
        word = []
        if(r is None): pass
        word = row[0]
        make_noisy(word)

"""

keyboard_order_jaum()
print(KEYBOARD)