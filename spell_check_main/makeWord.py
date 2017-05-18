#import time

#start = time.time()

# 이 프로그램은 분해된 자소를 한글 두벌 키보드 입력에 따라 합치는 프로그램이다.
# 입력값 예시 ['ㄱ','ㅏ','ㅇ','ㄹ','ㄱ','ㅏ']
# 출력값 예시 '강ㄹ가'
# 입력값에는 'ㅢ','ㅡ','ㅣ','ㄺ','ㄹ','ㄱ', 처럼 모든 형태의 자모음이 올 수 있음을 가정했다.
#키보드 오토마타를 제작한 후, 코드로 작성하였다.

MOUM = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JAUM = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ','ㄸ','ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ','ㅃ','ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ','ㅉ','ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


#키보드에서 두번을 눌러야 제작할 수 있는 겹자음을
#이 문서에서는 키보드겹자음이라고 칭한다
double_j = ['ㄳ','ㄵ','ㄶ','ㄺ','ㄻ','ㄼ','ㄽ','ㄿ','ㅄ']
double_j_list = [['ㄱ', 'ㅅ'], ['ㄴ', 'ㅈ'], ['ㄴ', 'ㅎ'], ['ㄹ', 'ㄱ'],  ['ㄹ', 'ㅁ'],
                ['ㄹ', 'ㅂ'], ['ㄹ', 'ㅅ'],['ㄹ', 'ㅍ'],['ㅂ', 'ㅅ']]

#키보드에서 두번을 눌러야 제작할 수 있는 겹모음을
#이 문서에서는 키보드겹모음이라고 칭한다
double_m = ['ㅘ','ㅙ','ㅚ','ㅝ','ㅞ','ㅟ','ㅢ']
double_m_list = [['ㅗ', 'ㅏ'], ['ㅗ', 'ㅐ'], ['ㅗ', 'ㅣ'], ['ㅜ', 'ㅓ'],  ['ㅜ', 'ㅔ'],
                ['ㅜ', 'ㅣ'], ['ㅡ', 'ㅣ']]

# 'ㄳ' : ['ㄱ','ㅅ']
K_double_j = {double_j[i] : double_j_list[i] for i in range(len(double_j))}
K_double_m = {double_m[i] : double_m_list[i] for i in range(len(double_m))}


#한글 음소 분할을 위한 변수 설정
UNICODE_N, CHOSUNG_N, JUNGSUNG_N = 44032, 588, 28
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


class make_word:
    def __init__(self,w):
        #분해되어 있는 자모들
        self.word = w
        #현재 검사 위치
        self.w_ptr = -1
        #글자로 통합이 완료된 다음 위치
        self.r_ptr = 0
        #완료된 글자
        self.result = ''

    def start_m(self):
        # 입력된 자음이 키보드겹모음인 경우
        # 현재값 문자 완성
        char = self.word[self.w_ptr]
        if (char in K_double_m.keys()):
            self.combine(self.w_ptr)
            return

        self.w_ptr += 1
        if(self.w_ptr >= len(self.word)):
            return

        char = self.word[self.w_ptr]
        #다음 입력이 자음인 경우 -> 초성 state
        #이전값 문자 완성
        if char in JAUM:
            self.combine(self.w_ptr-1)
            self.chosung()

       #다음 입력이 모음인 경우 -> 겹모음인지 검사
        else:
            # 이전 모음과, 현재 모음의 조합
            char_dic = [self.word[self.w_ptr - 1], self.word[self.w_ptr]]
            # 겹모음이 맞는 경우 -> 처음
            # 겹모음 합체
            if char_dic in K_double_m.values():
                # 겹모음 합체
                self.word[self.w_ptr - 1:self.w_ptr+1] = double_m[double_m_list.index(char_dic)]
                self.combine(self.w_ptr)
                return

            # 겹모음이 아닌 경우 -> 처음모음 state
            # 이전값 문자 완성
            else:
                self.combine(self.w_ptr - 1)
                self.start_m()

    def chosung(self):
        char = self.word[self.w_ptr]
        if(self.w_ptr+1 >= len(self.word)):
            return

        #입력된 자음이 키보드겹자음인 경우
        if(char in K_double_j.keys()):
            self.w_ptr += 1
            char = self.word[self.w_ptr]

            #다음 입력이 자음인 경우 -> 초성 state
            #이전값 문자 완성
            if char in JAUM:
                self.combine(self.w_ptr-1)
                self.chosung()
            #다음 입력이 모음인 경우 -> 중성 state
            #'ㄳ' -> 'ㄱ','ㅅ' 로 분해시킨다.
            # 자음 하나를 뺀 이전값 문자 완성
            else:
                self.word[self.w_ptr-1:self.w_ptr] = K_double_j[self.word[self.w_ptr-1]]
                self.w_ptr += 1
                self.combine(self.w_ptr-2)
                self.jungsung()

        #일반 자음인 경우
        else:
            self.w_ptr += 1
            char = self.word[self.w_ptr]
            #다음 입력이 자음인 경우 -> 겹자음 state
            if char in JAUM:
                self.is_double_j()

            #다음 입력이 모음인 경우 -> 중성 state
            else:
                self.jungsung()

    def jungsung(self):
        self.w_ptr += 1

        if(self.w_ptr >= len(self.word)):
            return

        char = self.word[self.w_ptr]

        # 다음 입력이 키보드겹자음인 경우
        if (char in K_double_j.keys()):
            self.w_ptr += 1
            if(self.w_ptr >= len(self.word)):
                return
            char = self.word[self.w_ptr]
            # 다음 입력이 자음인 경우 -> 초성 state
            # 이전값 문자 완성
            if char in JAUM:
                self.combine(self.w_ptr - 1)
                self.chosung()
            # 다음 입력이 모음인 경우 -> 중성 state
            # 'ㄳ' -> 'ㄱ','ㅅ' 로 분해시킨다.
            # 자음 하나를 뺀 이전값 문자 완성
            else:
                self.word[self.w_ptr - 1:self.w_ptr] = K_double_j[self.word[self.w_ptr - 1]]
                self.w_ptr += 1
                self.combine(self.w_ptr - 2)
                self.jungsung()

        # 다음 입력이 자음인 경우 -> 종성 state
        elif char in JAUM:
            self.jongsung()

        # 다음 입력이 모음인 경우 -> 겹모음 state
        else:
            self.is_double_m()

    def jongsung(self):
        char = self.word[self.w_ptr]
        # 받침이 불가능한 자음 -> 초성 state
        # 이전값 문자 완성
        if char in ['ㅉ', 'ㄸ', 'ㅃ']:
            self.combine(self.w_ptr-1)
            self.chosung()
            return

        self.w_ptr += 1
        if(self.w_ptr >= len(self.word)):
            return

        char = self.word[self.w_ptr]

        # 다음 입력이 자음인 경우 -> 겹자음 state
        if char in JAUM:
            self.is_double_j()

        # 다음 입력이 모음인 경우 -> 중성 state
        # 자음하나를 뺀 이전값 문자 완성
        else:
            self.combine(self.w_ptr - 2)
            self.jungsung()

    def is_double_j(self):
        #이전 자음과, 현재 자음의 조합
        char_dic = [self.word[self.w_ptr-1],self.word[self.w_ptr]]
        #겹자음이 맞는 경우
        if char_dic in K_double_j.values():
            self.w_ptr += 1
            if (self.w_ptr >= len(self.word)):
                return
            # 다음 입력이 자음인 경우 -> 초성 state
            # 겹자음으로 합체 후 이전값 문자 완성
            char = self.word[self.w_ptr]
            if char in JAUM:
                #겹자음 합체
                self.word[self.w_ptr-2:self.w_ptr] = double_j[double_j_list.index(char_dic)]
                self.combine(self.w_ptr-2)
                self.w_ptr -= 1
                self.chosung()

            # 다음 입력이 모음인 경우 -> 중성 state
            # 자음 하나를 뺀 이전값 문자 완성
            else:
                self.combine(self.w_ptr-2)
                self.jungsung()

        # 겹자음이 아닌 경우 -> 초성 state
        # 이전값 문자 완성
        else:
            self.combine(self.w_ptr-1)
            self.chosung()

    def is_double_m(self):
        #이전 모음과, 현재 모음의 조합
        char_dic = [self.word[self.w_ptr-1],self.word[self.w_ptr]]
        # 겹모음이 맞는 경우 -> 중성 state
        # 겹모음 합체
        if char_dic in K_double_m.values():
            # 겹모음 합체
            self.word[self.w_ptr - 2:self.w_ptr] = double_m[double_m_list.index(char_dic)]
            self.w_ptr -= 1
            self.jungsung()

        #겹모음이 아닌 경우 -> 처음모음 state
        #이전값 완료
        else:
            self.combine(self.w_ptr-1)
            self.start_m()


    #r_ptr 부터 f_ptr까지 통합한다.
    def combine(self,f_ptr):

        comb_list = self.word[self.r_ptr:f_ptr + 1]
        if(len(comb_list) == 3 and comb_list[0] in CHOSUNG):
            cho = CHOSUNG.index(comb_list[0])
            jung = JUNGSUNG.index(comb_list[1])
            jong = JONGSUNG.index(comb_list[2])
            self.result += chr(UNICODE_N + ( cho * CHOSUNG_N ) + ( jung * JUNGSUNG_N ) + jong)
        elif(len(comb_list) == 2 and comb_list[0] in CHOSUNG and comb_list[1] in JUNGSUNG):
            cho = CHOSUNG.index(comb_list[0])
            jung = JUNGSUNG.index(comb_list[1])
            jong = 0
            self.result += chr(UNICODE_N + ( cho * CHOSUNG_N ) + ( jung * JUNGSUNG_N ) + jong)
        else:
            for i in comb_list:
                self.result += i

        self.r_ptr = f_ptr + 1

    def __main__(self):
        self.w_ptr += 1
        #검사가 완료될 때 까지 반복한다.
        while(self.w_ptr < len(self.word)):

            char = self.word[self.w_ptr]
            #처음state
            if char in JAUM:
                self.chosung()
            else:
                self.start_m()
            self.w_ptr += 1

        #혹시 통합이 완료되지 않았다면 나머지를 합쳐준다.
        if(self.r_ptr <= len(self.word)):
            self.combine(self.w_ptr)


def combine_word(word):
    m = make_word(word)
    m.__main__()
    return m.result

#print(combine_word(['ㄱ', 'ㅡ']))
#print(time.time()-start)
