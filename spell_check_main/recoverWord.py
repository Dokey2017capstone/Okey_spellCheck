"""
recover_word(list)
    list: 글자가 인덱싱 된 결과물(tensorflow 작업 결과물)
    return : 단어 string
convert_num(string)
    string : 단어
    return : 글자가 인덱싱 된 결과물(tensorflow input data)
"""
#import time
#start = time.time()

import hangul as hg


#0 : ㄱ
index_word = hg.char_arr

#ㄱ : 0
dic_word = hg.char_dic

def recover_word(word_list):
    """
    인덱싱된 숫자 리스트를 다시 한글로 조합한다.
    """
    word = ""
    #자모음으로 변환
    for i in word_list:
        if(i == 0):
            continue
        word += index_word[i]

    #글자로 통합
    return word

def convert_num(word):
    """
    한글을 입력 인덱싱 형태로 변환한다.
    """
    error = []
    for i in word:
        error.append(dic_word[i])

    return error

"""
print(recover_word([4,9488,192,0,0,0,0]))
print(recover_word([   0, 1256 , 192 ,   0]))
print(convert_num("한글"))


#실행시간
end = time.time()
print(end-start)
"""