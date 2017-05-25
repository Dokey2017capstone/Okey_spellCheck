#세종 말뭉치로 단어 사전을 구축한다.
#빈도가 큰 단어 순으로 정리한다.
#
#그,244939
#수,230441
#있다,223017

import os
import re
import csv
import operator
import json

count = 0
file_location = 'C:/Users/kimhyeji/Desktop/데이터'
#directory 에 있는 file의 제목을 모두 가져온다.
directory = os.listdir(file_location)

#작업 위치를 변경한다.
os.chdir(file_location)

dic = {}
print(len(directory))
for f in directory:
    #확장자
    extension = f.split('.')[-1]

    if extension == 'json':
        json_data = open(f).read()
        data = json.loads(json_data)
    elif extension == 'csv':
        pass

    #python3부터는 ANSI 기준 작성 파일만 읽을 수 있다.
    #인코딩 문제를 위해 utf-16을 명시해준다.
    #윈도우즈에서의 text파일의 유니코드가 UTF-16이기 때문이다.
    try:
        print(f)
        with open(f, "r", encoding = 'utf16') as file:
            #한글 단어만 추출한다
            count += 1
            hangul = re.compile('[가-힣]+')
            for line in file:
                result = hangul.findall(line)
                if(len(result) > 6): continue
                for word in result:
                    if word in dic.keys():
                        dic[word] += 1
                    else:
                        dic[word] = 1
    except:
        try:
            print(f)
            with open(f, "r", encoding='utf8') as file:
                # 한글 단어만 추출한다
                count += 1

                hangul = re.compile('[가-힣]+')
                for line in file:
                    result = hangul.findall(line)
                    if (len(result) > 5): continue

                    for word in result:
                        if word in dic.keys():
                            dic[word] += 1
                        else:
                            dic[word] = 1
        except:
            try:
                with open(f, "r", encoding='cp949') as file:
                    # 한글 단어만 추출한다
                    count += 1

                    hangul = re.compile('[가-힣]+')
                    for line in file:
                        result = hangul.findall(line)
                        if (len(result) > 5): continue

                        for word in result:
                            if word in dic.keys():
                                dic[word] += 1
                            else:
                                dic[word] = 1
            except:
                print(f)
                print("error")
                continue

try:
    #빈도 수를 값에 따라서 내림차순으로 정렬하고, 리스트로 반환한다.

    with open('dic_.csv', 'w', newline="\n") as f:
        w = csv.writer(f)
        # list 에 들어있는 tuple을 작성한다.
        for key, value in dic.items():
            if(value > 110):
                w.writerow(key)

    print(len(dic))
    print(count)
    #newline = '\r\n 이 기본이기 때문에 \n 을 지정해주어야 한다.

except:
    with open('dic_.csv', 'w', newline="\n") as f:
        w = csv.writer(f)
        # list 에 들어있는 tuple을 작성한다.
        for key,value in dic.items():
            if(value > 100):
                w.writerow(key)

