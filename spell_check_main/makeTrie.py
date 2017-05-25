# encoding: utf-8
from __future__ import unicode_literals

word_dir = 'C:/Users/kimhyeji/PycharmProjects/tfTest/trie.json'
read_dir = 'C:/Users/kimhyeji/Desktop/데이터/dic_.csv'

import csv
import json

with open(read_dir, 'r', newline="\n") as f, open(word_dir,'w') as w:
    rf  = csv.reader(f)

    root = dict()
    for word in f:
        current_dict = root
        for letter in word:
            if (letter == "\r" or letter ==',' or letter == "\n"):
                continue
            current_dict = current_dict.setdefault(letter, {})
        current_dict[0] = 0
    w.write(json.dumps(root))
