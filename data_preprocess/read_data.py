# encoding=utf-8
import pickle
import re
import numpy as np

all_code = []  # icd9 code
all_text = []  # words
all_char = []  # chinese character

code_set = set()
word_size = []
m = re.compile(r'^[a-z]+$')
# new_merge_samples_fullcode_preprocess.preprocess
# new_merge_samples_code3_preprocess.preprocess
# MIMIC3_DSUM

with open('../data/MIMIC3_DSUMS', 'r', encoding='utf-8') as file:
    for line in file.readlines():
        columns = line.strip().split('|')
        code_list = columns[1].split(',')
        word_list = columns[2].split()
        if len(word_list) < 20:
            continue
        all_code.append(code_list)
        all_text.append([word for word in word_list if m.match(word)])
        word_size.append(len(word_list))
        code_set.update(code_list)
        sentence = ''.join(word_list)
        # all_char.append(re.findall(r'[\u4e00-\u9fa5]', sentence))
        for word in word_list:
            all_char.append(list(word))

print('word number in each document(max/min/avg):', max(word_size), min(word_size), int(np.mean(word_size)))
print('code num:', len(code_set))

with open('../cache/all_code.pkl', 'wb') as file:
    pickle.dump(all_code, file)

with open('../cache/all_text.pkl', 'wb') as file:
    pickle.dump(all_text, file)

with open('../cache/all_char.pkl', 'wb') as file:
    pickle.dump(all_char, file)

code_dict = {c: i for i, c in enumerate(list(code_set))}

with open('../cache/code_dict.pkl', 'wb') as file:
    pickle.dump(code_dict, file)
print('ICD-9 code number:', len(code_dict))
