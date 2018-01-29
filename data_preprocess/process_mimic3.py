# encoding=utf-8

# encoding=utf-8
# !/usr/bin/env python

import pickle
import re

with open('../data/MIMIC3_RAW_DSUMS', 'r') as file, open('../data/MIMIC3_DSUMS', 'w') as fileWriter:
    for i, line in enumerate(file.readlines()[1:]):
        rows = line.split('|')
        raw_dsum = rows[6].strip('"')
        codes = rows[5].strip('"').split(',')

        fileWriter.write(str(i + 1) + '|')
        for code in codes[:-1]:
            fileWriter.write(code + ',')
        fileWriter.write(codes[-1] + '|')
        fileWriter.write(raw_dsum + '\n')
