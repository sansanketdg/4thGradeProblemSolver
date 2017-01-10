from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import nltk
import csv
import numpy as np
import ast


verbs = {}


def add_verbs(v):
    # print(v)
    # if v not in verbs["data"]:
        verbs.update(v)


def fill_verbs(n, val):
    # v = {"name": n, "value": val}
    v = {n: val}
    add_verbs(v)

filename_v1 = "data/pverbs_h1.txt"
# filename_v2 = "C:/Kunal/masters/Admit/SUNY-SB/sem1/AI/Assignments/project/project code/data/pverbs_h2.txt"
infile = "data/data_format1"
outfile = "data/answers"

v1 = open(filename_v1, "r")
# v2 = open(filename_v2, "r")
inp = open(infile, "r")
out = open(outfile, "w")

for content in v1:
    #print(content)
    pair = content.split("\t")
    tp = pair[1].split("\n")
    pair[1] = tp[0]
#   print(pair)
    fill_verbs(pair[0], pair[1])

X = 0
x_ini = 0
x_final = 0
x_diff = 0
lines = inp.read().splitlines()    # print(content)
print(len(lines))

for i in range(len(lines)):
    X = 0
    x_final = 0
    x_diff = 0
    # question = np.array(lines[i])
    question = ast.literal_eval(lines[i])
    answer = 0
    for j in range(len(question)):
        if j + 1 < len(question):
            if "VB" in question[j][1] and "CD" in question[j+1][1]:
                val = verbs.get(question[j][0], None)
                if val == 0:
                    print(question[j+1][2], "   69")
                    if X == 0 and x_diff == 0 and x_final == 0:
                        print(question[j + 1][2], "   71")
                        X = question[j+1][2]
                    elif not X == 0 and x_final == 0:
                        print(question[j + 1][2], "   74")
                        x_final = question[j+1][2]
                elif val == "t-":
                    print(question[j + 1][2], "   77")
                    x_diff = - question[j+1][2]
                elif val == "t+":
                    print(question[j + 1][2], "   80")
                    x_diff = question[j+1][2]
    if not x_diff == 0:
        if not X == 0:
            answer = X + x_diff
        else:
            answer = x_final - x_diff
    else:
        answer = abs(X - x_final)
    out.write(str(answer) + "\n")
