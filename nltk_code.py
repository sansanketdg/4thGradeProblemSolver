from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
import csv

filename_pos = "positive"
filename_neg = "negative"

sentence = "Kunal owns 5 lamborghinis."
sentence1 = "Rasmhi sleeps for 5 hours."
sentence2 = "Sanket works for 5 hours."

i = 1
in_txt = csv.reader(open(filename_pos, "r"))
in_txt1 = csv.reader(open(filename_neg, "r"))
x = []
y = []
for content in in_txt:
    #print(content)
    x.append(content)
    y.append(0)

for content in in_txt1:
    #print(content)
    x.append(content)
    y.append(1)

vectorizerUnigram = TfidfVectorizer()
x_vector = []
for i in range(0, len(x)):
    x_vector.append(vectorizerUnigram.fit_transform(x[i], y[i]))

# x_vector_tt = x_vector[0].tolist()
# print(x_vector_tt)
# print(len(x_vector))
# print(len(y))
#print(x_vector)
clf = LogisticRegression()
#clf.fit(x_vector,y)
#sentence = "Sanket found 80 seashells on the beach . He gave SanBrian some of his seashells . He has 27 seashell . How many seashells did she give to SanBrian ? "
#sample_vector = vectorizerUnigram.transform(sentence)
#y_t = clf.predict(sample_vector)
#print(y_t)