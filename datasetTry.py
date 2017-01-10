from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectPercentile
from sklearn.feature_extraction import FeatureHasher

def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)

filename_pos = "positive"
filename_neg = "negative"

questions_file = "q.txt"
equations_file = "eq.txt"

ques_txt = open(questions_file, "r")
equa_txt = open(equations_file, "r")

test_questions = "test_positive"

i = 1
corpus = []
Y = []
in_txt = open(filename_neg, "r")
test_in_txt = open(test_questions, "r")

test_corpus = []
for test_content in test_in_txt :
    test_corpus.append(test_content)

for content in in_txt :
    corpus.append(content)
    Y.append("negative")

in_txt = open(filename_pos, "r")

for content in in_txt :
    corpus.append(content)
    Y.append("positive")

verbs_corpus = []
for each_que in corpus:
    sentences = each_que.split('.')
    verbs = []
    for each_sen in sentences:
        tokens = nltk.word_tokenize(each_sen)
        tagged = nltk.pos_tag(tokens)
        for each_tag in tagged:
            if("VB" in each_tag[1]):
                #print("found VB tag - " + str(each_tag))
                verbs.append(each_tag[0])
    #print(verbs)
    verbs_corpus.append(str(verbs))

print(len(verbs_corpus))
print(len(corpus))
print(verbs_corpus[0])

# raw_x = []
# for each_pair in verbs_corpus:
#     raw_x.append(token_features(each_pair[0], each_pair[1]))
#
# hasher = FeatureHasher(input_type='string')
# X_newwww = []
# for each_xx in raw_x:
#     X_newwww.append(hasher.transform(each_xx))

# print(len(X_newwww))
# print(X_newwww[0])

    #print(tagged)
    #break
vectorizer = CountVectorizer(min_df=1, ngram_range=(1,3))
X = vectorizer.fit_transform(verbs_corpus)
#test_X = vectorizer.transform(test_corpus)
selector = SelectPercentile(chi2, 35)
X = selector.fit_transform(X, Y)
#test_X = selector.transform(test_X)
clf = LogisticRegression()
clf.fit(X,Y)

#test1 = corpus[0]
verbs_test_corpus = []
for each_test_que in test_corpus:
    #test1 = "Alyssa picked 17 plums and Jason picked 10 plums . Melanie picked 35 pears . How many plums were picked in all ?"
    test1 = each_test_que
    sentences_in_test1 = test1.split('.')
    verbs_test = []
    for each_sentence_in_test1 in sentences_in_test1:
        tokens = nltk.word_tokenize(each_sentence_in_test1)
        tagged = nltk.pos_tag(tokens)
        for each_tag in tagged:
            if ("VB" in each_tag[1]):
                # print("found VB tag - " + str(each_tag))
                verbs_test.append(each_tag[0])
            # print(verbs)
    verbs_test_corpus.append(str(verbs_test))
#print(verbs_test)

#temp = []
#temp.append(str(verbs_test))
X_test = vectorizer.transform(verbs_corpus)
X_test = selector.transform(X_test)
y_predict = clf.predict(X_test)
print(y_predict)


#y = clf.predict(X)
#y = clf.predict(test_X)

# corpus = test_corpus
# answer = []
# for i in range(0,5):
#     tokens = nltk.word_tokenize(corpus[i])
#     tagged = nltk.pos_tag(tokens)
#     #print(tagged)
#     temp_answer = []


#     for tag in tagged:
#         if tag[1] == 'CD':
#             temp_answer.append(float(tag[0]))
#     t = temp_answer[0]
#     if(y[i] == 'positive'):
#         for j in range(1, len(temp_answer)):
#             t += temp_answer[j]
#
#     if(y[i] == 'negative'):
#         for j in range(1, len(temp_answer)):
#             t = t - temp_answer[j]
#         if(t < 0):
#             t = 0 - t
#     answer.append(t)
# for i in range (0, 5):
#     print(corpus[i])
#     print(answer[i])