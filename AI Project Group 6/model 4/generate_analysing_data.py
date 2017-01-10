import nltk
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


filename_pkl = "trained_model.pkl"
# filename_test = "data/positive"
filename_test = "data/model4_test_questions"
filename_out_format = "data/data_format1"
corpus = []

in_txt = open(filename_test, "r")
for content in in_txt:
    corpus.append(content)
    corpus.append("\n")

answer = []

clf = joblib.load(filename_pkl)
vector = CountVectorizer(min_df=1, ngram_range=(1,3))

for i in range(len(corpus)):
    if not corpus[i] == "\n":
        sentences = corpus[i].split(". ")
        # print(len(sentences))
        for sent in range(len(sentences)):
            #print(sentences[sent])
            tokens = nltk.word_tokenize(sentences[sent])
            tagged = nltk.pos_tag(tokens)
            j = 0;
            while j in range(len(tagged)):
                # print(tagged[j])
                # for k in range(len(tagged[i])):
                if "NNS" in tagged[j][1] or "VB" in tagged[j][1] or "CD" in tagged[j][1]:
                    if "NN" in tagged[j][1]:
                        tp = list(tagged[j])
                        tp.append(-5)
                        tagged[j] = tuple(tp)
                    elif "VB" in tagged[j][1]:
                        tp = list(tagged[j])
                        sample = []
                        sample.append(sentences[sent])
                        # print(sample)
                        # print("predict for sample is ")
                        # print(clf.predict(sample))
                        predicted_ = str(clf.predict(sample))
                        predicted_ = predicted_.strip("[']")
                        # print(predicted_)
                        tp.append(predicted_)
                        #tp.append(-7)
                        tagged[j] = tuple(tp)
                    elif "CD" in tagged[j][1]:
                        tp = list(tagged[j])
                        tp.append(float(tagged[j][0]))
                        tagged[j] = tuple(tp)
                    # print(tagged[j])
                    j += 1;
                else:
                    tagged.pop(j)
            answer.append(tagged)
            answer.append("\t")
    else:
        answer.append(corpus[i])

out_txt = open(filename_out_format, "w")
for i in range(len(answer)):
    out_txt.write(str(answer[i]))

out_txt.close()
in_txt.close()
