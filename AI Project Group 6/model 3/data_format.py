import nltk

filename_q = "./data/q2.txt"
filename_pos = "./data/positive"
# filename_neg = "C:/Kunal/masters/Admit/SUNY-SB/sem1/AI/Assignments/project/project code/data/negative"
filename_test = "./data/test.txt"
filename_out_format = "data/data_format1"
corpus = []

in_txt = open(filename_test, "r")
for content in in_txt:
    corpus.append(content)
    corpus.append("\n")

'''in_txt = open(filename_q, "r")
for content in in_txt:
    corpus.append(content)
    corpus.append("\n")'''
'''in_txt = open(filename_pos, "r")
for content in in_txt:
    corpus.append(content)
    corpus.append("\n")
'''
answer = []

for i in range(len(corpus)):
    if not corpus[i] == "\n":
        tokens = nltk.word_tokenize(corpus[i])
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
                    tp.append(-7)
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
    else:
        answer.append(corpus[i])

out_txt = open(filename_out_format, "w")
for i in range(len(answer)):
    out_txt.write(str(answer[i]))

out_txt.close()
in_txt.close()
