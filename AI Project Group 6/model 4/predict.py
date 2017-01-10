import ast

assign = 0
diff = []
final = 0
temp = []
sentences = []
content = []

filename_input = "./data/data_format1"
filename_output = "./data/answers"

input = open(filename_input, "r")
output = open(filename_output, "w")

content = input.read().splitlines()
# print(len(content))
# print(content[0])

for i in range(len(content)):
    sentences = content[i].split("\t")
    # print(len(sentences))
    assign = 0
    diff = []
    final = 0
    temp = []
    for j in range(len(sentences)):
        if sentences[j] == '':
            continue
        sentence = ast.literal_eval(sentences[j])
        # print(sentence)
        # print(len(sentence))
        for k in range(len(sentence)):
            if (k+1) < len(sentence):
                if "VB" in sentence[k][1] and "CD" in sentence[k+1][1]:
                    # print("found vb cd pair")
                    if sentence[k][2] == '0':
                        # print("assign")
                        if assign == 0:
                            assign = sentence[k+1][2]
                        else:
                            temp.append(sentence[k+1][2])
                    elif sentence[k][2] == '1':
                        # print("final")
                        if final == 0:
                            final = sentence[k+1][2]
                        else:
                            temp.append(sentence[k+1][2])
                    elif sentence[k][2] == 't+':
                        # print("pos")
                        diff.append(sentence[k+1][2])
                    elif sentence[k][2] == 't-':
                        # print("neg")
                        diff.append(-sentence[k+1][2])
    diff_eval = 0
    for n in range(len(diff)):
        diff_eval += diff[n]
    # print("diff_eval: ", diff_eval)
    temp_eval = 0
    for n in range(len(temp)):
        temp_eval += temp[n]
    '''print("temp_eval: ", temp_eval)
    print(assign)
    print(temp)
    print(diff)
    print(final)'''
    answer = abs(assign + temp_eval + diff_eval - final)
    output.write(str(answer) + "\n")


input.close()
output.close()
