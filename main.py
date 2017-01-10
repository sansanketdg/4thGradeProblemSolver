import csv

fname = "Questions.txt"

sentences = []
equations = []
answers = []

i = 0
in_txt = csv.reader(open(fname, "r"))
for line in in_txt:
    if(i%3 == 0):
        i = i + 1
        sentences.extend(line)

    elif( i%3 == 1):
        i = i + 1
        answers.extend(line)

    else:
        i = i + 1
        equations.extend(line)

print(len(sentences))
print(len(answers))
print(len(equations))
print(sentences[0:3])
print(answers[0:3])
print(equations[0:3])

# for each_sentence in sentences:
