import nltk
from nltk import word_tokenize

sentence = "Sanket has 5 oranges"
sentence1 = "Sanket gave me 2 oranges"
sentence2 = "How many oranges Sanket has"
text = word_tokenize(sentence)
text1 = word_tokenize(sentence1)
text2 = word_tokenize(sentence2)
tags = nltk.pos_tag(text)
tags1 = nltk.pos_tag(text1)
tags2 = nltk.pos_tag(text2)
print(tags)
print(tags1)
print(tags2)