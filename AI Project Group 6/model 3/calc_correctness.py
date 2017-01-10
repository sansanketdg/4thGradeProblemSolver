import sklearn

model_answer_file = "data/answers"
correct_answer_file = "data/sample_test_answers"

m_ans_fp = open(model_answer_file, "r")
c_ans_fp = open(correct_answer_file, "r")

m_ans = m_ans_fp.read().splitlines()
c_ans = c_ans_fp.read().splitlines()

right_ans = 0

if len(m_ans) == len(c_ans):
    for i in range(len(m_ans)):
        if float(m_ans[i]) == float(c_ans[i]):
            right_ans += 1
    accuracy = right_ans/len(m_ans) * 100
    print("Accuracy:", accuracy)
    exit()
print("Incompatible answers to verify")