4th GRADE WORD PROBLEM SOLVER
============================================================

Steps to run -

1. Model 1
=>	* model1.py - This is our first model code. This file reads the questions from negative and positive questions, train itself based on that. Then based on test_negative and test_positive questions, it tests its question solving skill.

2. Model 2
=>	* model2.py - This is the second model code. The training input and test input is same like model 1.

3. Model 3
=>	* verbAnalyzer.py - This piece of code will take in the training data - transferNeg, transPos, final_verbs, AssignTrainVerbs from data folder. It will generate a file containing the list of verbs and it’s operational meaning. This list is generated based on classification performed by classifier on training data after extracting verbs from each sentence.
	* data_format.py - This file generates an intermediate file after processing test data in order to solve equations by the next script. It provides data in the required fashion.
	* vector_try.py - This file solves the equations and outputs the answer based on the data generated by data_format.py and list of verbs and it’s operational value provided by the verbAnalyzer.py.
	* calc_correctness.py - This file will basically calculate the correctness of the model in solving the test-word problems. It will display the accuracy of this model.

4. Model 4
=>	* verb_sentence_analyzer - This piece of code will take in the training data - transferNeg, transPos, final_verbs, AssignTrainVerbs from data folder. It will generate a trained model of the classifier named ‘trained_model.pkl’.
	* generate_analysing_data.py - This code will load the ‘trained_model.pkl’ classifier, then based on the training question.
	* test_accuracy - This file will basically calculate the correctness of the model in solving the test-word problems. It will display the accuracy of this model.

	