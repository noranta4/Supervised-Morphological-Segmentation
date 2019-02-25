import sklearn_crfsuite
import pickle

# COLLECT DATA AND LABELLING ##########################################################################
training_dic = {}
dev_dic = {}
test_dic = {}

input_files = ['training.eng.txt', 'dev.eng.txt', 'test.eng.txt' ] # if crowd-sourced-annotations: change ".eng" with "-crowd-converted"
dictionaries = (training_dic, dev_dic, test_dic)
counter = 0
limit = 0 # init limit
n_samples = 1000

for filename in input_files:
    with open(filename) as inputfile:
        for line in inputfile:
            actual_line = line.split('\t')
            result = []
            actual_morph = ''
            flag = 0
            for c in actual_line[1]:
                if c == ':' and flag == 0:
                    result.append(actual_morph)
                    actual_morph = ''
                    flag = 1
                if flag == 0:
                    actual_morph += c
                if c == ',':
                    break
                if flag == 1 and c == ' ':
                    flag = 0

            label = ''
            for morph in result:
                if len(morph) == 1:
                    label += 'S'
                else:
                    label += 'B'
                    for i in range(len(morph)-2):
                        label += 'M'
                    label += 'E'
            dictionaries[counter][actual_line[0]] = label
            limit += 1 # LIMIT ON
            if limit > n_samples: break
        limit = 0
        counter += 1

# COMPUTE FEATURES ###############################################################################

delta = 6

epsilon = 0.001
max_iterations = 80

def prepare_data(word_dictonary, delta):
    X = [] # list (learning set) of list (word) of dics (chars), INPUT for crf
    Y = [] # list (learning set) of list (word) of labels (chars), INPUT for crf
    words = [] # list (learning set) of list (word) of chars
    for word in word_dictonary:
        word_plus = '[' + word + ']' # <w> and <\w> replaced with [ and ]
        word_list = [] # container of the dic of each character in a word
        word_label_list = [] # container of the label of each character in a word
        for i in range(len(word_plus)):
            char_dic = {} # dic of features of the actual char
            for j in range(delta):
                char_dic['right_' + word_plus[i:i + j + 1]] = 1
            for j in range(delta):
                if i - j - 1 < 0: break
                char_dic['left_' + word_plus[i - j - 1:i]] = 1
            char_dic['pos_start_' + str(i)] = 1  # extra feature: left index of the letter in the word
            # char_dic['pos_end_' + str(len(word) - i)] = 1  # extra feature: right index of the letter in the word
            if word_plus[i] in ['a', 's', 'o']: # extra feature: stressed characters (discussed in the report)
                char_dic[str(word_plus[i])] = 1
            word_list.append(char_dic)

            if word_plus[i] == '[': word_label_list.append('[') # labeling start and end
            elif word_plus[i] == ']': word_label_list.append(']')
            else: word_label_list.append(word_dictonary[word][i-1]) # labeling chars
        X.append(word_list)
        Y.append(word_label_list)
        temp_list_word = [char for char in word_plus]
        words.append(temp_list_word)
    return (X, Y, words)

print('features computed')

# DATA PREPARATION AND FIT #######################################################################

X_training, Y_training, words_training = prepare_data(training_dic, delta)
X_dev, Y_dev, words_dev = prepare_data(dev_dic, delta)
X_test, Y_test, words_test = prepare_data(test_dic, delta)
crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)
pickle.dump(crf, open("crf_model.model", "wb"))

print('training done')

# EVALUATION #####################################################################################

Y_predict = crf.predict(X_test)
H, I, D = 0, 0, 0
for j in range(len(Y_test)):
    for i in range(len(Y_test[j])):
        if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
            if Y_test[j][i] == Y_predict[j][i]:
                H += 1
            else:
                D += 1
        else:
            if (Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S'):
                I += 1
P = float(H)/(H+I)
R = float(H)/(H+D)
F1 = (2*P*R)/(P+R)

print('delta = ' + str(delta) + '\tNsamples = ' + str(n_samples) + '\tepsilon = ' + str(epsilon) + '\tmax_iter = ' + str(max_iterations))
print('Precision = ' + str(P))
print('Recall = ' + str(R))
print('F1-score = ' + str(F1))


