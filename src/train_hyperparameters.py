import sklearn_crfsuite


# COLLECT DATA AND LABELLING ###################################################################
training_dic = {}
dev_dic = {}
test_dic = {}

input_files = ['training.eng.txt', 'dev.eng.txt', 'test.eng.txt' ]
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
            limit += 1 # LIMIT ON #####################################################################
            if limit > n_samples: break
        limit = 0
        counter += 1

# COMPUTE FEATURES ###############################################################################

def prepare_data(word_dictonary, delta):
    X = []
    Y = []
    words = []
    for word in word_dictonary:
        word_plus = '[' + word + ']' # <w> and <\w> replaced with [ and ]
        word_list = []
        word_label_list = []
        for i in range(len(word_plus)):
            char_dic = {} # char_dictionary.copy()
            for j in range(delta):
                char_dic['right_' + word_plus[i:i + j + 1]] = 1
            for j in range(delta):
                if i - j - 1 < 0: break
                char_dic['left_' + word_plus[i - j - 1:i]] = 1

            char_dic['pos_start_' + str(i)] = 1  # extra feature: position of the letter in the word
            if word_plus[i] in ['a', 's', 'o']:
                char_dic[str(word_plus[i])] = 1

            #char_dic['pos_end_' + str(len(word)-i)] = 1
            # extra features: if letters are frequently last o first letters in a word
            #if word_plus[i] in ['e', 't', 'd', 's', 'n', 'r', 'y', 'f', 'l', 'o', 'g', 'h']:
                #char_dic[str(word_plus[i])] = 1
            # if word_plus[i] in ['e', 't', 'd', 's']:
            #     char_dic['most_ending1'] = 1
            # elif word_plus[i] in ['n', 'r', 'y', 'f']:
            #     char_dic['most_ending2'] = 1
            # if word_plus[i] in ['t', 'o', 'a', 'w']:
            #     char_dic['most_starting1'] = 1
            # elif word_plus[i] in ['b', 'c', 'd', 's']:
            #     char_dic['most_starting2'] = 1

            word_list.append(char_dic)
            if word_plus[i] == '[': word_label_list.append('[')
            elif word_plus[i] == ']': word_label_list.append(']')
            else: word_label_list.append(word_dictonary[word][i-1])
        X.append(word_list)
        Y.append(word_label_list)
        temp_list_word = [char for char in word_plus]
        words.append(temp_list_word)
    return X, Y, words

print('features computed')

# GRID SEARCH ##################################################################################

best_epsilon, best_max_iteration, best_delta = 0, 0, 0
maxF1 = 0
for epsilon in [0.001, 0.00001, 0.0000001]:
    for max_iterations in [80, 120, 160]:
        for delta in [3, 4, 5, 6, 7, 8, 9]:
            X_training, Y_training, words_training = prepare_data(training_dic, delta)
            X_dev, Y_dev, words_dev = prepare_data(dev_dic, delta)
            X_test, Y_test, words_test = prepare_data(test_dic, delta)
            crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
            crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

            Y_predict = crf.predict(X_dev)
            H, I, D = 0, 0, 0
            for j in range(len(Y_dev)):
                for i in range(len(Y_dev[j])):
                    if Y_dev[j][i] == 'E' or Y_dev[j][i] == 'S':
                        if Y_dev[j][i] == Y_predict[j][i]:
                            H += 1
                        else:
                            D += 1
                    else:
                        if (Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S'):
                            I += 1
            P = float(H)/(H+I)
            R = float(H)/(H+D)
            F1 = (2*P*R)/(P+R)
            if maxF1 < F1:
                maxF1 = F1
                best_epsilon = epsilon
                best_max_iteration = max_iterations
                best_delta = delta
                print('delta = ' + str(delta) + '\tNsamples = ' + str(n_samples) + '\tepsilon = ' + str(epsilon) + '\tmax_iter = ' + str(max_iterations))
                print('Precision = ' + str(P))
                print('Recall = ' + str(R))
                print('F1-score = ' + str(F1))

# EVALUATUION ON THE TEST SET OF THE BEST CONFIGURATION ##########################################################

X_training, Y_training, words_training = prepare_data(training_dic, best_delta)
X_dev, Y_dev, words_dev = prepare_data(dev_dic, best_delta)
X_test, Y_test, words_test = prepare_data(test_dic, best_delta)
crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)


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
print('\nEvaluation on the Test set\n')
print('delta = ' + str(best_delta) + '\tNsamples = ' + str(n_samples) + '\tepsilon = ' + str(best_epsilon) + '\tmax_iter = ' + str(best_max_iteration))
print('Precision = ' + str(P))
print('Recall = ' + str(R))
print('F1-score = ' + str(F1))
print('\n' + str(n_samples) + '\t' + str(best_delta) + '\t' + str(best_epsilon) + '\t' + str(best_max_iteration) + '\t' + str(round(P, 3)) + '\t' + str(round(R, 3)) + '\t' + str(round(F1, 3)))