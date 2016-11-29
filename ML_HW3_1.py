import math
import random
import matplotlib.pyplot as plt

g_CONVERGE_CONDITION = 0.01
g_ALPHA = 0.01
g_TRAIN_DATA_RANGE_PERCENT = 0.8
g_BANK_DATA_ADDRESS = 'data_banknote_authentication.txt'
g_K_FOLD_ARG = 5
g_NaN = float('nan')
g_LEARNING_FACTORS_LOOP_SIZE = 10


# <editor-fold desc="Train Function">
def train(alpha, x_matrix, y_list):
    print 'Training started...'
    c = len(x_matrix[0])
    r = len(y_list)

    # Coefficients Initialization
    a_list = [1] * c

    # <editor-fold desc="Normalization">
    for i in range(c):
        # min_x = x_matrix[0][i]
        max_x = x_matrix[0][i]
        for j in range(1, r):
            # if min_x > x_matrix[j][i]:
            #     min_x = x_matrix[j][i]
            if max_x < x_matrix[j][i]:
                max_x = x_matrix[j][i]
        for j in range(r):
            x_matrix[j][i] /= 1.0 * max_x
            # x_matrix[j][i] = 1.0 * (x_matrix[j][i] - min_x) / (max_x - min_x)
    # </editor-fold>

    iter = 0
    repeat = True
    while repeat:
        a_ = a_list[:]
        s = lambda a_list, x_list: sum(map(lambda a, x: a * x, a_list, x_list))  # < Likelihood
        g = lambda z: 1.0 / (1 + math.e ** (-z))
        log_l = lambda y_list, x_matrix, column_index, a_list: sum(
            map(lambda y, x_list, j, a_list: (y - g(s(a_list, x_list))) * x_list[j], y_list, x_matrix,
                [column_index] * r, [a_list] * r))  # < Log Likelihood
        for c_i in range(c):
            a_[c_i] = round(a_list[c_i] + alpha * log_l(y_list, x_matrix, c_i, a_list), 2)

        repeat = False
        for i in range(c):
            # repeat = repeat or (math.fabs(a_[i] - a_list[i]) > g_CONVERGE_CONDITION)
            repeat = math.fabs(a_[i] - a_list[i]) > g_CONVERGE_CONDITION
            if repeat:
                break
            elif i == c-1:
                break

        a_list = a_[:]
        iter += 1
        # print 'iter', iter, ': ', a_list
    # print '---------------------------------'
    print 'Training finished'
    print 'a_list: ', a_list
    return a_list, iter


# </editor-fold>


# <editor-fold desc="Prediction Function">
def predict(a_list, x_matrix):
    print 'Prediction startted...',
    predicted_class = []
    # index = 0
    for x_list in x_matrix:
        c = sum(map(lambda a, x: a * x, a_list, x_list))
        if c < 0.5:
            predicted_class.append(0)
        else:
            predicted_class.append(1)
    # print 'Predicted labels', predicted_class
    print 'prediction finished.'
    return predicted_class


# </editor-fold>


# <editor-fold desc="Reading data">
x_matrix = []
y_list = []
with open(g_BANK_DATA_ADDRESS, 'r') as bankFile:
    print 'Reading data'
    lines = bankFile.readlines()
    random.shuffle(lines)
    for row in lines:
        x_list = ['1'] + row.replace('\n', '').split(',')
        y_list.append(int(x_list.pop(-1)))
        x_matrix.append([float(x) for x in x_list])
# print 'x_matrix: \n', x_matrix
# print 'y_list: \n', y_list
# </editor-fold>

number_of_0 = y_list.count(0)
number_of_1 = y_list.count(1)
# print 'number_of_0 =', number_of_0
# print 'number_of_1 =', number_of_1

''''''
# <editor-fold desc="k-fold-cross-validation">
test_size = (len(x_matrix) / g_K_FOLD_ARG)
for k in range(g_K_FOLD_ARG):
    print k + 1, ":"
    # <editor-fold desc="Train and Test Data Separation">
    # train0_start_index = int(0)
    # train0_end_index = int(number_of_0 * g_TRAIN_DATA_RANGE_PERCENT)
    # train1_start_index = int(number_of_0)
    # train1_end_index = int(number_of_0 + number_of_1 * g_TRAIN_DATA_RANGE_PERCENT)
    # test0_start_index = int(train0_end_index)
    # test0_end_index = int(number_of_0)
    # test1_start_index = int(train1_end_index)
    # test1_end_index = int(-1)
    start_train1 = 0
    end_train1 = test_size * k
    start_train2 = test_size * (k + 1)
    end_train2 = -1
    start_test = test_size * k
    end_test = test_size * (k + 1)

    # train_feature = x_matrix[train0_start_index : train0_end_index] + x_matrix[train1_start_index : train1_end_index]
    # train_class = y_list[train0_start_index : train0_end_index] + y_list[train1_start_index : train1_end_index]
    train_feature = x_matrix[start_train1: end_train1] + x_matrix[start_train2: end_train2]
    train_class = y_list[start_train1: end_train1] + y_list[start_train2: end_train2]
    # print 'train_x:', train_feature
    # print 'train_y:', train_class
    # test_feature = x_matrix[test0_start_index : test0_end_index] + x_matrix[test1_start_index : test1_end_index]
    # test_class = y_list[test0_start_index : test0_end_index] + y_list[test1_start_index : test1_end_index]
    test_feature = x_matrix[start_test: end_test]
    test_class = y_list[start_test: end_test]
    # print 'test_x:', test_feature
    # print 'test_y:', test_class
    # del train0_start_index, train0_end_index, train1_start_index, train1_end_index, test0_start_index, test0_end_index, test1_start_index, test1_end_index
    del start_train1, start_train2, end_train1, end_train2, start_test, end_test
    # </editor-fold>

    a_list, iter = train(g_ALPHA, train_feature, train_class)
    # a_list = [2.8759221411906566, -14.490962172858172, -14.03073672538742, -24.321280250547986, 0.1291811178224639]
    # print 'a_list: ', a_list

    predicted_class = predict(a_list, test_feature)

    # <editor-fold desc="Initializing Measures">
    #  | P   N
    # _|_______
    # P| TP  FN
    # N| FP  TN
    # #TP: Number of Positive Predicted Truly
    # #FP: Number of Positive Predicted Falsely
    # #TN: Number of Negative Predicted Truly
    # #FN: Number of Negative Predicted Falsely
    # #T : Number of Predicted Truly    = #TP + #TN
    # #F : Number of Predicted Falsely  = #FP + #FN
    # #P : Number of Positive Predicted = #TP + #FP
    # #N : Number of Negative Predicted = #FN + #TN
    T0 = 0
    F0 = 0
    T1 = 0
    F1 = 0
    # </editor-fold>

    # <editor-fold desc="Output">
    # print 'Predicted', '\t', 'Class', '\t', 'Status'
    for i in range(len(predicted_class)):
        p = predicted_class[i]
        c = test_class[i]
        if p == c:
            # print p, '\t\t\t', c, '\t\t', 'T'
            if p == 0:
                T0 += 1
            else:
                T1 += 1
        else:
            # print p, '\t\t\t', c, '\t\t', 'F'
            if p == 0:
                F0 += 1
            else:
                F1 += 1
    # print '------------------------------------\n'

    # Accuracy  = #T  / (#T  + #F )
    T = T0 + T1
    F = F0 + F1
    print 'Accuracy =', round(1.0 * T / (T + F), 2)

    # # Precision = #TP / (#TP + #FP )
    # print 'Original Precision =', 1.0 * T0 / (T0 + F0)
    # print 'Fake Precision =', 1.0 * T1 / (T1 + F1)
    # # Recall(c) = #TP / (#TP + #FN)
    # print 'Original Precision =', 1.0 * T0 / (T0 + F1)
    # print 'Fake Precision =', 1.0 * T1 / (T1 + F0)
    if (T0 + F0) == 0:
        orig_prec = g_NaN
    else:
        orig_prec = round(1.0 * T0 / (T0 + F0), 2)
    if (T1 + F1) == 0:
        fake_prec = g_NaN
    else:
        fake_prec = round(1.0 * T1 / (T1 + F1), 2)
    if (T0 + F1) == 0:
        orig_recall = g_NaN
    else:
        orig_recall = round(1.0 * T0 / (T0 + F1), 2)
    if (T1 + F0) == 0:
        fake_recall = g_NaN
    else:
        fake_recall = round(1.0 * T1 / (T1 + F0), 2)
    orig_f1score = round(2.0 * (orig_prec * orig_recall) / (orig_prec + orig_recall), 2)
    fake_f1score = round(2.0 * (fake_prec * fake_recall) / (fake_prec + fake_recall), 2)
    orig_count = predicted_class.count(0)
    fake_count = predicted_class.count(1)
    print '\t\t\t', 'Precision', '\t', 'Recall', '\t\t', 'F1-Score', '\t', 'support'
    print 'Origin', '\t\t', orig_prec, '\t\t', orig_recall, '\t\t', orig_f1score, '\t\t', orig_count
    print 'Fake  ', '\t\t', fake_prec, '\t\t', fake_recall, '\t\t', fake_f1score, '\t\t', fake_count
    prec_avg = round((orig_prec * orig_count + fake_prec * fake_count) / (orig_count + fake_count), 2)
    recall_avg = round((orig_recall * orig_count + fake_recall * fake_count) / (orig_count + fake_count), 2)
    f1_avg = round((orig_f1score * orig_count + fake_f1score * fake_count) / (orig_count + fake_count), 2)
    print '\navg\\total', '\t', prec_avg, '\t\t', recall_avg, '\t\t', f1_avg, '\t\t', (orig_count + fake_count)
    print '===================================='
    # </editor-fold>
print '************************************'
print '************************************\n'
# </editor-fold>
''''''

alpha_list = []
iter_list = []
for n in range(1, g_LEARNING_FACTORS_LOOP_SIZE + 1):
    alpha = 1.0 * n / g_LEARNING_FACTORS_LOOP_SIZE
    a_list, iter = train(alpha, x_matrix[ : (number_of_0 + number_of_1) / g_K_FOLD_ARG], y_list[ : (number_of_0 + number_of_1) / g_K_FOLD_ARG])
    print 'alpha =', alpha, '\titeration count =', iter, '\n'
    alpha_list.append(alpha)
    iter_list.append(iter)

plt.plot(alpha_list, iter_list, 'ro')
min_iter = min(iter_list) - 1
max_iter = max(iter_list) + 1
plt.axis([0, 1, min_iter, max_iter])
plt.show()
