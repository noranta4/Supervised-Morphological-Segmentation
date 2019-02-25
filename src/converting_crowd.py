import sys

orig_stdout = sys.stdout
f = open('crowd-converted.txt', 'w')
sys.stdout = f

with open("crowd-sourced-annotations.txt") as inputfile:
    for line in inputfile:
        actual_line = line.split('\t')
        actual_line[-1] = actual_line[-1][:-1]
        flag_first = 0
        for item in actual_line:
            if flag_first == 0:
                print item + '\t',
                flag_first = 1
            else:
                print item + ':',
        print

sys.stdout = orig_stdout
f.close()