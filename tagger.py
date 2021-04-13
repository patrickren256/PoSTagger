# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np
from collections import defaultdict

np.set_printoptions(threshold=sys.maxsize)

# python3 tagger.py -d autotraining.txt -t autotest.txt -o autooutput.txt

pos_list = [ "AJ0-AV0", "AJ0-NN1", "AJ0-VVD", "AJ0-VVG", "AJ0-VVN", "AV0-AJ0", "AVP-PRP", "AVQ-CJS", "CJS-AVQ", "CJS-PRP", "CJT-DT0", "CRD-PNI", "DT0-CJT", "NN1-AJ0", "NN1-NP0", "NN1-VVB", "NN1-VVG", "NN2-VVZ", "NP0-NN1", "PNI-CRD", "PRP-AVP", "PRP-CJS", "VVB-NN1", "VVD-AJ0", "VVD-VVN", "VVG-AJ0", "VVG-NN1", "VVN-AJ0", "VVN-VVD", "VVZ-NN2", "AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD", "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI", "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0", "UNC", "VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHN", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ", "XX0", "ZZ0"]
dbug = True


def _initial_tables(training_list, pos_dict):
    '''
    mark the initial tables of the viterbi algorithm.

    returns pi, a, b, pos dictionary, word dictionary

    '''
    # initialize the initial tables.

    # generate reference dictionaries:
    # part of speech reference for indexing numpy arrays,
    # as progress down the file, create dictionary of numpy arrays that
    # map a word to emission counts. make everything lowercase i think
    bepis = len(pos_list)
    # emissions_table = defaultdict(lambda: np.ones(shape=(bepis, 1)))

    # create emissions as a python list, convert to np array
    # create dictionary alongside emissions mapping words to index.
    emissions_table = []
    emissions_dict = defaultdict(lambda:len(emissions_dict))

    # create the transition probability table, pos x pos size
    # check for either zeros or ones. maybe ones.
    trans_table = np.ones(shape=(bepis,bepis))/10000
    # after processing all files and grabbing all counts, normalize COLUMNS
    pie_table = np.ones(bepis)

    # begin processing training files.
    for training_file in training_list:
        with open(training_file, "r") as opened_file:
            tf = opened_file.readlines()
            # split the lines by " : ", generate list item of [word, pos]
            # prev used for storing
            prev = [".", "PUN"]
            em_dict_size = len(emissions_dict)
            for index in range(len(tf)):
                # current[0] == word ; currenT[1] == POS
                current = (tf[index].strip()).split(" : ")
                _word = current[0]
                _pos = current[1]

                emissions_dict[_word]

                # EMISSION TABLE
                # add the word to the emission table if it isnt already in
                if len(emissions_table) < len(emissions_dict):
                    # intially set to 1, maybe change later. This is to prevent
                    # too many zeros.
                    emissions_table.append([1/1000000000]*bepis)
                # update the count of the pos of the
                emissions_table[emissions_dict[_word]][pos_dict[_pos]] += 1

                #TRANS TABLE
                # print("Pre: {}".format(trans_table[pos_dict[prev[1]]][pos_dict[_pos]]))
                trans_table[pos_dict[prev[1]]][pos_dict[_pos]] += 1
                # print("Post: {}".format(trans_table[pos_dict[prev[1]]][pos_dict[_pos]]))

                #PIE TABLE
                # CURRENTLY CALCULATE AS POS AFTER A PUNCTUATION
                if prev[0] == "." and prev[1] == "PUN":
                    pie_table[pos_dict[_pos]] += 1

                prev = current
        if dbug: print(f"done training {training_file}")


    # normalize step
    # normalize by row
    trans_table = trans_table/trans_table.sum(axis=1, keepdims=1)
    pie_table = pie_table / pie_table.sum(keepdims=1)

    # normalize by column
    emissions_table = np.array(emissions_table)
    emissions_table = emissions_table/emissions_table.sum(axis=0, keepdims=1)

    # print("trans axis 1: {}".format(trans_table.sum(1)))
    # print("pieta axis 1: {}".format(pie_table.sum()))
    # print("emiss axis 0: {}".format(emissions_table.sum(0)))
    if dbug:
        print(f"done training")

    return (pie_table, trans_table, emissions_table, emissions_dict)


def viterbi(training_list, test_file, pos_dict):
    """
    an implementation of the viterbi algorithm.
    provided tagged data (training list),
    apply viterbi to untagged data (test_list), and return prob trellis and
    path trellis.

    :param training_list:
    :param test_file:
    :param output_file:
    :return:
    """
    # pie, a, b, are np arrays
    # pos_dict, emi_dict are dictionaries, used to index into pie, a, b.
    pie, a, b, emi_dict = _initial_tables(training_list, pos_dict)


    with open(test_file, "r") as testing_file:
        test = [line.strip() for line in testing_file ]
        # len observations
        obs_len = len(test)
        ste_len = len(pos_list)

        # no need to store entire trellis, only need 2 columns of trellis.
        # path trellis memory storage further mitigated by studying sentence
        # rather than entire input file.
        # can probably
        # prob_trellis = np.zeros(shape=(ste_len, obs_len))
        prob_prev = np.zeros(shape=(ste_len))
        prob_curr = np.zeros(shape=(ste_len))
        path_prev = np.empty(shape=(ste_len), dtype='object')
        path_curr = np.empty(shape=(ste_len), dtype='object')


        # s1
        for s in range(ste_len):
            prob_prev[s] = pie[s] * b[emi_dict[test[0]]][s]
            path_prev[s] = str(s) + ','


        for o in range(1, obs_len):
            path_curr = np.empty(shape=(ste_len), dtype='object')
            prob_curr = np.zeros(shape=(ste_len))
            if dbug and o%5000 == 0:
                print(f"still working... observation: {o}")
            for s in range(ste_len):
                # i see legit no reason to include B[s,o] in this calculation
                x = np.argmax(prob_prev * a[:,s])
                # deal with unseen words
                # emi_dict[test[o]]
                # if len(b) < len(emi_dict):
                #     np.append(b, np.ones(ste_len)/10000)
                #     b[emi_dict[test[o]]]
                # currently, solution is to ignore unseen words before.
                # this is a bad solution, but will work for the assignment
                # (((((( well enough))))))
                if emi_dict[test[o]] >= len(b):
                    bos = 1
                    print('unseen')
                else:
                    bos = b[emi_dict[test[o]],s]
                    # print('seen')

                # update trellis
                # prob_trellis[s,o] = prob_trellis[x,o-1] * a[x,s] * bos
                prob_curr[s] = prob_prev[x] * a[x,s] * bos


                path_curr[s] = path_prev[x] + str(s) + ','
                # path_trellis[s][o] = path_trellis[x][o-1] + str(s) + ','

            # normalize to avoid converging 0
            prob_curr = prob_curr/prob_curr.sum()

            # copy over path_curr to path_prev
            path_prev = np.copy(path_curr)
            prob_prev = np.copy(prob_curr)

            # print(prob_trellis[:,o])
            # print(prob_trellis.sum(axis=0))
        if dbug:
            print(f"done viterbi")
        return prob_curr, path_curr, test


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    #
    # YOUR IMPLEMENTATION GOES HERE
    #
    pos_dict = {}
    for _ in range(len(pos_list)):
        pos_dict[pos_list[_]] = _

    prob_t, path_t, words = viterbi(training_list=training_list, test_file=test_file, pos_dict=pos_dict)

    with open(output_file, "w") as output:
        x = np.argmax(prob_t)
        final_pos = path_t[x].split(',')
        for i in range(len(words)):
            output.write(words[i] + " : " + pos_list[int(final_pos[i])] + "\n")

        if dbug:
            print(f"done writing to file {output_file}")


    # print(path_t[:,-1])



if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)
    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
