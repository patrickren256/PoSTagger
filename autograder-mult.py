"""This is the main program file for the auto grader program.
The auto grader assumes that the following files are in the same directory as the autograder:
  - autotraining.txt  --> file of tagged words used to train the HMM tagger
  - autotest.txt      --> file of untagged words to be tagged by the HMM
  - autosolution.txt  --> file with correct tags for autotest words
This auto grader generates a file called results.txt that records the test results.
"""
import os
import random
import time


# t = ["training1.txt", "training2.txt", "training3.txt", "training4.txt", "training5.txt", "training6.txt", "training7.txt", "training8.txt", "training9.txt", "training10.txt"]

if __name__ == '__main__':
    # Invoke the shell command to train and test the HMM tagger
    print("Training on autotraining.txt, running tests on autotest.txt. "
          "Output --> autooutput.txt")

    for i in range(1,11):
        tt_file0 = "test{}.txt".format(i)
        tg_file1 = "training{}.txt".format(i)
        tg_file2 = "training{}.txt".format(random.randint(1,10))
        tg_file3 = "training{}.txt".format(random.randint(1,10))
        tg_file4 = "training{}.txt".format(random.randint(1,10))
        ot_file0 = "autooutput{}.txt".format(i)
        rs_file0 = "results{}.txt".format(i)

        start = time.time()
        os.system("python3 tagger.py -d {} {} {} {} -t {} -o {}".format(\
            tg_file1,\
            tg_file2,\
            tg_file3,\
            tg_file4,\
            tt_file0,\
            ot_file0))
        end = time.time()
        print(f"time elapsed {tt_file0}: {end - start}")


        # Compare the contents of the HMM tagger output with the reference solution.
        # Store the missed cases and overall stats in results.txt
        with open(ot_file0, "r") as output_file, \
                open(tg_file1, "r") as solution_file, \
                open(rs_file0, "w") as results_file:
            # Each word is on a separate line in each file.
            output = output_file.readlines()
            solution = solution_file.readlines()
            total_matches = 0
            results_file.write(f"test {i}\n")

            # generate the report
            for index in range(len(output)):
                if output[index] != solution[index]:
                    results_file.write(f"Line {index + 1}: "
                                       f"expected <{output[index].strip()}> "
                                       f"but got <{solution[index].strip()}>\n")
                else:
                    total_matches = total_matches + 1

            # Add stats at the end of the results file.
            results_file.write(f"Total words seen: {len(output)}.\n")
            results_file.write(f"Total matches: {total_matches}.\n")

            print("done writing to result")






