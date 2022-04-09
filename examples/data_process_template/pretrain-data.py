import numpy as np
import csv
import os

datasetPath = "data/"
kmer = 3

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers

fileList = os.listdir(datasetPath)
fileList.sort()
outfileList = str(kmer) + "_mer/"

for filename in fileList:
    print(filename)
    os.makedirs(outfileList + filename)

    train = datasetPath + filename + '/' + 'train.data'
    test = datasetPath + filename + '/' + 'test.data'
    outtrain = outfileList + filename + '/' + 'train.tsv'
    outtest = outfileList + filename + '/' + 'dev.tsv'

    trainfile = open(outtrain, 'a')
    trainfile.write("sequence" + '\t' + "label" + '\n')
    testfile = open(outtest, 'a')
    testfile.write("sequence" + '\t' + "label" + '\n')

    with open(train) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            trainfile.write(seq2kmer(row[1], kmer) + '\t' + row[2] + '\n')
    with open(test) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            testfile.write(seq2kmer(row[1], kmer) + '\t' + row[2] + '\n')

    trainfile.close()
    testfile.close()




