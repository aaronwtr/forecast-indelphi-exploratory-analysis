import os
import pandas as pd
from Lindel_prediction import check_pam


if __name__ == '__main__':
    path = "C:/Users/Aaron/Desktop/Nanobiology/MSc/MEP/interpreting-ml-based-drops/FORECasT/candidate_samples/test_data" \
           "/single_nucleotide_insertions_freq_50+"
    guideset = pd.read_csv("guideset_data.txt", sep='\t')
    guideset.set_index('ID', inplace=True)

    oligos = os.listdir(path)
    oligos = [x.split('_')[0] + x.split('_')[1] for x in oligos if x != 'archive']

    for oligo in oligos:
        seq = guideset.loc[oligo]['TargetSequence']

        # Use check_pam to check if the PAM is positioned correctly for this seq. If not, put the oligo in a separate folder
        # if it is the correct PAM, then keep the oligo in the same folder.
        if not check_pam(seq):
            os.rename(f"{path}/{oligo[:5]}_{oligo[5:]}", f"{path}/archive/{oligo[:5]}_{oligo[5:]}")
        else:
            print(f"{oligo} is in the correct PAM position")
