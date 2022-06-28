import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def get_ro_int_freqs(glob_path, ro_type):
    oligos = os.listdir(glob_path + ro_type)
    first_freqs = []
    second_freqs = []
    third_freqs = []
    fourth_freqs = []
    fifth_freqs = []

    for oligo in tqdm(oligos):
        oligo_test_data = pd.read_pickle(glob_path + ro_type + "/" + oligo)
        frac_sample_reads = oligo_test_data["Frac Sample Reads"]
        repair_outcomes_sorted = list(frac_sample_reads.sort_values(ascending=False))
        first_freqs.append(repair_outcomes_sorted[0])
        second_freqs.append(repair_outcomes_sorted[1])
        third_freqs.append(repair_outcomes_sorted[2])
        fourth_freqs.append(repair_outcomes_sorted[3])
        fifth_freqs.append(repair_outcomes_sorted[4])

    return first_freqs, second_freqs, third_freqs, fourth_freqs, fifth_freqs


def generate_boxplot(first_freqs, sec_freqs, third_freqs, fourth_freqs, fifth_freqs):
    """
    This function should generate three boxplots next to eachother of the first_freqs, sec_freqs and res_freqs.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot([first_freqs, sec_freqs, third_freqs, fourth_freqs, fifth_freqs])
    ax.set_xticklabels(["First", "Second", "Third", "Fourth", "Fifth"])
    ax.set_ylabel("Integration frequency")
    ax.set_xlabel("Repair outcome")
    ax.set_ylim(0, 0.2)
    plt.show()
    plt.show()


if __name__ == "__main__":
    path = "E:/Aaron/Nanobiology/MSc/Year3/MEP/candidate_samples/test_data/"
    current_ro_type = "dinucleotide_insertions_most_freq"

    first_freq, second_freq, third_freq, fourth_freq, fifth_freq = get_ro_int_freqs(path, current_ro_type)

    generate_boxplot(first_freq, second_freq, third_freq, fourth_freq, fifth_freq)
