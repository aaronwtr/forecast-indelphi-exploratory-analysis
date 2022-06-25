import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_mutation_type_indel(oligo_list):
    insertions = 0
    deletions = 0
    for oligo in tqdm(oligo_list):
        oligo_data_test = pd.read_pickle(f"{path}test/Tijsterman_Analyser/{oligo}")
        oligo_data_test = oligo_data_test.sort_values(by="Frac Sample Reads", ascending=False)
        top_row = oligo_data_test.iloc[0]
        top_indel = top_row["Indel"]
        if top_indel[0] == "I":
            insertions += 1
        else:
            deletions += 1

    return deletions, insertions


def get_mutation_type_ins(oligo_list):
    insertion_1 = 0
    insertion_2 = 0
    insertion_3_plus = 0
    for oligo in tqdm(oligo_list):
        oligo_data_test = pd.read_pickle(f"{path}test/Tijsterman_Analyser/{oligo}")
        oligo_data_test = oligo_data_test.sort_values(by="Frac Sample Reads", ascending=False)
        top_row = oligo_data_test.iloc[0]
        top_indel = top_row["Indel"]
        if top_indel[0] == "I":
            if top_indel[:2] == "I1":
                insertion_1 += 1
            elif top_indel[:2] == "I2":
                insertion_2 += 1
            else:
                insertion_3_plus += 1

    return insertion_1, insertion_2, insertion_3_plus


def plot_class_imbalance_bar_plot():
    test_ins = pd.read_csv("test_ins.csv")
    train_ins = pd.read_csv("train_ins.csv")

    fig, ax = plt.subplots(figsize=(6, 6))
    ind = [-0.5, 0.5]
    p1 = ax.bar(ind, [int(train_ins['train_ins_1']), int(test_ins['test_ins_1'])], color="#0077ea",
                label="Insertions length 1")
    p2 = ax.bar(ind, [int(train_ins['train_ins_2']), int(test_ins['test_ins_2'])], color="#00ea73",
                label="Insertions length 2")
    p3 = ax.bar(ind, [int(train_ins['train_ins_3_plus']), int(test_ins['test_ins_3_plus'])], color="#8d1eff",
                label="Insertions length 3+", bottom=[int(train_ins['train_ins_1']), int(train_ins['train_ins_2'])])
    ax.bar_label(p1, label_type="center")
    #ax.bar_label(p2, label_type="center")
    #ax.bar_label(p3, label_type="center")
    ax.set_xticks(ind, labels=["Training data", "Testing data"])
    ax.set_xlabel("Data type")
    ax.set_ylabel("Number of occurences")
    plt.ylim(0, 1500)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    path = "E:/Aaron/Nanobiology/MSc/Year3/MEP/"
    test_tijsterman_oligos = os.listdir(path + "test/Tijsterman_Analyser")
    train_tijsterman_oligos = os.listdir(path + "train/Tijsterman_Analyser")

    # train_ins_1, train_ins_2, train_ins_3_plus = get_mutation_type_ins(test_tijsterman_oligos)
    #
    # train_ins = pd.DataFrame({"test_ins_1": [train_ins_1], "test_ins_2": [train_ins_2], "test_ins_3_plus": [train_ins_3_plus]})
    # train_ins.to_csv("test_ins.csv", index=False)

    plot_class_imbalance_bar_plot()
