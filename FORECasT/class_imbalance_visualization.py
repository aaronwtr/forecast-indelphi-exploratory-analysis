import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_mutation_type(oligo_list):
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


def plot_class_imbalance_bar_plot():
    test_dels_ins = pd.read_csv("test_dels_ins.csv")
    train_dels_ins = pd.read_csv("train_dels_ins.csv")

    # plot test_dels and ins in one bar and train dels and ins in another
    fig, ax = plt.subplots(figsize=(6, 6))
    ind = [-0.5, 0.5]
    p1 = ax.bar(ind, [int(train_dels_ins['train_dels']), int(test_dels_ins['test_dels'])], color="#0077ea", label="Deletions")
    p2 = ax.bar(ind, [int(train_dels_ins['train_ins']), int(test_dels_ins['test_ins'])], color="#fb7000", label="Insertions")
    ax.bar_label(p1, label_type="center")
    ax.bar_label(p2, label_type="center")
    ax.set_xticks(ind, labels=["Training data", "Testing data"])
    ax.set_xlabel("Data type")
    ax.set_ylabel("Number of occurences")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    path = "E:/Aaron/Nanobiology/MSc/Year3/MEP/"
    test_tijsterman_oligos = os.listdir(path + "test/Tijsterman_Analyser")
    train_tijsterman_oligos = os.listdir(path + "train/Tijsterman_Analyser")

    # test_dels, test_ins = get_mutation_type(test_tijsterman_oligos)
    #
    # train_dels_ins = pd.DataFrame({"test_dels": [test_dels], "test_ins": [test_ins]})
    # train_dels_ins.to_csv("test_dels_ins.csv", index=False)

    plot_class_imbalance_bar_plot()
