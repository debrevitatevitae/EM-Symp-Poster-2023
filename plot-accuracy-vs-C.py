"""Date: 28-9-23
Author: Giorgio Tosti Balducci

Description: plots the mean accuracies of the cross validation runs for different values of the constraint penalty C (in SVMs). Different lines correspond to different values of the kernel hyperparameters (e.g. \gamma for RBF).
"""

from matplotlib import pyplot as plt
import pandas as pd

DATA_FILE_PATH = './accuracy-C-data.csv'
OUT_DIR = './pics/'


def main():

    plt.rcParams['text.usetex'] = True

    # Read csv into a Pandas dataframe
    df = pd.read_csv(DATA_FILE_PATH)

    # Set-up figure and axis for the untrained curves (pre-KTA)
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$C$')
    ax.set_ylabel(r'mean test accuracy')
    ax.set_title(r"OHCP: cross-validation results")

    # Add curves to the semilogx plot
    ax.semilogx(df['C'], df['acc_rbf_gamma1'],
                linestyle='--', label='RBF, $\gamma$=1.0')
    ax.semilogx(df['C'], df['acc_qek_0304'], label='qek0304, rand params')
    ax.semilogx(df['C'], df['acc_qek_0316'], label='qek0316, rand params')
    ax.semilogx(df['C'], df['acc_qek_0608'], label='qek0608, rand params')

    ax.legend()
    ax.grid()

    # Save figure
    fig.savefig(OUT_DIR+'acc-untrained.pdf')

    # Set-up figure and axis for the trained curves (post-KTA)
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$C$')
    ax.set_ylabel(r'mean test accuracy')
    ax.set_title(r"OHCP: cross-validation results post-KTA")

    # Add curves to the semilogx plot
    ax.semilogx(df['C'], df['acc_rbf_trained'],
                linestyle='--', label='RBF, trained')
    ax.semilogx(df['C'], df['acc_qek_0304_trained'],
                label='qek0304, trained')
    ax.semilogx(df['C'], df['acc_qek_0316_trained'],
                label='qek0316, trained')
    ax.semilogx(df['C'], df['acc_qek_0608_trained'],
                label='qek0608, trained')

    ax.legend()
    ax.grid()

    # Save figure
    fig.savefig(OUT_DIR+'acc-trained.pdf')


if __name__ == '__main__':
    main()
