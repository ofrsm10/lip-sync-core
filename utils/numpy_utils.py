import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


def plot_clusters(data, labels):
    data = data.reshape(data.shape[0], -1)

    pca = PCA(n_components=2)
    components = pca.fit_transform(data)

    x = components[:, 0]
    y = components[:, 1]

    fig, ax = plt.subplots()

    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        ax.scatter(x[mask], y[mask], label=label)

    ax.legend()
    plt.show()
    plt.savefig("Data Clusters.png")


def interpolate_matrix(matrix, new_size):
    interp_matrix = np.zeros((new_size, matrix.shape[1]))
    for col in range(matrix.shape[1]):
        column = matrix[:, col]
        x = np.linspace(0, 1, column.size)
        f = interp1d(x, column)
        new_x = np.linspace(0, 1, new_size)
        interp_matrix[:, col] = f(new_x)
    return interp_matrix


def pad_sequence(sequence, max_len, padding_value=0):
    padded_sequence = np.zeros((max_len, sequence.shape[1]), dtype=np.float32)
    seq_len = sequence.shape[0]
    padded_sequence[:seq_len, :] = sequence
    padded_sequence[seq_len:, :] = padding_value
    return padded_sequence
