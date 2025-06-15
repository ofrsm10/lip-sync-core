import os

import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from constants.constants import STATS_PATH, CLASSES


def plot_train_loss(train_loss):
    plt.figure(1)
    plt.plot(train_loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(STATS_PATH, "Loss.png"))
    plt.close(1)

def plot_train_accuracy(train_acc):
    plt.figure(2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(STATS_PATH, "Accuracy.png"))
    plt.close(2)

def save_boundary_plot(df2, base_path):
    file_path = os.path.join(base_path, "filtered_boundary.png")
    x = df2["area"].values
    y = df2["area1"].values

    plt.figure()
    plt.plot(x)
    plt.plot(y)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    class_counts = np.bincount(y_true, minlength=len(CLASSES))  # calculate class counts overall
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / class_counts[:, np.newaxis] * 100  # calculate percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im = ax1.imshow(cm_percent, cmap='Blues')
    ax1.set_title('Confusion Matrix')
    ax1.set_xticks(range(len(CLASSES)))
    ax1.set_xticklabels(CLASSES, rotation=45)
    ax1.set_yticks(range(len(CLASSES)))
    ax1.set_yticklabels(CLASSES)
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm_percent[i, j] > 0:
                ax1.text(j, i, '{:.1f}%'.format(cm_percent[i, j]), ha='center', va='center', color='black')
    cbar = ax1.figure.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel('Percentage', rotation=-90, va="bottom")
    ax2.bar(range(len(CLASSES)), class_counts, align='center')
    ax2.set_xticks(range(len(CLASSES)))
    ax2.set_xticklabels(CLASSES, rotation=45)
    ax2.set_xlabel('True label')
    ax2.set_ylabel('Class count')
    fig.tight_layout()
    plt.savefig(os.path.join(STATS_PATH, "Confusion_Matrix.png"))

def plot_umap(features, labels):
    features = np.vstack(features)
    labels = np.hstack(labels)
    unique_labels = np.unique(labels)

    # Convert labels to integers for better compatibility with Matplotlib
    labels = np.array(labels).astype(int)
    # Reverse the Hebrew text for legend display
    reversed_classes = [label[::-1] for label in CLASSES]
    umap_features = umap.UMAP(n_components=3).fit_transform(features)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in unique_labels:
        mask = labels == label
        ax.scatter(umap_features[:, 0][mask], umap_features[:, 1][mask], umap_features[:, 2][mask],
                   label=reversed_classes[label])  # Reverse the label
    ax.legend()
    plt.savefig(os.path.join(STATS_PATH, "UMAP.png"))
    plt.close()

def save_files_and_plots(c_features, m_features, base_path):
    print(f"Saving..")
    for i, df in enumerate(c_features):
        path = "sequence_{:03d}".format(i)
        path = f'{base_path}/{path}'
        os.mkdir(path)

        file_path = os.path.join(path, "features_norm" + ".csv")
        df.columns = ['ratio', 'area', 'teeth', 'tongue']
        # df['area'] = df['area'] / 17000
        df['area'] = minmax_scale(df['area'])
        df['area'] = df['area'].round(4)
        df.to_csv(file_path)

        plt.figure()
        plt.title(f'normalized features')
        for col in df.columns:
            plt.plot(df[col], label=col)
        plt.tight_layout()
        file_path = os.path.join(path, f"features_norm.png")
        plt.legend()
        plt.savefig(file_path)
        # plt.show()
        plt.close()

    print(f"Saving mirrored..")
    base_path = base_path + '_mirror'
    for i, df in enumerate(m_features):
        path = "sequence_{:03d}".format(i)
        path = f'{base_path}/{path}'
        os.mkdir(path)

        file_path = os.path.join(path, "features_norm" + ".csv")
        df.columns = ['ratio', 'area', 'teeth', 'tongue']
        df['area'] = minmax_scale(df['area'])
        df.to_csv(file_path)


    # apply t-SNE to the features
    # tsne = TSNE(n_components=3)
    # tsne_features = tsne.fit_transform(features)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for label in unique_labels:
    #     mask = labels == label
    #     ax.scatter(tsne_features[:, 0][mask], tsne_features[:, 1][mask], tsne_features[:, 2][mask],
    #                label=reversed_classes[label])  # Reverse the label
    # ax.legend()
    # plt.savefig(os.path.join(full_path, "T-SNE.png"))
    # plt.close()
    #
    # # apply PCA to the features
    # pca = PCA(n_components=3)
    # pca_features = pca.fit_transform(features)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for label in unique_labels:
    #     mask = labels == label
    #     ax.scatter(pca_features[:, 0][mask], pca_features[:, 1][mask], pca_features[:, 2][mask],
    #                label=reversed_classes[label])  # Reverse the label
    # ax.legend()
    # plt.savefig(os.path.join(full_path, "PCA.png"))
    # plt.close()