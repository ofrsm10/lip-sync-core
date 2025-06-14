import os

import pandas as pd
from sklearn.preprocessing import minmax_scale

from constants.constants import DATA_SAMPLES_PATH
from utils.plots import save_boundary_plot, save_files_and_plots


def normalize_dataframe(matrix):
    df = pd.DataFrame(matrix, columns=['ratio', 'area', 'teeth', 'tongue'])
    df['area'] = minmax_scale(df['area'])
    # fig = plt.figure()
    # for col in df.columns:
    #     plt.plot(df[col], label=col)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    # plt.close(fig)

    return df


def crop_helper(d1, d, min_seq_len1=5, min_seq_len2=17, max_break_len=6, mirror=False):
    start_indices = []
    end_indices = []
    start_index = None

    if mirror:
        d1_area = d1["area1"]
    else:
        d1_area = d1["area"]
    for i, area in enumerate(d1_area):
        if area > 600 and start_index is None:
            start_index = i
        elif area < 500 and start_index is not None:
            start_indices.append(start_index)
            end_indices.append(i)
            start_index = None
    if start_index is not None:
        start_indices.append(start_index)
        end_indices.append(len(d1) - 1)

    merged_start_indices = []
    merged_end_indices = []
    start_index = start_indices[0]
    end_index = end_indices[0]
    for i in range(1, len(start_indices)):
        if start_indices[i] - end_index <= max_break_len:
            end_index = end_indices[i]
        else:
            if end_index - start_index + 1 >= min_seq_len1:
                merged_start_indices.append(start_index)
                merged_end_indices.append(end_index)
            start_index = start_indices[i]
            end_index = end_indices[i]
    if end_index - start_index + 1 >= min_seq_len1:
        merged_start_indices.append(start_index)
        merged_end_indices.append(end_index)

    # Crop both dataframes into the same number of continuous sequences
    cropped_df = []
    for start, end in zip(merged_start_indices, merged_end_indices):
        if 55 >= end - start >= min_seq_len2:
            cropped_df.append(d.iloc[start - 1:end + 5])

    return cropped_df


def crop_sequences(df1, df2, df3):
    print("Cropping data frames")
    crop1 = crop_helper(df1, df2)
    crop2 = crop_helper(df1, df3, mirror=True)
    print(f"Cropped {len(crop1)} samples..")
    print(f"Cropped {len(crop2)} samples from mirrored video..")
    return crop1, crop2


def save_features_for_word(word, frames_features, frames_features1, x_d, y_d):
    word_path = os.path.join(DATA_SAMPLES_PATH, word)
    if not os.path.exists(word_path):
        os.mkdir(word_path)
    k = 1
    while True:
        new_dir1 = "iteration" + "{:03d}".format(k)
        if not os.path.exists(f'{word_path}/{new_dir1}'):
            os.mkdir(f'{word_path}/{new_dir1}')
            new_dir2 = new_dir1 + "_mirror"
            os.mkdir(f'{word_path}/{new_dir2}')
            break
        k += 1
    with open(f'{word_path}/{new_dir1}/features_raw.csv', 'w', newline='') as file:
        features_df = pd.DataFrame(frames_features)
        features_df.to_csv(file)
    with open(f'{word_path}/{new_dir2}/features_raw.csv', 'w', newline='') as file:
        m_features_df = pd.DataFrame(frames_features1)
        m_features_df.to_csv(file)
    data = {"features": frames_features, "area": x_d, "area1": y_d}
    collection_df = pd.DataFrame(data)
    save_boundary_plot(collection_df, f"{word_path}")
    cropped_features, cropped_m_features = crop_sequences(collection_df, features_df, m_features_df)
    save_files_and_plots(cropped_features, cropped_m_features, f"{word_path}/{new_dir1}")

