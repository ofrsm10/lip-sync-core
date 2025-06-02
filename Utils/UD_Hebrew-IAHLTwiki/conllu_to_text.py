import os
import re
import json
import string


def has_english_character(line):
    for char in line:
        if ord(char) < 127 and not (char in (" ", ".")):  # ASCII range
            return True
    return False


def has_number(line):
    for char in line:
        if char.isdigit():
            return True
    return False


def remove_parentheses(text):
    # Define the regex pattern to match parentheses and their content
    pattern = r'\([^)]*\)'
    # Remove the matched pattern using re.sub
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def extract_conllu_info(directory, output_file):
    new_lines = []
    for filename in os.listdir(directory):
        if filename.endswith(".conllu"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.readlines()
            sentence_id = ''
            text = ''

            for line in lines:
                if line.startswith("# meta::domain ="):
                    new_lines.append(line)
                elif line.startswith("# sent_id ="):
                    sentence_id = re.search(r"# sent_id =(.+)", line).group(1).strip()
                elif line.startswith("# text = "):
                    text = re.search(r"# text = (.+)", line).group(1)
                    text = remove_parentheses(text)
                    text_list = text.split(" ")
                    text_list = [elem for elem in text_list if len(elem) >= 2]
                    if 13 <= len(text_list):
                        new_lines.append(sentence_id + ": " + text + "\n")
                elif line.startswith('#'):
                    continue

    with open(output_file, "w", encoding="utf-8") as out_file:
        out_file.writelines(new_lines)


def parse_conllu_directory(directory_path, num_words=10):
    parser = {}

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.conllu'):
            file_path = os.path.join(directory_path, file_name)
            dataset = None
            subject = None
            num_id = None
            upper_index = None
            lower_index = None
            flag = True

            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        if line.startswith('# sent_id'):

                            _, id = line.split('=', 1)
                            dataset, rest = id.split('_', 1)
                            dataset = dataset.strip()
                            subject, num_id = rest.split('-', 1)

                        elif line.startswith("# text = "):
                            text = re.search(r"# text = (.+)", line).group(1)
                            text = remove_parentheses(text)
                            text_list = text.split(" ")
                            text_list = [elem for elem in text_list if len(elem) >= 2]
                            flag = False if not (3 <= len(text_list) <= 12) else True

                        elif line.startswith('#'):
                            continue

                        elif line[0].isdigit() and flag:
                            index = line.split('\t')[0]
                            word = line.split('\t')[1]
                            if len(index.split("-")) == 2:
                                lower_index = index.split("-")[0]
                                upper_index = index.split("-")[1]
                                parser.setdefault(dataset, {}).setdefault(subject, {}).setdefault(num_id,
                                                                                                  {}).setdefault(
                                    index)
                                parser[dataset][subject][num_id][index] = word
                            elif upper_index:
                                if lower_index <= index <= upper_index:
                                    parser.setdefault(dataset, {}).setdefault(subject, {}).setdefault(num_id,
                                                                                                      {}).setdefault(
                                        index)
                                    parser[dataset][subject][num_id][index] = word
                                if index == upper_index:
                                    lower_index, upper_index = None, None
    return parser


import json


def count_word_appearances(data):
    word_count = {}

    def traverse_dict(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                traverse_dict(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse_dict(item)
        elif isinstance(obj, str):
            word_count[obj] = word_count.get(obj, 0) + 1

    traverse_dict(data)
    return word_count


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def write_text(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(data, ensure_ascii=False))


def count_unique_words(text_file_path):
    word_count = {}

    with open(text_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove punctuation and convert to lowercase
            line = line.translate(str.maketrans('', '', string.punctuation)).lower()
            words = line.split()[1:]

            for word in words:
                word_count[word] = word_count.get(word, 0) + 1

    return word_count


def create_word_count_json(text_file_path, output_file_path):
    word_count = count_unique_words(text_file_path)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(word_count, file, ensure_ascii=False, indent=2)


directory_path = "/Users/ofersimchovitch/PycharmProjects/lipSyncBeta/UD_Hebrew-IAHLTwiki"


json_output_file_path = 'parsed_data.json'
text_output_file_path = "sentences_13.txt"

# parsed_data = parse_conllu_directory(directory_path)
# save_json(parsed_data, json_output_file_path)

extract_conllu_info(directory_path, text_output_file_path)

# Count word appearances
#
# input_file_path = 'parsed_data.json'
# with open(input_file_path, 'r', encoding='utf-8') as file:
#     json_data = json.load(file)
# word_count = count_word_appearances(json_data)
# output_file_path = 'word_count.json'
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(word_count, file, ensure_ascii=False, indent=2)
#
# create_word_count_json(text_output_file_path, output_file_path)
