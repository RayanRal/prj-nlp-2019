import os
import re
from pathlib import Path
from typing import Dict, List, NamedTuple
from zipfile import ZipFile

import requests
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

_re_tokens = re.compile(r"([\w][\w]*'?\w?)")


class Message(NamedTuple):
    id: int
    text: str

Corpus = Dict[str, List[Message]]

_local_filename = "data/1551.zip"
_data_link = "https://github.com/vseloved/prj-nlp-2019/raw/master/tasks/1551.zip"
_uk_chars = set('іїє')


def text2tokens(text: str)->List[str]:
    return _re_tokens.findall(text)


def check_if_file_exist_make_dir(filename: str) -> bool:
    """
    Function performs the following checks:
       if file directory does not exist, directory is created.
    :param filename: name of a new file.
    :return: boolean sign if file is already existed.
    """
    file_path = Path(filename)
    dir_ = str(file_path.parent.absolute())
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    return file_path.is_file()


def download_with_progress(link: str, filename: str):
    file_path = Path(filename)

    dir_ = str(file_path.parent.absolute())
    if not os.path.exists(dir_):
        os.mkdir(dir_)

    if file_path.is_file():
        print(f"File '{file_path.absolute()}' is already existed, downloading was skipped.")
        return

    with open(filename, "wb") as f:
        print("Downloading '%s'" % filename)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            print(f"Length of the downloaded file is unknown, start downloading")
            f.write(response.content)
        else:
            wrote = 0
            total_size: int = int(total_length)
            chunk_size = 1024 * 8
            with tqdm(total=total_size, unit="B") as p_bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    bl_size = f.write(data)
                    wrote += bl_size
                    p_bar.update(bl_size)

    print(f"File downloaded, length = {file_path.stat().st_size} b")


def is_ua_text(text):
    return any(c for c in text.lower() if c in _uk_chars)


def parse_raw_category(raw_text: str) -> List[Message]:
    res = []

    buff = []
    prev_was_empty_line = False

    def flush_buff():
        if buff:
            id_ = int(buff[0])
            text = '\n'.join(buff[1:])
            if is_ua_text(text):
                res.append(Message(id_, text))
        buff.clear()
        nonlocal prev_was_empty_line
        prev_was_empty_line = False

    lines = raw_text.splitlines()
    lines_num = len(lines)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line and prev_was_empty_line:
            if i + 1 < lines_num:
                if lines[i + 1].isdigit():
                    flush_buff()
        else:
            buff.append(line)
        prev_was_empty_line = not line
    flush_buff()

    return res


def load_corpora() -> Corpus:
    res = {}
    dir_ = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dir_, _local_filename)
    if not check_if_file_exist_make_dir(filename):
        download_with_progress(_data_link, filename)
    with ZipFile(filename) as zip_corpus:
        all_names = zip_corpus.namelist()
        all_names = [name for name in all_names if name.endswith('.txt')]
        for name in all_names:
            with zip_corpus.open(name) as cat:
                raw_text = cat.read().decode(encoding='utf-8', errors='replace')
            messages = parse_raw_category(raw_text)
            category_name = os.path.basename(os.path.splitext(name)[0])
            res[category_name] = messages
    return res


def load_train_and_test() -> (Corpus, Corpus):
    full_corpora = load_corpora()
    res_train, res_test = {}, {}
    for name, messages in full_corpora.items():
        train, test = train_test_split(messages, random_state=1974)
        res_train[name] = train
        res_test[name] = test
    return res_train, res_test


def main():
    texts = load_corpora()
    texts_list = list(texts.items())
    print(f"Categories found: {len(texts_list)}")
    min_cat_len = min(len(val) for name, val in texts_list)
    min_cat_ind, min_cat_name = next((i, x[0]) for i, x in enumerate(texts_list) if len(x[1]) == min_cat_len)
    print(f"Min len cat: '{min_cat_name}' ({min_cat_len})")
    max_cat_len = max(len(val) for name, val in texts_list)
    max_cat_ind, max_cat_name = next((i, x[0]) for i, x in enumerate(texts_list) if len(x[1]) == max_cat_len)
    print(f"Max len cat: '{max_cat_name}' ({max_cat_len})")

    train, test = load_train_and_test()
    train_total_len = sum(len(m) for k, m in train.items())
    test_total_len = sum(len(m) for k, m in test.items())
    print(f"Total train messages: {train_total_len}, total test messages: {test_total_len}")


if __name__ == "__main__":
    main()
