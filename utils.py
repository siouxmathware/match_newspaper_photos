import json
import os


def write_json_file(filename, data):
    """
    Generic write function for JSON format. ASCII ensured, keys are not sorted.
    :param filename:
    :param data: the data to be written, should not contain numpy arrays
    :return: file being written
    """
    str_ = json.dumps(data,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=True, allow_nan=False)

    with open(filename, 'w', encoding='utf8') as outfile:
        outfile.write(str_)


def read_json_file(filename):
    """
    Reading generic JSON file function
    :param filename:
    :return: the data
    """
    if not os.path.isfile(filename):
        raise RuntimeError(f'Config path does not exist: {filename}')

    with open(filename) as data_file:
        data = json.load(data_file)
    return data
