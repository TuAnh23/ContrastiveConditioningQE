def load_text_file(file_path):
    """
    Load text file into a list, each item is a line of the file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def write_text_file(lines, file_path):
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")
