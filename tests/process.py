import os 

def read_file_in_directory(path):
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), 'r') as f:
                text = f.read()
                print(text)

def list_files(dir, good=False):
    if not os.path.exists(f'{dir}/../labels'):
        os.makedirs(f'{dir}/../labels', exist_ok=True)
    for file in os.listdir(dir):
        if file.endswith(".bmp"):
            print(file)
            file_name = file.split('.')[0]
            label_file = open(f'{dir}/../labels/{file_name}.txt', 'w')
            if good:
                label_file.writelines(["0\n"])
            else:
                label_file.writelines(["1\n"])
            label_file.close()
            print(f"label file {label_file}")


if __name__ == "__main__":
    current_dir = "/Users/xbkaishui/Downloads/syzn/F4面小块数据集/top"
    current_dir = "/Users/xbkaishui/Downloads/syzn/F4面小块数据集/big"
    current_dir = "/opt/product/eoe/data/datasets/chip"
    current_dir = "/opt/product/eoe/data/datasets/hole_detection_small"
    # current_dir = "/Users/xbkaishui/Downloads/syzn/F4面小块数据集/bottom"
    list_files(f'{current_dir}/NG')
    list_files(f'{current_dir}/OK', good=True)
