def get_train_list():
    with open("train.txt", "r") as f:
        return [line.strip() for line in f.readlines()]
