import os

os.makedirs('Data Visualization', exist_ok=True)
os.makedirs('Saved Data', exist_ok=True)
os.makedirs('Results', exist_ok=True)

from datagen import datagen, train_for_seg, segmentation


def main():
    datagen()
    train_for_seg()

    segmentation()


if __name__ == "__main__":
    main()