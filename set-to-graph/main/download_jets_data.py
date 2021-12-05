import os
import urllib.request


def mkdir_if_not_exists(dir_name):
    if not (os.path.exists(dir_name) and os.path.isdir(dir_name)):
        os.mkdir(dir_name)


if __name__ == '__main__':
    train_link = 'https://zenodo.org/record/4044628/files/training_data.root?download=1'
    val_link = 'https://zenodo.org/record/4044628/files/valid_data.root?download=1'
    test_link = 'https://zenodo.org/record/4044628/files/test_data.root?download=1'

    print('Creating data directories...')
    mkdir_if_not_exists('data')
    mkdir_if_not_exists('data/train')
    mkdir_if_not_exists('data/validation')
    mkdir_if_not_exists('data/test')

    print('Downloading training data to data/train/training_data.root...', flush=True)
    urllib.request.urlretrieve(train_link, 'data/train/training_data.root')
    print('Downloading validation data to data/validation/valid_data.root...', flush=True)
    urllib.request.urlretrieve(val_link, 'data/validation/valid_data.root')
    print('Downloading test data data/test/test_data.root...', flush=True)
    urllib.request.urlretrieve(test_link, 'data/test/test_data.root')

    print('Done!')

