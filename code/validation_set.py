from load_data import *
import numpy as np
import os

def generate_validation_set():
    child_paths = np.array(training_path_by_class('child', 'train'))
    history_paths = np.array(training_path_by_class('history', 'train'))
    religion_paths = np.array(training_path_by_class('religion', 'train'))
    science_paths = np.array(training_path_by_class('science', 'train'))
    random_child_indexes = np.random.choice(
        range(len(child_paths)),
        size=(len(child_paths) * .15) / 1,
        replace=False)
    random_history_indexes = np.random.choice(
        range(len(history_paths)),
        size=(len(history_paths) * .15) / 1,
        replace=False)
    random_religion_indexes = np.random.choice(
        range(len(religion_paths)),
        size=(len(religion_paths) * .15) / 1,
        replace=False)
    random_science_indexes = np.random.choice(
        range(len(science_paths)),
        size=(len(science_paths) * .15) / 1,
        replace=False)
    validation_document_paths = np.concatenate((child_paths[
        random_child_indexes], history_paths[
            random_history_indexes], religion_paths[
                random_religion_indexes], science_paths[random_science_indexes]
                                                ))
    for f in validation_document_paths:
        os.rename(f, f.replace('train', 'validate'))


if __name__ == '__main__':
    generate_validation_set()