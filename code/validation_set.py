from load_data import *
import numpy as np
import os

def generate_validation_set():
    child_paths = training_path_by_class('child')
    history_paths = training_path_by_class('history')
    religion_paths = training_path_by_class('religion')
    science_paths = training_path_by_class('science')
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