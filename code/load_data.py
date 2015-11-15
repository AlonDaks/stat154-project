from os import listdir

RELATIVE_DATA_PATH = '../data/'


def document_paths(data_set):
    if data_set == 'train':
        child_paths = [RELATIVE_DATA_PATH + 'train/child/' + i
                       for i in listdir(RELATIVE_DATA_PATH + 'train/child/')
                       if i != '.DS_Store']
        history_paths = [
            RELATIVE_DATA_PATH + 'train/history/' + i
            for i in listdir(RELATIVE_DATA_PATH + 'train/history/')
            if i != '.DS_Store'
        ]
        religion_paths = [
            RELATIVE_DATA_PATH + 'train/religion/' + i
            for i in listdir(RELATIVE_DATA_PATH + 'train/religion/')
            if i != '.DS_Store'
        ]
        science_paths = [
            RELATIVE_DATA_PATH + 'train/science/' + i
            for i in listdir(RELATIVE_DATA_PATH + 'train/science/')
            if i != '.DS_Store'
        ]
        return child_paths + history_paths + religion_paths + science_paths
    if data_set == 'test':
        return [
            RELATIVE_DATA_PATH+ 'test/' + i
            for i in listdir(RELATIVE_DATA_PATH + 'test/')
            if i != '.DS_Store'
        ]
