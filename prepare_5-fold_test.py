import numpy as np
from sklearn.model_selection import GroupKFold


def split_5fold(data_root):
    meta_path = data_root + 'all_meta.npy'
    all_data = np.load(meta_path, allow_pickle=True)

    pid_list = [img_dict['family_id'] + img_dict['personal_id'] for img_dict in all_data]
    pid_set = list(set(pid_list))
    id_list = [pid_set.index(pid) for pid in pid_list]

    kf = GroupKFold(n_splits=5)

    count = 0

    X = np.ones(shape=(len(id_list), 1))
    y = np.ones(shape=(len(id_list), 1))

    for count, split_indices in enumerate(kf.split(X, y, id_list)):
        train_idx_test, test_idx_test = split_indices
        train_meta = []
        for j in train_idx_test:
            train_meta.append(all_data[j])
        test_meta = []
        for j in test_idx_test:
            test_meta.append(all_data[j])
        np.save(data_root + 'train_{}.npy'.format(count), train_meta)
        np.save(data_root + 'test_{}.npy'.format(count), test_meta)
        print('{} train : {}'.format(count, len(train_meta)))
        print('{} test : {}'.format(count, len(test_meta)))


if __name__ == '__main__':
    cropped_root = 'nia_cropped/'
    split_5fold(cropped_root)
