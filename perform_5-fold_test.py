import argparse
import csv
import datetime
import os
import sys

import numpy as np

from get_exp_environment import save_env
from main_ae import main, validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()

    # training
    data_root = 'nia_cropped/'
    batch_size = 128
    train_meta_path_temp = data_root + 'train_{}.npy'
    test_meta_path_temp = data_root + 'test_{}.npy'
    leave_subject = 0
    learning_rate = 0.001
    epoch = 10
    resume = None
    log_dir = 'result_model'
    pred_image = None
    pred_model = None
    is_mean_variance = True
    num_RS = 5

    log_dir = log_dir + '/' if log_dir[-1] != '/' else log_dir
    os.makedirs(log_dir, exist_ok=True)
    save_env(log_dir)

    with open(log_dir + '2_result.csv', 'w') as fid:
        csv_writer = csv.writer(fid)
        csv_writer.writerow(['1) timestamp(start): ', str(datetime.datetime.now())])
        csv_writer.writerow(['2) execution command: ', 'python3 ' + ' '.join(sys.argv)])
        csv_writer.writerow(['3) result for each item'])

        maes = []
        ag_accs = []
        for i in range(0, num_RS):
            train_meta_path = train_meta_path_temp.format(i)
            test_meta_path = test_meta_path_temp.format(i)
            exp_name = '{}'.format(i)
            if args.train:
                mae, ag_acc, log_dict = main(exp_name=exp_name,
                                             batch_size=batch_size,
                                             train_meta_path=train_meta_path,
                                             test_meta_path=test_meta_path,
                                             leave_subject=leave_subject,
                                             learning_rate=learning_rate,
                                             epoch=epoch,
                                             resume=resume,
                                             result_directory=log_dir,
                                             pred_image=pred_image,
                                             pred_model=pred_model,
                                             is_mean_variance=is_mean_variance)
            else:
                model_path = os.path.join(log_dir, "model_{}".format(exp_name))
                mae, ag_acc, log_dict = validation(model_path, test_meta_path)
            csv_writer.writerow(
                ['3-{}) {} test from {} RS protocol'.format(i, i, num_RS)])
            key_list = list(log_dict.keys())
            csv_writer.writerow([''] + key_list)
            for i in range(0, len(log_dict[key_list[0]])):
                record = [''] + [log_dict[key][i] for key in key_list]
                csv_writer.writerow(record)

            maes.append(mae)
            ag_accs.append(ag_acc)
        csv_writer.writerow(['4) intermediate results for the final value'])
        for i, zipped in enumerate(zip(maes, ag_accs)):
            mae, ag_acc = zipped
            csv_writer.writerow(['{}-test MAE :'.format(i), '{:.2f}'.format(mae),
                                 '{}-test age group classification accuracy :'.format(i), '{:.3f}'.format(ag_acc)])
        csv_writer.writerow(
            ['5) the final value'])
        csv_writer.writerow(
            ['average MAE', '{:.2f}'.format(np.mean(maes))])
        csv_writer.writerow(
            ['average age group classification accuracy', '{:.3f}'.format(np.mean(ag_accs))])
        csv_writer.writerow(['6) timestamp(end): ', str(datetime.datetime.now())])
