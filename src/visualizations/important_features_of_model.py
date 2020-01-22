from joblib import load
import argparse
import numpy as np
import pandas as pd
import csv


def main(args):
    clf = load(args.model_path)
    feature_importances = clf.feature_importances_
    feature_weights = np.argsort(feature_importances)
    indices, associations = [], []
    for item in feature_weights:
        print(item, feature_importances[item])
        indices.append(item)
        associations.append(feature_importances[item])

    csv_write_path = '/home/adnan/PycharmProjects/BUET-CSE-6411-project/data/{}'.format('RFC_83_7.csv')
    with open(csv_write_path, 'w', newline='') as csvfile:
        fieldnames = ['index', 'association']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for index, association in zip(indices, associations):
            writer.writerow({'index': str(index),'association': str(association)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/adnan/PycharmProjects/BUET-CSE-6411-project/data/models/random_forest_classifier_83.79925452609159.joblib')
    parser_args = parser.parse_args()

    main(parser_args)
