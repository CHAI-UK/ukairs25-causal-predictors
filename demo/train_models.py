import os
import sys
import argparse

from sklearn.linear_model import LogisticRegression
from pickle import dump

from my_utils import init_features, get_features, get_labels, sample, define_setup


def generate_train_samples(n, adj, mechanism, noise):
    return sample(n, adj, mechanism, noise)

def get_model(opt):
    if opt.model == 'lr':
        m = LogisticRegression()
    else:
        raise ValueError('Incorrect model selection')
    
    return m

def init_models(opt):
    models = {}
    models['causal'] = get_model(opt)
    models['all'] = get_model(opt)

    return models

def train_models(X, y, models, features):
    for name in models:
        models[name].fit(X[:, features[name]], y)

def store_models(models, opt):
    for m in models:
        with open(f"{opt.output_path}/model_{opt.case}_{opt.conf}_{opt.scm}_{opt.noise}_{opt.model}_{opt.n_samples}_{m}.pkl", "wb") as f:
            dump(models[m], f, protocol=5)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--case', type=str, choices=['3-conf', 'TC-conf', 'TAC-conf', 'TS-conf', 'SC-conf'], default='3-conf')
    parser.add_argument('--conf', type=str, choices=['hidden', 'observed'], default='observed')
    parser.add_argument('--noise', type=str, choices=['normal'], default='normal')
    parser.add_argument('--scm', type=str, choices=['linear'], default='linear')
    parser.add_argument('--model', type=str, choices=['lr'], default='lr')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('-o', type=str, dest='output_path', default='./')

    return parser

def main():
    parser = get_parser()
    opt = parser.parse_args()

    if not os.path.isdir(opt.output_path):
        os.mkdir(opt.output_path)

    adj, noise, mechanism, _  = define_setup(opt.case, opt.noise, opt.scm)

    train_dataset = generate_train_samples(opt.n_samples, adj, mechanism, noise)
    X_train = get_features(train_dataset)
    y_train = get_labels(train_dataset)

    m = init_models(opt)
    f = init_features(opt.case, opt.conf)
    train_models(X_train, y_train, m, f)

    store_models(m, opt)

    return 0

if __name__ == '__main__':
    sys.exit(main())