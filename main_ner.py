import os, random, logging
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import GroupKFold
import logging
from lib.util import load_data_cv, load_tokenizer, train_val_split_doc, make_idx, make_train_vecs, make_test_vecs, save_csv, save_ner_result, cut_length, train_val_split_doc
from lib.loop import train_val_loop_ner, test_loop_ner
import argparse


def main():
    logger.info("----------{0}の実験を開始----------".format(hyper.exp_name))
    # 実験データ
    df = load_data_cv(hyper)
    # 系列長を設定
    df = cut_length(df, hyper.max_words)
    kf = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(kf.split(df, df, df["name"])):
        logger.info("----------{0}-foldの実験を開始----------".format(fold))
        # 学習とテストに分割
        X_train, X_test = df.iloc[train_index].copy(), df.iloc[test_index].copy()
        # 学習から検証を取得
        X_train, X_val = train_val_split_doc(X_train)
        # トーカナイザー
        bert_tokenizer = load_tokenizer(hyper)
        # 検証
        X_train, X_val = train_val_split_doc(X_train)
        # ベクトル
        tag2idx, _ = make_idx(pd.concat([X_train, X_val, X_test]), hyper)
        # 訓練ベクトルを作成
        train_vecs, ner_train_labels = make_train_vecs(X_train, bert_tokenizer, tag2idx)
        # 検証
        val_vecs, ner_val_labels = make_test_vecs(X_val, bert_tokenizer, tag2idx, "NER")
        # テスト
        test_vecs, ner_test_labels = make_test_vecs(X_test, bert_tokenizer, tag2idx, "NER")
        logger.info("train: {0}, val: {1}, test: {2}".format(len(train_vecs), len(val_vecs), len(test_vecs)))
        # 学習
        train_val_loop_ner(train_vecs, ner_train_labels,
                           X_val, val_vecs, ner_val_labels, 
                           tag2idx, fold, hyper, device, logger)
        # テスト
        res_df, ner_res = test_loop_ner(X_test, test_vecs, ner_test_labels, fold, tag2idx, hyper, device)
        # 保存
        save_ner_result(ner_res, hyper, fold, tag2idx)
        save_csv(res_df, hyper, fold, "NER")


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('use cuda device')
        seed=1478754
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        print('use cpu')
        device = torch.device('cpu')
        torch.manual_seed(999)
        np.random.seed(999)

    
    # 引数
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert_path', type=str, default='/home/shibata/Desktop/BERT/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K')
    parser.add_argument('--neologd_path', type=str, default='/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    parser.add_argument('--manbyo_path', type=str, default="/home/shibata/Desktop/resource/MANBYO_201907_Dic-utf8.dic")
    parser.add_argument('--data_path', type=str, default="../preprocess/brat2csv/data/csv/UTH_CR_conll_format_arbitrary_UTH.csv")
    parser.add_argument('--exp_name', type=str, default="UTH")
    parser.add_argument('--bert_type', type=str, default="UTH")
    
    parser.add_argument('--max_words', type=int, default=510)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--skip_epoch', type=int, default=0)

    parser.add_argument('--task', type=str, default='Pipeline')
    parser.add_argument('--idx_flag', type=str, default='F')

    hyper = parser.parse_args()

    # フォルダ作成
    for SAMPLE_DIR in ["./logs", "./models/{0}/{1}/NER".format(hyper.task, hyper.bert_type), "./results/{0}/{1}/NER".format(hyper.task, hyper.bert_type)]:
        if not os.path.exists(SAMPLE_DIR):
            os.makedirs(SAMPLE_DIR)

    # ログの出力名を設定（1）
    logger = logging.getLogger('LoggingTest')
    # ログレベルの設定（2）
    logger.setLevel(10)
    # ログのコンソール出力の設定（3）
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    # ログのファイル出力先を設定（4）
    fh = logging.FileHandler('./logs/Pipeline_NER_{0}.log'.format(hyper.bert_type), "w")
    logger.addHandler(fh)
    # ログの出力形式の設定
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    # hyper = Hyper(os.path.join('experiments',"{0}_NER_icorpus_full".format(bert_type) + '.json'))
    main()