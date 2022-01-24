import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from pathlib import Path
import json
from transformers import BertTokenizer
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as ner_eval


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def load_data_cv(hyper):
    df = pd.read_csv(hyper.data_path)
    list_df = []
    for ids in df["unique_no"].unique():
        tmp_df = df[df["unique_no"] == ids]
        if tmp_df.shape[0] <= hyper.max_words:
            list_df.append(tmp_df)
    return pd.concat(list_df) 


def load_data_for_re(hyper):
    list_df = []
    for i in range(0, 5, 1):
        list_df.append(pd.read_csv("./results/{0}/{1}/NER/{2}.csv".format(hyper.task, hyper.exp_name, i)))
    return pd.concat(list_df) 


def load_tokenizer(hyper):
    return BertTokenizer(Path(hyper.bert_path) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)


def cut_length(df, num_words):
    list_df = []
    for ids in df["unique_no"].unique():
        tmp_df = df[df["unique_no"]==ids]
        if tmp_df.shape[0] <= num_words:
            list_df.append(tmp_df)
    return pd.concat(list_df)   


def train_val_split_doc(X_train):
    # 訓練データのid一覧
    ids = list(sorted(X_train["name"].unique()))
    # シャッフル
    random.seed(1478754)
    random.shuffle(ids)
    # 分割
    train_ids, val_ids = ids[:int(len(ids) * 0.8)], ids[int(len(ids) * 0.8):]
    # 確認
    assert len(set(ids[:int(len(ids) * 0.8)]) & set((ids[int(len(ids) * 0.8):])))==0, "trainとvalが重複してますよ"
    # 分割
    X_train, X_val = X_train[X_train["name"].isin(train_ids)].copy(), X_train[X_train["name"].isin(val_ids)].copy()
    return X_train, X_val


def make_idx(df, hyper):
    if hyper.idx_flag == "T":
        # タグ
        tag_vocab = list(sorted(set([x for x in df["IOB"]])))
        tag2idx = {x: i + 2 for i, x in enumerate(tag_vocab)}
        tag2idx["PAD"] = 0
        tag2idx["UNK"] = 1
        # 関係
        rel_vocab = list(sorted(set([y for x in df["rel_type"] for y in x.split(",")])))
        rel2idx = {x: i + 2 for i, x in enumerate(rel_vocab)}
        rel2idx["PAD"] = 0
        rel2idx["UNK"] = 1
    else:
        tag_vocab = list(sorted(set([x for x in df["IOB"]])))
        tag2idx = {x: i + 1 for i, x in enumerate(tag_vocab)}
        tag2idx["PAD"] = 0
        # 関係
        # rel_vocab = list(sorted(set([y for x in df["rel_type"] for y in x.split(",")])))
        # rel2idx = {x: i + 1 for i, x in enumerate(rel_vocab)}
        # rel2idx["PAD"] = 0
        # 修正版rel2idx
        rel_vocab = list(sorted(set([y for x in df["rel_type"] for y in x.split(",")])))
        rel2idx = {}
        last_value = 1
        for _, x in enumerate(rel_vocab):
            if x == "None":
                pass
            else:
                rel2idx["R-" + x] = last_value
                rel2idx["L-" + x] = last_value + 1
                last_value +=2
        rel2idx["PAD"] = 0
        rel2idx["None"] = max(list(rel2idx.values())) + 1
    return tag2idx, rel2idx


def make_train_vecs(df, tokenizer, tag2idx):
    vecs1, vecs2 = [], []
    # テキスト
    for no in df["unique_no"].unique():
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        # NER
        ner = [tag2idx[x] for x in list(tmp_df["IOB"])]
        # REL
        # ADD
        vecs1.append(ids)
        vecs2.append(ner)
    return vecs1, vecs2


def make_test_vecs(df, tokenizer, tag2idx, exp_type):
    vecs1, vecs2 = [], []
    # テキスト
    for no in df["unique_no"].unique():
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        # NERは正解タグを使用
        ner = [tag2idx[x] if x in tag2idx else tag2idx["UNK"] for x in list(tmp_df["IOB"])]
        # NER
        # if exp_type == "NER":
            # ner = [tag2idx[x] if x in tag2idx else tag2idx["UNK"] for x in list(tmp_df["IOB"])]
        # else:
            # ner = [tag2idx[x] if x in tag2idx else tag2idx["UNK"] for x in list(tmp_df["pred_IOB"])]
        # REL
        # ADD
        vecs1.append(ids)
        vecs2.append(ner)
    return vecs1, vecs2


def create_re_labels(df, rel2idx):
    gold_labels = []
    for ids in df["unique_no"].unique():
        tmp_df = df[df["unique_no"]==ids]
        # 固有表現、タグ、serialを取得する
        tokens, labels, indexs = list(tmp_df["word"]), list(tmp_df["IOB"]), list(tmp_df["serial"])
        seqs, tags, ids = [], [], []
        for i in range(0, len(tokens), 1):
            if labels[i].startswith("B-"):
                if i == len(labels) - 1:
                    seqs.append(tokens[i])
                    tags.append(labels[i])
                    ids.append([int(indexs[i])])
                else:
                    tmp1, tmp2, tmp3 = [tokens[i]], [labels[i]], [int(indexs[i])]
                    for j in range(i+1, len(tokens), 1):
                        if labels[j].startswith("I-"):
                            tmp1.append(tokens[j])
                            tmp2.append(labels[j])
                            tmp3.append(int(indexs[j]))
                            if j ==  len(labels) - 1:
                                seqs.append(" ".join(tmp1))
                                tags.append(" ".join(tmp2))
                                ids.append(tmp3)
                        else:
                            seqs.append(" ".join(tmp1))
                            tags.append(" ".join(tmp2))
                            ids.append(tmp3)
                            break  

        # 関係、tailの位置、headの位置を得る
        index2unnamed = {y: x for x, y in zip(tmp_df["serial"], tmp_df["index"])}
        gold_rels, gold_tails, gold_unnamed = [], [], []
        for index, types, tails in zip(tmp_df["serial"], tmp_df["rel_type"], tmp_df["rel_tail"]):
            if types == "None": continue
            for typex, tail in zip(types.split(","), tails.split(",")):
                gold_rels.append(typex)
                gold_tails.append(index2unnamed[int(tail)])
                gold_unnamed.append(index)

        # とりあえず方向は気にしない (ルールで与えられるやろ...)
        rel_label = torch.full((tmp_df.shape[0], tmp_df.shape[0]), rel2idx["None"])

        for rel, tail, index in zip(gold_rels, gold_tails, gold_unnamed):
            head_index = index
            tail_index = tail
            # 関係、スタート位置、終了位置
            #print(rel, head_index, tail_index)
            # 全indexを得る
            for i, idx in enumerate(ids):
                if idx[0] == head_index:
                    all_head_index, head_index = idx, i
                elif idx[0] == tail_index:
                    all_tail_index, tail_index = idx, i
            # index
            #print(all_head_index, all_tail_index)
            # 固有表現
            #print(seqs[head_index], seqs[tail_index])
            #print()
            # 置換
            for h_i in all_head_index:
                for t_i in all_tail_index:
                    """
                    if h_i > t_i:
                        rel_label[t_i, h_i] = rel2idx[rel]
                    elif t_i > h_i:
                        rel_label[h_i, t_i] = rel2idx[rel]
                    else:
                        pass
                    """
                    # h_i > t_iの場合: headとなる単語がtailとなる単語より右側にある = 矢印は右から左
                    # t_i > h_iの場合: headとなる単語がtailとなる単語より左側にある = 矢印は左から右
                    if h_i > t_i:
                        rel_label[t_i, h_i] = rel2idx["L-" + rel]
                    elif t_i > h_i:
                        rel_label[h_i, t_i] = rel2idx["R-" + rel]
                    else:
                        pass

        gold_labels.append(rel_label)
    return gold_labels


def simple_evaluate_re(re_val_gold_labels, re_preds, rel2idx):
    golds4eval, preds4eval = [], []
    for re_gold, re_logit in zip(re_val_gold_labels, re_preds):
        _, rel_preds = re_logit.max(dim=1)
        rel_preds_np = torch.triu(rel_preds, diagonal=1).detach().cpu().numpy()[0].tolist()
        rel_gold_np = torch.triu(re_gold, diagonal=1).detach().cpu().numpy().tolist()
        for gold, pred in zip(rel_gold_np, rel_preds_np):
            golds4eval.extend(gold)
            preds4eval.extend(pred)
    re_labels = [v for k, v in rel2idx.items() if k not in {"PAD": "PAD", "None": "None"}]
    return classification_report(golds4eval, preds4eval, output_dict=True, labels=re_labels)


def evaluate_ner(model, labels, preds, tag2idx, vecs):
    idx2tag = {v: k for k, v in tag2idx.items()}
    pred_tags = []
    index = 0
    # NERの予測結果のDECODE
    for predx in preds:
        predx = model.module.crf.decode(predx)
        for pred in predx:
            # 予測結果の[PAD]部分を削除する
            # vecsは[CLS]と[SEP]が入ってる
            pred_tags.append([idx2tag[pred] for pred in pred[:len(vecs[index])-2]])
            index += 1
    labels = [[idx2tag[l] for l in label] for label in labels]
    # 評価
    res = ner_eval(labels, pred_tags, output_dict=True)
    return res, pred_tags


def save_re_result(i_th_res, hyper, fold, rel2idx):
    with open('./results/{0}/{1}/RE/RESULT_{2}.json'.format(hyper.task, hyper.exp_name, fold), 'w') as f:
        json.dump(i_th_res, f, indent=4, cls=NpEncoder)
    with open('./results/{0}/{1}/RE/rel2idx_{2}.json'.format(hyper.task, hyper.exp_name, fold), 'w') as f:
        json.dump(rel2idx, f, indent=4, cls=NpEncoder)


def save_ner_result(i_th_res, hyper, fold, tag2idx):
    with open('./results/{0}/{1}/NER/RESULT_{2}.json'.format(hyper.task, hyper.exp_name, fold), 'w') as f:
        json.dump(i_th_res, f, indent=4, cls=NpEncoder)
    with open('./results/{0}/{1}/NER/tag2idx_{2}.json'.format(hyper.task, hyper.exp_name, fold), 'w') as f:
        json.dump(tag2idx, f, indent=4, cls=NpEncoder)


def save_csv(fold_res_df, hyper, fold, name):
    fold_res_df.to_csv('./results/{0}/{1}/{3}/{2}.csv'.format(hyper.task, hyper.exp_name, fold, name), index=False)


def evaluate_rel_v2(res_df, rel2idx):
    re_gold_results, re_pred_results = [], []
    for i, ids in enumerate(res_df["unique_no"].unique()):
        tmp_df = res_df[res_df["unique_no"] == ids]
        # 予測結果
        for types, tails in zip(tmp_df["pred_rel_type"], tmp_df["pred_rel_tail"]):
            pred = ["None"] * tmp_df.shape[0]
            if types == "None":
                pass
            else:
                for typex, tail in zip(types.split(","), tails.split(",")):
                    # サブワードにかかる場合は落とす
                    if tail == str(-999):
                        pass
                    else:
                        pred[int(tail)] = typex
            re_pred_results.extend(pred)
        # 正解
        for types, tails in zip(tmp_df["rel_type"], tmp_df["rel_tail"]):
            label = ["None"] * tmp_df.shape[0]
            if types == "None":
                pass
            else:
                for typex, tail in zip(types.split(","), tails.split(",")):
                    label[int(tail)] = typex
            re_gold_results.extend(label)


    rel_labels = list(set([key.replace("R-", "").replace("L-", "") for key in rel2idx.keys() if key not in ["None", "PAD"]]))
    res = classification_report(re_gold_results, re_pred_results, output_dict=True, labels=rel_labels)
    return res


def result2df_for_ner(X_test, ner_preds_decode):
    list_df = []
    for i, idx in enumerate(X_test["unique_no"].unique()):
        # DataFrame
        tmp_df = X_test[X_test["unique_no"]==idx]
        # NERを代入
        tmp_df["pred_IOB"] = ner_preds_decode[i]
        list_df.append(tmp_df)
    return pd.concat(list_df)


def result2df_for_re(X_test, re_preds, rel2idx, tag2idx):
    list_df = []
    idx2tag = {v: k for k, v in tag2idx.items()}
    for batch, idx in enumerate(X_test["unique_no"].unique()):
        # DataFrame
        tmp_df = X_test[X_test["unique_no"]==idx]
        # 予測結果の変換
        _, rel_logit = re_preds[batch].max(dim=1)
        rel_logit = torch.triu(rel_logit, diagonal=1).detach().cpu().numpy()[0].tolist()
        # 固有表現、タグ、serialを取得する
        # tokens, labels, indexs = list(tmp_df["word"]), list(tmp_df["pred_IOB"]), list(tmp_df["serial"])
        tokens, labels, indexs = list(tmp_df["word"]), list(tmp_df["IOB"]), list(tmp_df["serial"])
        seqs, tags, ids = [], [], []
        for i in range(0, len(tokens), 1):
            if labels[i].startswith("B-"):
                if i == len(labels) - 1:
                    seqs.append(tokens[i])
                    tags.append(labels[i])
                    ids.append([int(indexs[i])])
                else:
                    tmp1, tmp2, tmp3 = [tokens[i]], [labels[i]], [int(indexs[i])]
                    for j in range(i+1, len(tokens), 1):
                        if labels[j].startswith("I-"):
                            tmp1.append(tokens[j])
                            tmp2.append(labels[j])
                            tmp3.append(int(indexs[j]))
                            if j ==  len(labels) - 1:
                                seqs.append(" ".join(tmp1))
                                tags.append(" ".join(tmp2))
                                ids.append(tmp3)
                        else:
                            seqs.append(" ".join(tmp1))
                            tags.append(" ".join(tmp2))
                            ids.append(tmp3)
                            break  
        # Bタグのindex
        begin_index = [idx[0] for idx in ids]
        # Iタグのindex
        inside_index = [idxx for idx in ids for idxx in idx[1: ]]
        # 予測結果を集計
        idx2rel = {v: k for k, v in rel2idx.items()}
        decode_rels = []
        for i in range(0, tmp_df.shape[0], 1):
            # 各行の予測結果
            i_th_rel_logit = rel_logit[i]#.detach().cpu().numpy()
            # Iタグは飛ばして良い (本当は良くないが、変換が難しいので -> 誤ってIタグから関係が出ている場合、これを見逃す)
            if i in inside_index: continue
            # 各行の各要素を確認
            for j in range(0, tmp_df.shape[0], 1):
                # 同様
                if j in inside_index: continue
                # 関係がある場合の処理 (右下は0、つまりPADになることに注意)
                if i_th_rel_logit[j] != rel2idx["None"] and i_th_rel_logit[j] != rel2idx["PAD"]:
                    rel_pred = idx2rel[i_th_rel_logit[j]]
                    if rel_pred[0] == "R":
                        # この値 (headとtail)はserialであり、通し番号である (正解ラベルはindex列)
                        decode_rels.append({"head": i, "tail": j, "rel": rel_pred[2:]})
                    else:
                        decode_rels.append({"head": j, "tail": i, "rel": rel_pred[2:]})
        # データフレームへ代入
        types, tails, indexs = [], [], []
        # 変換辞書
        unnamed2index = {int(x): int(y) for x, y in zip(tmp_df["serial"], tmp_df["index"])}
        for i in range(0, tmp_df.shape[0], 1):
            tmp1, tmp2, tmp3 = [], [], []
            for res in decode_rels:
                if res["head"] == i:
                    tmp1.append(str(unnamed2index[res["tail"]]))
                    tmp2.append(str(res["rel"]))
                    tmp3.append(str(res["tail"]))
            if len(tmp1) != 0:
                tails.append(",".join(tmp1))
                types.append(",".join(tmp2))
                indexs.append(",".join(tmp3))
            else:
                tails.append("None")
                types.append("None")
                indexs.append("None")
        tmp_df["pred_rel_tail"] = tails
        tmp_df["pred_rel_type"] = types
        tmp_df["pred_rel_unnamed"] = indexs
        list_df.append(tmp_df)
    return pd.concat(list_df)