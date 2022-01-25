# Development of comprehensive annotation criteria for patients’ states of clinical texts

This is the experimental code for the paper Development of comprehensive annotation criteria for patients’ states of clinical texts.

## Setpu
### Requirements

- Python 3.8+
- pandas 1.2.4
- numpy 1.20.1
- torch 1.10.1+cu113
- scikit-learn 0.24.1
- transformers 4.11.3
- seqeval 1.2.2

## Run

Download UTH-BERT [here](https://ai-health.m.u-tokyo.ac.jp/home/research/uth-bert).

To train and evaluate a NER model, run
```
python main_ner.py --bert_path <UTH-BERT path>
```

To train and evaluate a RE model, run
```
python main_re.py --bert_path <UTH-BERT path>
```

## Run in Google Colaboratory

If you have a Google account, you are able to run our code in Google Colab.
Please run the following code in Google Colab.

(Note that if you want to run an experiment with the same experimental setup as ours, you maight have to subscribe Colab Pro.)

```python
import os
! git clone XXX
! wget https://ai-health.m.u-tokyo.ac.jp/labweb/dl/uth_bert/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K_pytorch.zip
! unzip UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K_pytorch.zip
! pip install transformers seqeval
os.chdir("./UTH-29")
! python main_ner.py --bert_path ../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K
! python main_re.py --bert_path ../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K
```


## References
```
[1] 篠原 恵美子， 河添 悦昌， 柴田 大作， 嶋本 公徳， 関 倫久. (2021). 医療テキストに対する網羅的な所見アノテーションのためのアノテーション基準の構築. 第25回日本医療情報学春季学術大会. (Japanese)
[2] Kawazoe, Y., Shibata, D., Shinohara, E., Aramaki, E., & Ohe, K. (2021). A clinical specific BERT developed using a huge Japanese clinical text corpus. Plos one, 16(11), e0259763.
[3] Ma, Y., Hiraoka, T., & Okazaki, N. (2020). Named Entity Recognition and Relation Extraction using Enhanced Table Filling by Contextualized Representations. arXiv preprint arXiv:2010.07522.
[4] Ptroch CRF: https://pytorch-crf.readthedocs.io/en/stable/
[5] Seqeval: https://github.com/chakki-works/seqeval
```

## Citation

If you use our code in your word, please cite the following paper:
```
Add Paper Info
```
