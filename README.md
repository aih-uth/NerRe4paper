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
python main_ner.py --bert_path UTH-BERT --batch_size 16
```

To train and evaluate a RE model, run
```
python main_re.py --bert_path UTH-BERT
```

## Run in Google Colaboratory

If you have a Google account, you are able to run our code in Google Colab.
Please run the following code in Google Colab.
Note that if you want to run an experiment with the same experimental setup as ours, you maight have to subscribe Colab Pro.

```python
import os
! git clone https://github.com/aih-uth/UTH-29
! wget https://ai-health.m.u-tokyo.ac.jp/labweb/dl/uth_bert/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K_pytorch.zip
! unzip UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K_pytorch.zip
! pip install transformers seqeval
os.chdir("./UTH-29")
! python main_ner.py --bert_path ../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K --batch_size 16
! python main_re.py --bert_path ../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K
```

## Prefromance
| Fold | NER |RE|
|:---|---:|---:|
|1 |0.917|0.859|
|2 |0.913|0.849|
|3 |0.926|0.855|
|4 |0.925|0.847|
|5 |0.917|0.849|
|Avg. |0.920|0.852|

# License
CC BY-NC-SA 4.0

## References

- Ma, Y., Hiraoka, T., & Okazaki, N. (2020). Named Entity Recognition and Relation Extraction using Enhanced Table Filling by Contextualized Representations. arXiv preprint arXiv:2010.07522. Software available from https://github.com/YoumiMa/TablERT.
- pytorch-crf. Software available from https://pytorch-crf.readthedocs.io/en/stable/.
- Hiroki Nakayama. seqeval: A python framework for sequence labeling evaluation, 2018. Software available from https://github.com/chakki-works/seqeval.

## Citation

If you use our code in your work, please cite the following paper:
```
Add Paper Info
```
