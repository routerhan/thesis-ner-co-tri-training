# thesis-ner-co-tri-training
This repo is aiming to implementing the co-training and tri-training framework to compare the result of fine-grained NER task

# Overview
The task is focusing on `semi-supervised learning`, therefore there are a small set of labeled data and larger set of unlabeled data in our experiemnt.
## Dataset
* Labeled corpus:
    - `Israel Corpus`, German spoken corpus.
    - `OntoNotes 5.0`, English labeled set.
    - `Tweet`, German.
* Unlabled corpus:
    - `Movie subtiles`, German.

## Models
* Self-training with BERT fine-tuned model as base classifier.
* Co-training with BERT fine-tuned model as base classifier with cross-lingual approach.
* Tri-training with teacher-student paradigm, BERT fine-tuned model as base classifier with cross-lingual approach.
* Tri-training with teacher-student paradigm, BERT fine-tuned model as base classifier but using different BERT pretrained embedding model, with cross-lingual approach.

# 1. ISW BERT baseline model - German
* __BERT model : `bert-base-german-cased`__

## Train the baseline BERT model
`python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir baseline_model/ --max_seq_length 128 --do_train`

## Evaluate the model on dev or test set
`python run_ner.py --data_dir data/full-isw-release.tsv --output_dir baseline_model/ --max_seq_length 128 --do_eval --eval_on test`

## ISW BERT Performance:
`Train/dev/test : 70/20/10`

```
- INFO - preprocessor -   ------ Preprocssing ISW German corpus ------
- INFO - preprocessor -   Number of sentences: 16084 
- INFO - preprocessor -   Number of tags: 62 
- INFO - train_test_split -   ***** Train/dev/test split: 70/20/10 *****
- INFO - train_test_split -     Num train = 11258
- INFO - train_test_split -     Num dev = 3217
- INFO - train_test_split -     Num test = 1609

- INFO - __main__ -   ***** Running training *****
- INFO - __main__ -     Num examples = 11258
- INFO - __main__ -     Batch size = 32
- INFO - __main__ -     Num steps = 1053

- INFO - preprocessor -   *** Features Example ***
- INFO - preprocessor -   textlist: Ich kannte sie nur in Wien
- INFO - preprocessor -   labellist: O O O O O B-GPE
- INFO - preprocessor -   tokens: Ich kannte sie nur in Wien
```


```
- INFO - __main__ -   ***** Running evaluation: test *****

           precision    recall  f1-score   support

      LAN     0.9376    0.9592    0.9483       564
     TIME     0.8598    0.9093    0.8839      1059
     DATE     0.7752    0.7642    0.7697       352
      DUR     0.6850    0.6493    0.6667       422
      GPE     0.9653    0.9739    0.9696      1113
      AGE     0.6614    0.5915    0.6245       142
      PER     0.8299    0.8971    0.8622       272
      NRP     0.9226    0.9140    0.9183       756
  ORDINAL     0.8197    0.8264    0.8230       121
      EVT     0.9020    0.7797    0.8364        59
 CARDINAL     0.8542    0.8039    0.8283       153
     PROJ     1.0000    0.6667    0.8000         3
      FAC     0.7143    0.8125    0.7602        80
      LOC     0.7400    0.6852    0.7115        54
     FREQ     0.7716    0.8224    0.7962       152
     FRAC     0.4000    0.5000    0.4444         4
     MISC     0.7714    0.6136    0.6835        44
      ORG     0.6900    0.6832    0.6866       101
      ART     0.4091    0.3103    0.3529        29
     SORD     0.7763    0.7284    0.7516        81
    TITLE     0.9032    0.9032    0.9032        31
      MON     0.8421    0.8421    0.8421        19
     RATE     0.0000    0.0000    0.0000         4
  PRODUCT     0.0000    0.0000    0.0000         9
    QUANT     0.7778    0.7000    0.7368        10
     PERC     1.0000    0.8182    0.9000        11
      ADD     0.0000    0.0000    0.0000         1
      MED     0.0000    0.0000    0.0000         1
      LAW     1.0000    1.0000    1.0000         1

micro avg     0.8587    0.8631    0.8609      5648
macro avg     0.8564    0.8631    0.8592      5648

- INFO - __main__ -   ***** Save the results to baseline_model/: test_results.txt *****
```

## Basline isw-model configuration
```
{
    "bert_model": "bert-base-german-cased", 
    "do_lower": false, 
    "train_data_dir": "data/full-isw-release.tsv", 
    "train_batch_size": 32, 
    "num_train_epochs": 3.0, 
    "learning_rate": 5e-05, 
    "adam_epsilon": 1e-08, 
    "max_grad_norm": 1.0, 
    "max_seq_length": 128, 
    "output_dir": "baseline_model/", 
    "seed": 42, 
    "gradient_accumulation_steps": 1, 
    "num_labels": 63, 
    "label_map": 
        {"1": "B-ADD", "2": "B-AGE", "3": "B-ART", "4": "B-CARDINAL", "5": "B-CREAT", 
        "6": "B-DATE", "7": "B-DUR", "8": "B-EVT", "9": "B-FAC", "10": "B-FRAC", 
        "11": "B-FREQ", "12": "B-GPE", "13": "B-LAN", "14": "B-LAW", "15": "B-LOC", 
        "16": "B-MED", "17": "B-MISC", "18": "B-MON", "19": "B-NRP", "20": "B-ORDINAL", 
        "21": "B-ORG", "22": "B-PER", "23": "B-PERC", "24": "B-PRODUCT", "25": "B-PROJ", 
        "26": "B-QUANT", "27": "B-RATE", "28": "B-SORD", "29": "B-TIME", "30": "B-TITLE", 
        "31": "I-ADD", "32": "I-AGE", "33": "I-ART", "34": "I-CARDINAL", "35": "I-DATE", 
        "36": "I-DUR", "37": "I-EVT", "38": "I-FAC", "39": "I-FRAC", "40": "I-FREQ", 
        "41": "I-GPE", "42": "I-LAN", "43": "I-LAW", "44": "I-LOC", "45": "I-MED", 
        "46": "I-MISC", "47": "I-MON", "48": "I-NRP", "49": "I-ORDINAL", "50": "I-ORG", 
        "51": "I-PER", "52": "I-PERC", "53": "I-PRODUCT", "54": "I-PROJ", "55": "I-QUANT", 
        "56": "I-RATE", "57": "I-SORD", "58": "I-TIME", "59": "I-TITLE", 
        "60": "O", "61": "[CLS]", "62": "[SEP]"}
}
```

# 2. OntoNotes 5.0 BERT Model - English
* __BERT model : `bert-base-uncased`__

## Train the OntoNote 5.0 Eng model
* __*Noted that `--do_lower_case` should be triggered as we are using uncased model.__ \

`python run_ner.py --data_dir ../OntoNotes-5.0-NER-BIO/onto.train.ner --bert_model bert-base-uncased --output_dir onto_model/ --max_seq_length 128 --do_train --do_lower_case`

## Evaluate the model on dev or test set
`python run_ner.py --data_dir ../OntoNotes-5.0-NER-BIO/onto.test.ner --output_dir onto_model/ --max_seq_length 128 --do_eval --eval_on test --do_lower_case`

## OntoNotes 5.0 BERT Performance

* `dev : /onto.development.ner` \
```
***** Running evaluation: dev *****
  Num examples = 15680
  Batch size = 8
- INFO - __main__ -
             precision    recall  f1-score   support

        ORG     0.8406    0.8606    0.8505      3794
     PERSON     0.9186    0.9480    0.9331      3154
       NORP     0.8747    0.8802    0.8774      1277
       DATE     0.8479    0.8812    0.8642      3200
   CARDINAL     0.8017    0.8749    0.8367      1719
    ORDINAL     0.7609    0.8358    0.7966       335
        GPE     0.9202    0.9113    0.9157      3630
      MONEY     0.8916    0.9064    0.8989       844
        LAW     0.3804    0.5385    0.4459        65
        FAC     0.3756    0.6015    0.4624       133
   LANGUAGE     0.6122    0.8571    0.7143        35
    PRODUCT     0.5779    0.4159    0.4837       214
    PERCENT     0.9064    0.9009    0.9037       656
       TIME     0.7103    0.7812    0.7441       361
WORK_OF_ART     0.4375    0.4505    0.4439       202
        LOC     0.6250    0.6646    0.6442       316
   QUANTITY     0.7778    0.7000    0.7368       190
      EVENT     0.5205    0.4246    0.4677       179

avg / total     0.8491    0.8704    0.8590     20304
```
* `test : /onto.test.ner`
```
***** Running evaluation: test *****
  Num examples = 12217
  Batch size = 8
- INFO - __main__ -
             precision    recall  f1-score   support

        FAC     0.5600    0.6577    0.6049       149
       TIME     0.5479    0.6356    0.5885       225
        GPE     0.9133    0.9057    0.9095      2546
        ORG     0.8084    0.8432    0.8254      2002
       NORP     0.8777    0.9131    0.8950       990
   QUANTITY     0.7394    0.6863    0.7119       153
     PERSON     0.9074    0.9049    0.9061      2134
   CARDINAL     0.7813    0.8000    0.7906      1005
       DATE     0.7997    0.8673    0.8321      1786
        LOC     0.6034    0.6512    0.6264       215
      MONEY     0.8522    0.9099    0.8801       355
    ORDINAL     0.6846    0.7971    0.7366       207
    PERCENT     0.8571    0.8971    0.8766       408
WORK_OF_ART     0.5087    0.5207    0.5146       169
   LANGUAGE     0.7059    0.5455    0.6154        22
    PRODUCT     0.6111    0.6111    0.6111        90
      EVENT     0.5667    0.6000    0.5829        85
        LAW     0.4333    0.5909    0.5000        44

avg / total     0.8287    0.8545    0.8410     12585
```

## OntoNote 5.0 model configuration
```
{
    "bert_model": "bert-base-uncased", 
    "do_lower": true, 
    "train_data_dir": "../OntoNotes-5.0-NER-BIO/onto.train.ner", 
    "train_batch_size": 32, 
    "num_train_epochs": 3.0, 
    "learning_rate": 5e-05, 
    "adam_epsilon": 1e-08, 
    "max_grad_norm": 1.0, 
    "max_seq_length": 128, 
    "output_dir": "onto_model/", 
    "seed": 42, 
    "gradient_accumulation_steps": 1, 
    "num_labels": 40, 
    "label_map": 
    {"1": "B-CARDINAL", "2": "B-DATE", "3": "B-EVENT", "4": "B-FAC", "5": "B-GPE", 
    "6": "B-LANGUAGE", "7": "B-LAW", "8": "B-LOC", "9": "B-MONEY", "10": "B-NORP", 
    "11": "B-ORDINAL", "12": "B-ORG", "13": "B-PERCENT", "14": "B-PERSON", "15": "B-PRODUCT", 
    "16": "B-QUANTITY", "17": "B-TIME", "18": "B-WORK_OF_ART", "19": "I-CARDINAL", "20": "I-DATE", 
    "21": "I-EVENT", "22": "I-FAC", "23": "I-GPE", "24": "I-LANGUAGE", "25": "I-LAW", 
    "26": "I-LOC", "27": "I-MONEY", "28": "I-NORP", "29": "I-ORDINAL", "30": "I-ORG", 
    "31": "I-PERCENT", "32": "I-PERSON", "33": "I-PRODUCT", "34": "I-QUANTITY", "35": "I-TIME", 
    "36": "I-WORK_OF_ART", "37": "O", "38": "[CLS]", "39": "[SEP]"}
}
```

# Co-Training method
Co-training algorithm is considered as bootstrap method to boost the amount of labeled set from unlabeled set. 

To apply this method to our project, you should do the following:

## Prerequisite
1. Make sure you have two model trained. i.e. ISW and Onto models.
```
python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir baseline_model/ --max_seq_length 128 --do_train

python run_ner.py --data_dir ../OntoNotes-5.0-NER-BIO/onto.train.ner --bert_model bert-base-uncased --output_dir onto_model/ --max_seq_length 128 --do_train --do_lower_case
```

2. Get cross-lingual training features. i.e. Unlabled set with in German and English version, where we get these by introducing machine translation tools.

* First you need to follow the steps of machine_translation/README.md.
* Once you have the `de_sents.txt` and `en_sents.txt` as our unlabeled set, we can start out co-training process.


## Steps
1. Execute the co-training script to get `extended labeled set`, which will be later used to extend the original labeled set.
* You may need to decide the value of co-training params.

| Environment Variable| Default| Description|
|---------------------|--------|------------|
| `ext_output_dir`  | ext_data/ |The dir that you save the extended L set. |
| `modelA_dir` | baseline_model/ |The dir of pre-trained model that will be used in the cotraining algorithm on the X1 feature set, e.g. German.|
| `modelB_dir` | onto_model/ |The dir of another pre-trained model can be specified to be used on the X2 feature set, e.g. English.|
|`de_unlabel_dir`| machine_translation/2017_de_sents.txt |The dir of unlabeled sentences in German.|
|`en_unlabel_dir`| machine_translation/2017_en_sents.txt |The dir of unlabeled sentences in English.|
|`top_n`|5|The number of the most confident examples that will be 'labeled' by each classifier during each iteration.|
|`k`|30|The number of iterations.|
|`u`|75|The size of the pool of unlabeled samples from which the classifier can choose.|
```
python run_cotrain.py --ext_output_dir ext_data/ext_data_1000_top_20 --modelA_dir baseline_model --modelB_dir onto_model --de_unlabel_dir machine_translation/2017_de_sents.txt --en_unlabel_dir machine_translation/2017_en_sents.txt --k 10 --u 10 --top_n 3 --save_preds --save_agree
```
The output of this script would be a `ext_data/ext_data_1000_top_20` directory.
```
ext_data/ext_data_1000_top_20
  ├── 1521_ext_L_A_labels.pkl
  ├── 1521_ext_L_A_sents.pkl
  ├── 1521_ext_L_B_labels.pkl
  ├── 1521_ext_L_B_sents.pkl
  ├── agree_results.txt
  └── cotrain_config.json
```
cotrain_config
```
{
  "ext_output_dir": "ext_data/ext_data_1000_top_20", 
  "Approach": "Cross-lingual Co-training", 
  "Model A de": "baseline_model", 
  "Model B en": "onto_model", 
  "Pool value u": 100, 
  "Confident top_n": 10, 
  "Iteration k": 1000, 
  "Agree threshold cos_score": 0.7, 
  "Ext number of L_": 1521, 
  "Prefix": 1521
}
```
Examples: agree_results that pass the `identity check` and `confident threshold`, for sequence labeling task. i.e. NER
```
sent_id : 248969

['Ja', 'Ich', 'war', 'hier', '1989', 'tätig', 'als', 'Catlin', 'das', 'absolute', 'Sagen', 'hatte']	
['O', 'O', 'O', 'O', 'B-DATE', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O']	
avg_cfd_score: 0.9936

['i', 'mean', 'yes', 'in', '1989', 'i', 'was', 'here', 'to', 'say', 'absolute']	
['O', 'O', 'O', 'O', 'B-DATE', 'O', 'O', 'O', 'O', 'O', 'O']	
avg_cfd_score: 0.918

cos_score : 0.7071
```

2. Execute the `run_ner.py` script to train the ext model again, with `extent_L` args enabled, which will take you to retrain the model with new extended labeled set.
```
python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir baseline_model/ --max_seq_length 128 --do_train --extend_L --ext_data_dir ext_data/ext_data_1000_top_20 --ext_output_dir co-models/ext_1521_isw_model/
```

With enabling flag `--extend_L`, you will see the train data is extended
```
- INFO - __main__ -   ***** Loading ISW data *****
- INFO - __main__ -   Origin de L size: 11258
- INFO - __main__ -   Ext de L_ size: + 1521 = 12779
...
```

3. Evaluate the new model as we did before but enable `extend_L`
```
python run_ner.py --data_dir data/full-isw-release.tsv --output_dir baseline_model/ --max_seq_length 128 --do_eval --eval_on test --extend_L --ext_output_dir co-models/ext_1521_isw_model/
```
* the ext_model will be saved in the directory `--ext_output_dir`


# Tri-Training method
Tri-training is also a semi-supervised training method, it subsamples the labeled set and learn three initial classifiers.
The teacher-student algorithm will be implemented here in more details.
## Prerequisite
1. Make sure you have save the subset of `isw-train-data` in to `s1, s2 and s3` respectively with desired proportion `r1, r2 and r3`
```
# r1, r2, r3, dataset are paras you should look after
python run_tritrain.py --save_subsample --sample_dir sub_data/ --r1 0.4 --r2 0.4 --r3 0.2 --dataset isw
```
2. Learn initialized three classifiers with subsets respectively.
```
# --output_dir : the subset model dir, --do_subtrain: enable train subset, --subtrain_dir: the dir where subset data is stored
python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir tri-models/s1_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-s1.pkl

python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir tri-models/s2_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-s2.pkl

python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir tri-models/s3_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-s3.pkl
```
* Once you have 3 initial classifiers candidates for teacher-student, you may start tri-training !


## Steps

1. Execute the `run_tritrain.py`, which will assign `teacher-student roles` and also give you the `teachabel samples` as new adding labels for re-train the student model.

* You may need to decide the value of teacher-student tri-training params.

| Environment Variable| Default| Description|
|---------------------|--------|------------|
| `ext_data_dir`  | tri_ext_data/ |The dir saved the teachable samples as sents and labels pkl file. |
| `val_on` | test |The test samples for you to validate the error rate of teacher candidates.|
| `U` | data/dev-isw-sentences.pkl |The file of unlabeled set with the format of list of sentences.|
|`u`| 100 | The pool value of U, to limit the amount of unlabeled samples to use.|
|`cos_score_threshold`| 0.7 |The similarity socre threshold to check identicality of sequance labeling task.|
|`mi_dir`| tri-models/s1_model/ |The model dir trained on subset s1.|
|`mj_dir`| tri-models/s2_model/ |The model dir trained on subset s2.|
|`mk_dir`| tri-models/s3_model/ |The model dir trained on subset s3.|
|`tcfd_threshold`|0.7|The teacher confidence threshold for checking whether the sample x is teachable. i.e. teacher's cfd must both over this threshold.|
|`scfd_threshold`|0.6|The student confidence threshold for checking whether the sample x is teachable. i.ie student's cfd must lower that the threshold.|
|`r_t`|0.1|The addaptive rate of threshold for teacher clf after each iteration.|
|`r_s`|0.1|The addaptive rate of threshold for student clf after each iteration.|
```
python run_tritrain.py --ext_data_dir tri_ext_data/u_3000 --val_on test --U data/dev-isw-sentences.pkl --u 3000 --mi_dir tri-models/s1_model/ --mj_dir tri-models/s2_model/ --mk_dir tri-models/s3_model/ --tcfd_threshold 0.7 --scfd_threshold 0.6 --r_t 0.1 --r_s 0.1
```

* Assign teacher-student relationship
```
***** Assigning teacher and student clf ***** 
e_ij : 0.0878
e_ik : 0.1051
e_jk : 0.0905
teacher models are assign to : mi, mj
```

* We use the labeled set(i.e. test|dev set) for computing the `error_rate`, which is used to determiner teacher and student roles.

* Example of teachable samples
```
***** Picking teachable samples ***** 
num of teachable = 108
***** Example samples ***** 
Sent =      Ah das ist der Herr Düringer heißt er
t1 preds = (['O', 'O', 'O', 'O', 'B-TITLE', 'B-PER', 'O', 'O']	0.7388)
t2 preds = (['O', 'O', 'O', 'O', 'B-TITLE', 'B-PER', 'O', 'O']	0.7675)
s preds = (['O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O']	0.4584)
```

2. Use `teacherable samples` to extend the training data and re-train the model (i.e. student model)

Execute the `run_ner.py` script to train the ext model again, with `extent_L_tri` args enabled, which will take you to retrain the model with new adding `teachable samples` set.
```
python run_ner.py --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-s3.pkl --extend_L_tri --ext_data_dir tri_ext_data/u_3000 --ext_output_dir tri-models/ext_108_tri_model/
```


# Simple API
JUST for giving you an idea on the model prediction and further investigation. 

`python api.py`

## POST
`curl -X POST http://0.0.0.0:8080/predict -H 'Content-Type: application/json' -d '{ "sentence" : " ich aus EU" }'`
```
{
	"sentence" : " ich aus EU"
}
```

```
{
    "result": [
        {
            "confidence": 0.2621215581893921,
            "tag": "O",
            "word": "ich"
        },
        {
            "confidence": 0.0977315902709961,
            "tag": "I-SORD",
            "word": "aus"
        },
        {
            "confidence": 0.1431599259376526,
            "tag": "O",
            "word": "EU"
        }
    ]
}
```