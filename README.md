# thesis-ner-co-tri-training
This repo aims to implement novel cross-lingual co-training approach and tri-training with teacher-student learning paradigm approach on fine-grained named entity recognition task in German language.

# Outline
### 1. Dataset & Baseline models
* Labeled corpus:
    - `Israel Corpus`, German spoken corpus.
    - `OntoNotes 5.0`, English labeled set.
* Unlabled corpus:
    - `OpenSubtitles`,Movie subtiles in German.

### 2. How to run the experiemts?
* Obtain baseline models:
    - `ISW BERT baseline model`
    - `OntoNotes 5.0 BERT Model`
* Cross-lingual co-training approach:
    - Prerequisite : machine translation setup
    - Parameters setup
    - Training
    - Evaluation
    - Further analysis
      - Influence of Confident Selection (top n)
      - Influence of Unlabeled Samples Size (pool value u)
* Tri-training with teacher-student learning approach:
    - Prerequisite : single-view spliting
    - Parameters setup
    - Training
    - Evaluation
    - Further analysis
      - Influence of Pool Value (pool value u)
* Comparison Evaluation
    - Quantitive Analysis
      - Fixed amount of unlabeled samples
      - Fixed amount of pseudo-labeled samples

* Single Tag Evaluation
    - Co-training with single tag
    - Tri-training with single tag

# Obtain Baseline Models
## ISW BERT baseline model - German
### Train the model
`python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir baseline_model/ --max_seq_length 128 --do_train`

### Evaluate the model on dev or test set
`python run_ner.py --data_dir data/full-isw-release.tsv --output_dir baseline_model/ --max_seq_length 128 --do_eval --eval_on test`

## OntoNotes 5.0 BERT Model
### Train the model
`python run_ner.py --data_dir ../OntoNotes-5.0-NER-BIO/onto.train.ner --bert_model bert-base-uncased --output_dir onto_model/ --max_seq_length 128 --do_train --do_lower_case`

### Evaluate the model on dev or test set
`python run_ner.py --data_dir ../OntoNotes-5.0-NER-BIO/onto.test.ner --output_dir onto_model/ --max_seq_length 128 --do_eval --eval_on test --do_lower_case`

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


## Train Steps
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

## Further analysis
Here we also experiemnted how different initialization of our proposed approach affects the model performance, simply trigger script is described as follows:
1. Influence of Confident Selection (top n):

`pyhton hack_co.py --{tune_top_n} --do_retrain`

2. Influence of Unlabeled Samples Size (pool value u)

`pyhton hack_co.py --{tune_pool_value} --do_retrain`

# Tri-Training method
Tri-training is also a semi-supervised training method, it subsamples the labeled set and learn three initial classifiers.
The teacher-student algorithm will be implemented here in more details.
## Prerequisite
1. Make sure you have save the subset of `isw-train-data` in to `s1, s2 and s3` with replacement and desired proportion `r`
```
# r and dataset are paras you should look after
python run_tritrain.py --save_subsample --sample_dir sub_data/ --r 0.7 --dataset isw
```
2. Learn initialized three classifiers with subsets respectively.
```
# --output_dir : the subset model dir, --do_subtrain: enable train subset, --subtrain_dir: the dir where subset data is stored

python run_ner.py --output_dir tri-models/s1_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-s1.pkl
```

One-line trigger for whole Prerequisite setup by enabling arg: `--do_prerequisite`, which will give you three initial classifiers, i.e. s1, s2 and s3
```
python hack_tri.py --do_prerequisite
```

The initial classifiers are saved under `tri-models`
```
$ tree tri-models/
tri-models/
|-- eval_monitor
|-- s1_model
|   |-- added_tokens.json
|   |-- config.json
|   |-- model_config.json
|   |-- pytorch_model.bin
|   |-- special_tokens_map.json
|   |-- test_results.txt
|   |-- tokenizer_config.json
|   `-- vocab.txt
|-- s2_model
|   |-- added_tokens.json
|   |-- config.json
|   |-- model_config.json
|   |-- pytorch_model.bin
|   |-- special_tokens_map.json
|   |-- test_results.txt
|   |-- tokenizer_config.json
|   `-- vocab.txt
`-- s3_model
    |-- added_tokens.json
    |-- config.json
    |-- model_config.json
    |-- pytorch_model.bin
    |-- special_tokens_map.json
    |-- test_results.txt
    |-- tokenizer_config.json
    `-- vocab.txt
```

Once you have 3 initial classifiers candidates for teacher-student, you may start tri-training !


## Steps

1. Execute the `run_tritrain.py`, which will rotately assign `teacher-student roles` and also give you the `teachabel samples` as new adding labels to re-train each student model.

You may need to decide the value of teacher-student tri-training params.

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
|`scfd_threshold`|0.6|The student confidence threshold for checking whether the sample x is teachable. i.e student's cfd must lower that the threshold.|
|`r_t`|0.1|The addaptive rate of threshold for teacher clf after each iteration.|
|`r_s`|0.1|The addaptive rate of threshold for student clf after each iteration.|


Start Tri-training pipeline by executing the following command:
```
python hack_tri.py --u 3000 --tcfd_threshold 0.9 --scfd_threshold 0.5
```

The command above will automatically runs the following steps:
```
For each iteration, while teacher's cfd threshold >= student's cfd threshold
  Given a set of instances:
    for each roles rotatation, i.e. either c1, c2 or c3 is student:
      Return a set of `teachable` instances for student to learn

  Re-train c1, c2, c3 with extended subsets respectively (s1, s2, s3 plus its `teachable instances`)

  At the end of each iteration, the thresholds have to be adjusted, as we assume that the knowledge gap between teachers and student is becoming smaller. 
```
1. It will get each `teachable samples` of three init models and saved as pkl file under `student model dir` respectively, e.g. `tri-models/s2_model/`
```
***** Picking teachable samples ***** 
num of teachable = 108
***** Example samples ***** 
Sent =      Ah das ist der Herr Düringer heißt er
t1 preds = (['O', 'O', 'O', 'O', 'B-TITLE', 'B-PER', 'O', 'O']	0.7388)
t2 preds = (['O', 'O', 'O', 'O', 'B-TITLE', 'B-PER', 'O', 'O']	0.7675)
s preds = (['O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O']	0.4584)
```

2. Re-train c1, c2, c3 with extended teachable samples respectively (s1, s2, s3 plus its `teachable instances`)
```
Origin Student sub_data/ext-train-isw-s3.pkl L size: 7882
---Tri-training---: Ext teachable L_ size: + 3 = 7885
***** Save ext-train-isw.pkl for next iteration : s3 *****

***** Running training *****
Num examples = 7885
Batch size = 32
Num steps = 738

Epoch: 100%|########################################| 3/3 
***** Success to save model in dir : tri-ext-models/2_ext_s3_model/ *****
```

3. The thresholds will be adjusted, as we assume that the knowledge gap between teachers and student is becoming smaller. 
```
Adapted teacher threshold: 0.7, student threshold: 0.6
***** End of iteration : 1 *****
```

4. Monitoring the eval result of each classifies at each iteration.
```
***** Save the results to tri-models/eval_monitor/: 1_s2_test_results.txt *****
```


## Important Key Script
| Environment Variable| Default| Description|
|---------------------|--------|------------|
| `do_subtrain`  | store_true | Enable loading the subset of train data, i.e. s1, s2 or s3|
| `subtrain_dir` | sub_data/train-isw-s3.pkl  | we need to specify the train data of `student` (either s1, s2 or s3), decided by the `error_rate`|
| `extend_L_tri` | store_true |Enable executing the training phrase with new adding teachable samples for `retrain` the student clf.|
| `ext_data_dir` | sub_data/train-isw-{s3}.pkl | The data dir that saved teachable samples.|
| `ext_output_dir` | tri-ext-models/{it}_ext_{s1}_model/ |The dir which saved the retrained student clf from ext_teachable_data.|
```
// the ext_model will be saved in the directory `--ext_output_dir`

python run_ner.py --max_seq_length 128 --do_train --do_subtrain --extend_L_tri --it {1} --subtrain_dir sub_data/train-isw-{s3}.pkl --ext_data_dir tri-models/{s3}_model/ --ext_output_dir tri-ext-models/{it}_ext_{s1}_model/
```

4. Evaluate the retrained student model as we did before but enable `extend_L_tri`
```
python run_ner.py --do_eval --eval_on test --extend_L_tri --eval_dir tri-models/eval_monitor/ --ext_output_dir tri-ext-models/{it}_ext_{s1}_model/ --it_prefix {1}_{s3}
```

# Comparison Evaluation
Here we conducted the experiments with same setting to see the difference between two proposed methods with 5 trials.

| Environment Variable| Default| Description|
|---------------------|--------|------------|
| `get_random_baselines`  | store_true | Whether to train trials baseline model trained on random selected train set|
| `get_random_co_train_result_fix_u` | store_true  | Whether to train trials co-models with fix amount of unlabeled samples|
| `get_random_tri_train_result_fix_u` | store_true  | Whether to train trials tri-models with fix amount of unlabeled samples|
| `get_random_co_train_result_fix_n` | store_true  | Whether to train trials co-models with fix amount of selected samples|
| `get_random_tri_train_result_fix_n` | store_true  | Whether to train trials tri-models with fix amount of selected samples|
| `n_trials` | int  | the number of trials for experiemt|
| `selected_n` | int  | the number of n for experiemt|

## Fixed amount of unlabeled samples
* Co-training:

`python hack_exp.py --get_random_co_train_result_fix_u --n_trials 5`
* Tri-training:

`python hack_exp.py --get_random_tri_train_result_fix_u --n_trials 5`
## Fixed amount of pseudo-labeled samples
* Co-training:

`python hack_exp.py --get_random_co_train_result_fix_n --n_trials 5 --n 100`
* Tri-training:

`python hack_exp.py --get_random_tri_train_result_fix_n --n_trials 5 --n 100`


# Single Tag Evaluation
Here we limited the newly added samples with selected tag, which means only the samples with certain tags will be added into original train data. The example scripts are as follows:

| Environment Variable| Default| Description|
|---------------------|--------|------------|
| `do_single_co`  | store_true | Whether to train single-tag baseline model with co-training ext|
| `do_single_tri` | store_true  | Whether to train single-tag baseline model with tri-training ext|
| `co_sents_dir` | random-co-train/co-ext-data/ext-data-t0/1482_ext_L_A_sents.pkl  | the sent dir of co-training ext data|
| `co_labels_dir` | random-co-train/co-ext-data/ext-data-t0/1482_ext_L_A_labels.pkl  | the label dir of co-training ext data|
| `tri_all_dir` | sub_data/ext-train-isw-s3.pkl | the ext_all dir of tri-training approach|
| `fix_len` | 50  | the number of single tag be introduced|
| `tag` | `PER`  | the tag for single-tag retraining|

## Co-training with single tag
`python --do_single_co --co_sents_dir "random-co-train/co-ext-data/ext-data-t0/1482_ext_L_A_sents.pkl" --co_labels_dir random-co-train/co-ext-data/ext-data-t0/1482_ext_L_A_labels.pkl --fix_len 50 --tag PER`
## Tri-training with single tag
`python --do_single_tri --tri_all_dir sub_data/ext-train-isw-s3.pkl --fix_len 50 --tag PER`

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