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

# Quick Start : 

## Train the baseline BERT model
`python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir baseline_model/ --max_seq_length 128 --do_train --do_eval --eval_on dev`

## Evaluate the model on dev or test set
``python run_ner.py --data_dir data/full-isw-release.tsv --output_dir baseline_model/ --max_seq_length 128 --do_eval --eval_on test``

## Performance:
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
***** Running evaluation: dev *****
Num examples = 3217
Batch size = 8
Evaluating: 100%|############################################| 403/403 

             precision    recall  f1-score   support

        NRP     0.9226    0.9189    0.9207       493
       DATE     0.7750    0.7561    0.7654       246
        LAN     0.9432    0.9555    0.9493       382
        EVT     0.9143    0.7805    0.8421        41
        GPE     0.9642    0.9706    0.9674       749
       TIME     0.8556    0.9163    0.8849       705
       FREQ     0.7838    0.8056    0.7945       108
        FAC     0.6716    0.7759    0.7200        58
        DUR     0.6942    0.6844    0.6893       282
    ORDINAL     0.8750    0.8235    0.8485        85
        PER     0.8211    0.8715    0.8455       179
        AGE     0.6444    0.5743    0.6073       101
        ART     0.4545    0.2778    0.3448        18
       SORD     0.7273    0.7111    0.7191        45
        LOC     0.7674    0.7500    0.7586        44
       MISC     0.7500    0.6667    0.7059        27
       PERC     1.0000    1.0000    1.0000         8
    PRODUCT     0.0000    0.0000    0.0000         7
   CARDINAL     0.8652    0.8280    0.8462        93
       FRAC     0.4000    0.5000    0.4444         4
        MON     0.6667    0.6667    0.6667         9
      TITLE     0.8235    0.8235    0.8235        17
        ORG     0.6471    0.6667    0.6567        66
      QUANT     0.7143    0.6250    0.6667         8
        ADD     0.0000    0.0000    0.0000         1
       PROJ     1.0000    1.0000    1.0000         2
        MED     0.0000    0.0000    0.0000         1
       RATE     0.0000    0.0000    0.0000         3
        LAW     1.0000    1.0000    1.0000         1

avg / total     0.8541    0.8631    0.8580      3783
```

```
- INFO - __main__ -   ***** Running evaluation: test *****
- INFO - __main__ -     Num examples = 1609
- INFO - __main__ -     Batch size = 8
- INFO - __main__ -   ***** Eval results: test *****
- INFO - __main__ -   
             precision    recall  f1-score   support

    ORDINAL     0.7143    0.8333    0.7692        36
        NRP     0.9225    0.9049    0.9136       263
        GPE     0.9675    0.9808    0.9741       364
        LAN     0.9263    0.9670    0.9462       182
       DATE     0.7757    0.7830    0.7793       106
   CARDINAL     0.8364    0.7667    0.8000        60
        DUR     0.6639    0.5786    0.6183       140
        AGE     0.7027    0.6341    0.6667        41
       TIME     0.8685    0.8955    0.8818       354
        PER     0.8462    0.9462    0.8934        93
        ORG     0.7812    0.7143    0.7463        35
       FREQ     0.7451    0.8636    0.8000        44
        EVT     0.8750    0.7778    0.8235        18
       SORD     0.8438    0.7500    0.7941        36
        ART     0.3636    0.3636    0.3636        11
       MISC     0.8182    0.5294    0.6429        17
      TITLE     1.0000    1.0000    1.0000        14
        LOC     0.5714    0.4000    0.4706        10
       RATE     0.0000    0.0000    0.0000         1
    PRODUCT     0.0000    0.0000    0.0000         2
        FAC     0.8333    0.9091    0.8696        22
        MON     1.0000    1.0000    1.0000        10
      QUANT     1.0000    1.0000    1.0000         2
       PERC     1.0000    0.3333    0.5000         3
       PROJ     0.0000    0.0000    0.0000         1

avg / total     0.8614    0.8633    0.8609      1865
```

# Basline model configuration
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

# Simple API
For giving you an idea on the model prediction and further investigation. 

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