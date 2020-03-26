# thesis-ner-co-tri-training
This repo is aiming to implementing the co-training and tri-training framework to compare the result of fine-grained NER task

# To train model : 

`python run_ner.py --do_train --do_eval --max_seq_length 128 --data_dir data/train-full-isw-release.tsv`

# API 
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