# Machine translation

## Install JoeyNMT
1. go to `https://joeynmt.readthedocs.io/en/latest/install.html` for more details
2. switch to your venv `conda activate venv` , make sure you have python=3.6, pip=20
3. 
```
git clone https://github.com/joeynmt/joeynmt.git
cd joeynmt
pip install .
```

## File Translation
1. go to `https://joeynmt.readthedocs.io/en/latest/tutorial.html#file-translation`
2. Download pretrained machine translation model. i.e. `en-de`, `de-en`
3. In the folder you may find the model I have downloaded. i.e. `wmt_ende_best`, `wmt_ende_transformer`, `iwslt14-deen-bpe`

### Use En-De
There are two options, but runing in same way:
```
cd wmt_ende_best
python -m joeynmt translate wmt_ende_best/config.yaml
```
Or
```
cd wmt_ende_transformer
python -m joeynmt translate wmt_ende_transformer/config.yaml
```

### Use De-En
```
python -m joeynmt translate iwslt14-deen-bpe/config.yaml
```
