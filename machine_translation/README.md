# Machine translation

## Install JoeyNMT
1. go to `https://joeynmt.readthedocs.io/en/latest/install.html` for more details
2. switch to your venv `conda activate venv` , make sure you have python=3.6, pip=20
3. 
```
cd machine_translation
git clone https://github.com/joeynmt/joeynmt.git
cd joeynmt
pip install .

* Under dir /machine_translation
mkdir models
cd models
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
 
```
cd wmt_ende_transformer
python -m joeynmt translate wmt_ende_transformer/config.yaml
```

### Use De-En
```
Under dir /machine_translation
cd models
curl -o iwslt14-deen-bpe.tar.gz https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/iwslt14-deen-bpe.tar.gz
tar -xvf iwslt14-deen-bpe.tar.gz

cd /machine_translation
python -m joeynmt translate models/iwslt14-deen-bpe/config.yaml

python -m joeynmt translate models/iwslt14-deen-bpe/config.yaml < 2017_de_sents.txt --output_path 2017_en_sents.txt
```
