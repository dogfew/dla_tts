# Speaker Separation Project 

## Installation guide

Make shore that your python version >= 3.10

Run commands in `evaluate_script.sh`
```shell 
bash evaluate_script.sh
```
The commands in file `evaluate_script.sh` are: 
```shell
pip install -r requirements.txt
pip install gdown>4.7
mkdir -p default_test_model
cd default_test_model
gdown 1Cv50C8s3Qq54_lndi6AobRQlljUCExLl -O checkpoint.pth
gdown 1YNkqjKbgz3GzqN5NNUQ5q9suPvMXydOf -O config.json
cd ..

```
To get tests in 
```shell
python test.py -r default_test_model/checkpoint.pth -t <your_test>
```
If you want to check my custom test scores, you need to generate it using script `datasetscript.sh` and not use -t argument

## Training
To prepare data, run: 
```shell
pip install -r requirements.txt
bash prep_script.sh
```

To reproduce my final model, train FastSpeech2 with this config (400 epochs): 
```shell
python train.py -c src/configs/config_fastspeech2.json
```

**Optional Tasks:**

- (up to +1.5) MFA alignments (in textgrid format) which are downloadable with `prep_script.sh`.
Alignments are created in a script `generate_data_mfc.py` and the final model
is trained on them. 