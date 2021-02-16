Bash instructions on how to run the scripts.

1) Create virtual environment in the root directory (where this file lies)

>> python3 -m venv env

2) Activate the environment and install dependencies using pip

>> . env/bin/activate
>> pip install -r requirements.txt

3) Run script from the root directory, i.e.

>> python app/logistic_regr.py # Runs logistic regression for you

>> python app/gru.py # runs Encoding + 2GRUs model (this also trains the model, takes long)

>> python app/train_bert.py test cnn bert-base-uncased gpu # Run testing bert on gpu
>> python app/train_bert.py test cnn bert-base-uncased cpu # Run testing bert on cpu if you don't have cuda

------ IMPORTANT -------

If you want to train BERT you have to

1) Go inside app/train_bert.py and change the USAGE doc type into

"""
Usage: 
    train_bert.py train MODEL BERT_CONFIG CUDA [options]
    train_bert.py train MODEL BERT_CONFIG CUDA [options]
Options:
    -h --help                               show this screen.
    --train=<file>                          train file [default: bert/train_bert.csv]
    --dev=<file>                            dev file [default: bert/valid_bert.csv]
    --test=<file>                           test file [default: bert/test_bert.csv]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --hidden-size=<int>                     hidden size for lstm [default: 256]
    --out-channel=<int>                     out channel for cnn [default: 16]
    --clip-grad=<float>                     gradient clipping [default: 1.0]
    --log-every=<int>                       log every [default: 5]
    --max-epoch=<int>                       max epoch [default: 20]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 3]
    --max-num-trial=<int>                   terminate training after how many trials [default: 3]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr-bert=<float>                       BERT learning rate [default: 0.00002]
    --lr=<float>                            learning rate [default: 0.001]
    --valid-niter=<int>                     perform validation after how many iterations [default: 300]
    --dropout=<float>                       dropout [default: 0.3]
    --verbose                               whether to output the test results
"""

then run

>> python app/train_bert.py train cnn bert-base-uncased gpu # train Bert on GPU
>> python app/train_bert.py test cnn bert-base-uncased cpu # train Bert on CPU



------------IMPORTANT

If you run out of memory when running Bert, decrease the batch_size into 15


"""
Usage: 
    train_bert.py train MODEL BERT_CONFIG CUDA [options]
    train_bert.py train MODEL BERT_CONFIG CUDA [options]
Options:
    -h --help                               show this screen.
    --train=<file>                          train file [default: bert/train_bert.csv]
    --dev=<file>                            dev file [default: bert/valid_bert.csv]
    --test=<file>                           test file [default: bert/test_bert.csv]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 15]
    --hidden-size=<int>                     hidden size for lstm [default: 256]
    --out-channel=<int>                     out channel for cnn [default: 16]
    --clip-grad=<float>                     gradient clipping [default: 1.0]
    --log-every=<int>                       log every [default: 5]
    --max-epoch=<int>                       max epoch [default: 20]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 3]
    --max-num-trial=<int>                   terminate training after how many trials [default: 3]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr-bert=<float>                       BERT learning rate [default: 0.00002]
    --lr=<float>                            learning rate [default: 0.001]
    --valid-niter=<int>                     perform validation after how many iterations [default: 300]
    --dropout=<float>                       dropout [default: 0.3]
    --verbose                               whether to output the test results
"""



 