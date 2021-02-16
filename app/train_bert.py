"""
Usage: 
    train_bert.py test MODEL BERT_CONFIG CUDA [options]
    train_bert.py test MODEL BERT_CONFIG CUDA [options]
Options:
    -h --help                               show this screen.
    --train=<file>                          train file [default: bert/train_bert.csv]
    --dev=<file>                            dev file [default: bert/valid_bert.csv]
    --test=<file>                           test file [default: bert/test_bert.csv]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 20]
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

from pytorch_pretrained_bert import BertAdam
from bert_model import CustomBertConvModel, CustomBertLSTMModel
import logging
import pickle
import numpy as np
import torch
import pandas as pd
import time
import sys
from docopt import docopt
from utils_bert import batch_iter
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

def validation(model, df_val, bert_size, loss_func, device):
    """ validation of model during training.
        @param model (nn.Module): the model being trained
        @param df_val (dataframe): validation dataset
        @param bert_size (str): large or base
        @param loss_func(nn.Module): loss function
        @param device (torch.device)
        @return avg loss value across validation dataset
    """

    was_training = model.training
    model.eval()

    df_val = df_val.sort_values(by='preprocessed_text_bert'+bert_size+'_length', ascending=False)

    preprocessed_text_bert = list(df_val.preprocessed_text_bert)
    information_label = list(df_val.information_label)

    val_batch_size = 32

    n_batch = int(np.ceil(df_val.shape[0]/val_batch_size))

    total_loss = 0.

    with torch.no_grad():
        for i in range(n_batch):
            sents = preprocessed_text_bert[i*val_batch_size: (i+1)*val_batch_size]
            targets = torch.tensor(information_label[i*val_batch_size: (i+1)*val_batch_size],
                                    dtype=torch.long, device=device)
            batch_size = len(sents)
            pre_softmax = model(sents).double()
            batch_loss = loss_func(pre_softmax, targets)

            # Get the loss multiplied to the batch_size
            total_loss += batch_loss.item() * batch_size

    if was_training:
        model.train()

    return total_loss/df_val.shape[0]

'''
Optimization steps, still run out of memory

If you want to train with batch size of desired_batch_size, then divide it by a reasonable number like 4 or 8 or 16…, this number is know as accumtulation_steps. 
Now change your batch size for the dataset to desired_batch_size/accumulation_steps and train your model as below

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(training_set):
        predictions = model(inputs)                     # Forward pass
        loss = loss_function(predictions, labels)       # Compute loss function
        loss = loss / accumulation_steps                # Normalize our loss (if averaged)
        loss.backward()                                 # Backward pass
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            model.zero_grad()                           # Reset gradients tensors

'''

def train(args):
    
    label_name = ['fake', 'real']

    device = torch.device("cuda:0" if args['CUDA'] == 'gpu' else "cpu")

    prefix = args['MODEL'] + '_' + args['BERT_CONFIG']
    
    bert_size = args['BERT_CONFIG'].split('-')[1]

    start_time = time.time()
    print('Importing data...', file=sys.stderr)
    df_train = pd.read_csv(args['--train'], index_col=0)
    df_val = pd.read_csv(args['--dev'], index_col=0)

    train_label = dict(df_train.information_label.value_counts())

    print("Train label", train_label)

    label_max = float(max(train_label.values()))

    print("Label max", label_max)

    train_label_weight = torch.tensor([label_max/train_label[i] for i in range(len(train_label))], device=device)

    print(train_label_weight)
    
    print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    start_time = time.time()
    print('Set up model...', file=sys.stderr)

    if args['MODEL'] == 'cnn':
        model = CustomBertConvModel(args['BERT_CONFIG'], device, float(args['--dropout']), len(label_name),
                                    out_channel=int(args['--out-channel']))
        optimizer = BertAdam([
                {'params': model.bert.parameters()},
                {'params': model.conv.parameters(), 'lr': float(args['--lr'])},
                {'params': model.hidden_to_softmax.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))
    elif args['MODEL'] == 'lstm':
        model = CustomBertLSTMModel(args['BERT_CONFIG'], device, float(args['--dropout']), len(label_name), lstm_hidden_size=int(args['--hidden-size']))

        optimizer = BertAdam([
                {'params': model.bert.parameters()},
                {'params': model.lstm.parameters(), 'lr': float(args['--lr'])},
                {'params': model.hidden_to_softmax.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))
    else:
        print('please input valid model')
        exit(0)
    
    model = model.to(device)
    print('Use device: %s' % device, file=sys.stderr)
    print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    model.train()

    cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight, reduction='mean')
    torch.save(cn_loss, 'loss_func')  # for later testing

    train_batch_size = int(args['--batch-size'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = prefix+'_model.bin'

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_examples = report_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Begin Maximum Likelihood training...')

    while True:
        epoch += 1

        for sents, targets in batch_iter(df_train, batch_size=train_batch_size, shuffle=True, bert=bert_size):  # for each epoch
            train_iter += 1 # increase training iteration
            # set gradients to zero before starting to do backpropagation.
            # Pytorch accummulates the gradients on subsequnt backward passes.
            optimizer.zero_grad() 
            batch_size = len(sents)
            pre_softmax = model(sents).double()

            loss = cn_loss(pre_softmax, torch.tensor(targets, dtype=torch.long, device=device))
            # The gradients are "stored" by the tensors themselves once you call backwards
            # on the loss.
            loss.backward()
            '''
             After computing the gradients for all tensors in the model, calling optimizer.step() makes the optimizer iterate over 
             all parameters (tensors) it is supposed to update and use their internally stored grad to update their values.
            '''
            optimizer.step()

            # loss.item() contains the loss for the mini-batch, but divided by the batch_size
            # that's why multiply by the batch_size
            batch_losses_val = loss.item() * batch_size
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, '
                      'cum. examples %d, speed %.2f examples/sec, '
                      'time elapsed %.2f sec' % (epoch, train_iter, report_loss / report_examples, cum_examples, report_examples / (time.time() - train_time),
                                        time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_examples = 0.
            
            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. examples %d' % (epoch, train_iter, cum_loss / cum_examples, 
                                cum_examples), file=sys.stderr)
                
                cum_loss = cum_examples = 0

                print('begin validation....', file=sys.stderr)

                validation_loss = validation(model, df_val, bert_size, cn_loss, device)   # dev batch size can be a bit larger

                print('validation: iter %d, loss %f' % (train_iter, validation_loss), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or validation_loss < min(hist_valid_scores)
                hist_valid_scores.append(validation_loss)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)

                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')

                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        print('load previously best model and decay learning rate to %f%%' %
                              (float(args['--lr-decay'])*100), file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= float(args['--lr-decay'])

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)    

def test(args):

    label_name = ['false', 'real']

    prefix = args['MODEL'] + '_' + args['BERT_CONFIG']

    bert_size = args['BERT_CONFIG'].split('-')[1]

    device = torch.device("cuda:0" if args['CUDA'] == 'gpu' else "cpu")
    
    print('load best model...')

    if args['MODEL'] == 'cnn':
        model = CustomBertConvModel.load(prefix+'_model.bin', device)
    elif args['MODEL'] == 'lstm':
        model = CustomBertLSTMModel.load(prefix+'_model.bin', device)

    model.to(device)

    model.eval()

    df_test = pd.read_csv(args['--test'], index_col=0)

    df_test = df_test.sort_values(by='preprocessed_text_bert'+bert_size+'_length', ascending=False)

    test_batch_size = 32

    n_batch = int(np.ceil(df_test.shape[0]/test_batch_size))

    cn_loss = torch.load('loss_func', map_location=lambda storage, loc: storage).to(device)

    preprocessed_text_bert = list(df_test.preprocessed_text_bert)
    information_label = list(df_test.information_label)

    test_loss = 0.
    prediction = []
    prob = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i in range(n_batch):
            sents = preprocessed_text_bert[i*test_batch_size: (i+1)*test_batch_size]
            targets = torch.tensor(information_label[i * test_batch_size: (i + 1) * test_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)

            pre_softmax = model(sents).double()
            
            batch_loss = cn_loss(pre_softmax, targets)
            test_loss += batch_loss.item()*batch_size
            prob_batch = softmax(pre_softmax)
            prob.append(prob_batch)

            prediction.extend([t.item() for t in list(torch.argmax(prob_batch, dim=1))])

    prob = torch.cat(tuple(prob), dim=0)
    loss = test_loss/df_test.shape[0]

    pickle.dump([label_name[i] for i in prediction], open(prefix+'_test_prediction', 'wb'))
    pickle.dump(prob.data.cpu().numpy(), open(prefix + '_test_prediction_prob', 'wb'))

    accuracy = accuracy_score(df_test.information_label.values, prediction)
    matthews = matthews_corrcoef(df_test.information_label.values, prediction)

    print(f'F score on the test set: {f1_score(df_test.information_label.values, prediction)}')
    print(f'Accuracy on the test set: {accuracy}')
    print('For more information look at the metrics_csv.csv file we created for you')

    precisions = {}
    recalls = {}
    f1s = {}
    aucrocs = {}

    for i in range(len(label_name)):
        prediction_ = [1 if pred == i else 0 for pred in prediction]
        true_ = [1 if label == i else 0 for label in df_test.information_label.values]
        f1s.update({label_name[i]: f1_score(true_, prediction_)})
        precisions.update({label_name[i]: precision_score(true_, prediction_)})
        recalls.update({label_name[i]: recall_score(true_, prediction_)})
        aucrocs.update({label_name[i]: roc_auc_score(true_, list(t.item() for t in prob[:, i]))})

    metrics_dict = {'loss': loss, 'accuracy': accuracy, 'matthews coef': matthews, 'precision': precisions,
                         'recall': recalls, 'f1': f1s, 'aucroc': aucrocs}
    
    metrics_dataframe = pd.DataFrame.from_dict(metrics_dict)
    metrics_dataframe.to_csv("metrics_csv.csv")

    pickle.dump(metrics_dict, open(prefix+'_evaluation_metrics', 'wb'))\


if __name__ == '__main__':

    args = docopt(__doc__)

    logging.basicConfig(level=logging.INFO)

    if args['train']:
        train(args)
    elif args['test']:
        test(args)
    else:
        raise RuntimeError('invalid run mode')