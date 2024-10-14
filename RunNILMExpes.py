import torch
import torch.nn as nn

from Helpers.data_processing import *
from Helpers.torch_dataset import *
from Helpers.torch_trainer import *
from Helpers.utils import *

# SOTA Models
from Models.Sota.NILM.BiGRU import BiGRU
from Models.Sota.NILM.BiLSTM import BiLSTM
from Models.Sota.NILM.CNN1D import CNN1D
from Models.Sota.NILM.UNET_NILM import UNetNiLM
from Models.Sota.NILM.FCN import FCN
from Models.Sota.NILM.BERT4NILM import BERT4NILM
from Models.Sota.NILM.DResNets import DAResNet, DResNet
from Models.Sota.NILM.STNILM import STNILM
from Models.Sota.NILM.DiffNILM import DiffNILM

# NILMFormer Model
from Models.NILMFormers.NILMFormer  import NILMFormer


def get_instance_Model(name_model, c_in, window_size, **kwargs):

    if name_model == 'BiGRU':
        inst = BiGRU(c_in=1, window_size=window_size, **kwargs)
    elif name_model == 'BiLSTM':
        inst = BiLSTM(c_in=1,  window_size=window_size, **kwargs)
    elif name_model == 'CNN1D':
        inst = CNN1D(c_in=1, window_size=window_size, **kwargs)
    elif name_model == 'UNET_NILM':
        inst = UNetNiLM(c_in=1, window_size=window_size, **kwargs)
    elif name_model == 'FCN':
        inst = FCN(c_in=1, window_size=window_size, **kwargs)
    elif name_model == 'BERT4NILM':
        inst = BERT4NILM(c_in=1, window_size=window_size, **kwargs)
    elif name_model == 'DResNet':
        inst = DResNet(c_in=1, window_size=window_size, **kwargs)
    elif name_model == 'DAResNet':
        inst = DAResNet(c_in=1, window_size=window_size, **kwargs)
    elif name_model =='STNILM':
        inst = STNILM(c_in=1, window_size=window_size, **kwargs)
    elif name_model =='DiffNILM':
        inst = DiffNILM(**kwargs)
    elif name_model == 'NILMFormer':
        inst = NILMFormer(c_in=c_in, c_embedding=c_in-1, **kwargs)
    else:
        raise ValueError('Model name {} unknown'
                         .format(name_model))

    return inst


def launch_training_one_model(path, name_model, inst_model, param_training_model, param_training_global, tuple_data):

    if name_model == 'NILMFormer':
        train_dataset = NILMDataset(tuple_data[0], st_date=tuple_data[4], list_exo_variables=param_training_global['list_exo_variables'], 
                                    freq=param_training_global['sampling_rate'])

        valid_dataset = NILMDataset(tuple_data[1], st_date=tuple_data[5], list_exo_variables=param_training_global['list_exo_variables'], 
                                    freq=param_training_global['sampling_rate'])

        test_dataset  = NILMDataset(tuple_data[2], st_date=tuple_data[6], list_exo_variables=param_training_global['list_exo_variables'], 
                                    freq=param_training_global['sampling_rate'])
    elif 'DiffNILM' in path:
        train_dataset = NILMDataset(tuple_data[0], st_date=tuple_data[4], list_exo_variables=['hour', 'dow', 'month'], freq=param_training_global['sampling_rate'], cosinbase=False, newRange=(-0.5, 0.5))
        valid_dataset = NILMDataset(tuple_data[1], st_date=tuple_data[5], list_exo_variables=['hour', 'dow', 'month'], freq=param_training_global['sampling_rate'], cosinbase=False, newRange=(-0.5, 0.5))
        test_dataset  = NILMDataset(tuple_data[2], st_date=tuple_data[6], list_exo_variables=['hour', 'dow', 'month'], freq=param_training_global['sampling_rate'], cosinbase=False, newRange=(-0.5, 0.5))
    else:
        train_dataset = NILMDataset(tuple_data[0])
        valid_dataset = NILMDataset(tuple_data[1])
        test_dataset  = NILMDataset(tuple_data[2])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param_training_global['batch_size'], shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1, shuffle=False)

    model_trainer = SeqToSeqTrainer(inst_model,
                                    train_loader=train_loader, valid_loader=valid_loader,
                                    learning_rate=param_training_model['lr'], weight_decay=param_training_model['wd'],
                                    criterion=nn.MSELoss(),
                                    f_metrics=NILMmetrics(),
                                    patience_es=param_training_global['p_es'], patience_rlr=param_training_global['p_rlr'],
                                    n_warmup_epochs=param_training_global['n_warmup_epochs'],
                                    verbose=True, plotloss=False, 
                                    save_fig=False, path_fig=None,
                                    device=param_training_global['device'], all_gpu=param_training_global['all_gpu'],
                                    save_checkpoint=True, path_checkpoint=path)

    if not check_if_model_is_train(path+'.pt'):
        print('Training...')
        model_trainer.train(param_training_global['epochs'])
    else:
        print('Already trained.')
        model_trainer.log = torch.load(path+'.pt', map_location=param_training_global['device'])

    model_trainer.restore_best_weights()

    if not check_if_model_is_train_and_evaluate(path+'.pt'):
        print('Evaluation..')
        model_trainer.evaluate(valid_loader, scaler=param_training_global['scaler'], threshold_small_values=param_training_global['threshold'], save_outputs=True, mask='valid_metrics')
        model_trainer.evaluate(test_loader,  scaler=param_training_global['scaler'], threshold_small_values=param_training_global['threshold'], save_outputs=True, mask='test_metrics')
    else:
        print('Already evaluated.')

    print('Per period window aggregation estimation evaluation...')
    eval_win_energy_aggregation(tuple_data[2], tuple_data[6], model_trainer, 
                                scaler=param_training_global['scaler'], metrics=NILMmetrics(round_to=5),
                                window_size=param_training_global['window_size'], 
                                freq=param_training_global['sampling_rate'], list_exo_variables=param_training_global['list_exo_variables'] if name_model=='NILMFormer' else [],
                                threshold_small_values=param_training_global['threshold']
                                )

    model_trainer.save()

    return


def launch_models_experiments(path, tuple_data, param_training_global, seed):

    c_in = 1 + 2 * len(param_training_global['list_exo_variables'])

    list_models = {'BiGRU': {'kwargs': {}, 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   ,
                   'BiLSTM': {'kwargs': {}, 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   ,
                   'CNN1D': {'kwargs': {}, 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   ,
                   'UNET_NILM': {'kwargs': {}, 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   ,
                   'FCN': {'kwargs': {}, 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   ,
                   'DResNet': {'kwargs': {}, 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   ,
                   'DAResNet': {'kwargs': {}, 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   ,
                   'BERT4NILM': {'kwargs': {'cutoff': param_training_global['cutoff'], 'threshold': param_training_global['threshold']},
                                 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   ,
                   'DiffNILM': {'kwargs': {}, 'param_training_model': {'lr': 1e-3, 'wd': 0, 'training_in_model': True}} # Parameters define in model trainer for DiffNILM
                    ,
                    'STNILM': {'kwargs': {}, 'param_training_model': {'lr': 1e-3, 'wd': 0, 'training_in_model': True}}
                    ,
                   'NILMFormer': {'kwargs': {}, 'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                   }

    name_model = param_training_global['name_model']
    values     = list_models[param_training_global['name_model']]
    instance_model = get_instance_Model(name_model=name_model, c_in=c_in, window_size=param_training_global['window_size'], **values['kwargs'])
    launch_training_one_model(path+name_model+'_'+str(seed), name_model, instance_model, values['param_training_model'], param_training_global, tuple_data)
    del instance_model

    return