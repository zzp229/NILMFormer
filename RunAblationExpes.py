import torch
import torch.nn as nn

from Helpers.data_processing import *
from Helpers.torch_dataset import *
from Helpers.torch_trainer import *
from Helpers.utils import *

# Homemade Models
from Models.NILMFormers.NILMFormerAbPE import NILMFormerAbPE
from Models.NILMFormers.NILMFormerAbPEAdd import NILMFormerAbPEAdd
from Models.NILMFormers.NILMFormerAbSt import NILMFormerAbStats
from Models.NILMFormers.NILMFormerAbEmbed import NILMFormerAbEmbed
from Models.NILMFormers.NILMFormerAbEmbedPatch import NILMFormerAbEmbedPatch


def get_instance_NILMFormerAb(name_model, c_in, **kwargs):

    if 'NILMFormerAbPE' in name_model:
        if name_model=='NILMFormerAbPEAdd':
            inst = NILMFormerAbPEAdd(c_in=1, c_embedding=c_in-1, **kwargs)
        else:
            inst = NILMFormerAbPE(c_in=1, **kwargs)
    elif 'NILMFormerAbSt' in name_model:
        inst = NILMFormerAbStats(c_in=1, c_embedding=c_in-1, **kwargs)
    elif 'NILMFormerAbEmbed' in name_model:
        if name_model=='NILMFormerAbEmbedPatch':
            inst = NILMFormerAbEmbedPatch(c_in=1, **kwargs)
        else:
            inst = NILMFormerAbEmbed(c_in=1, c_embedding=c_in-1, **kwargs)

    return inst


def launch_training_one_model(path, inst_model, param_training_model, param_training_global, tuple_data):

    train_dataset = NILMDataset(tuple_data[0], st_date=tuple_data[4], list_exo_variables=param_training_global['list_exo_variables'], 
                                freq=param_training_global['sampling_rate'])

    valid_dataset = NILMDataset(tuple_data[1], st_date=tuple_data[5], list_exo_variables=param_training_global['list_exo_variables'], 
                                freq=param_training_global['sampling_rate'])

    test_dataset  = NILMDataset(tuple_data[2], st_date=tuple_data[6], list_exo_variables=param_training_global['list_exo_variables'], 
                                freq=param_training_global['sampling_rate'])

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

    print('Eval win aggregation...')
    eval_win_energy_aggregation(tuple_data[2], tuple_data[6], model_trainer, 
                                scaler=param_training_global['scaler'], metrics=NILMmetrics(round_to=5),
                                window_size=param_training_global['window_size'], 
                                freq=param_training_global['sampling_rate'], list_exo_variables=param_training_global['list_exo_variables'],
                                threshold_small_values=param_training_global['threshold']
                                )

    model_trainer.save()

    return





def launch_models_experiments(path, tuple_data, param_training_global, seed):

    c_in = 1 + 2 * len(param_training_global['list_exo_variables'])

    list_models = {'NILMFormerAbStwols': {'kwargs': {'instance_norm': True, 'add_token_stat': True, 'learn_stats' : False},  
                                          'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbStwotokenst': {'kwargs': {'instance_norm': True, 'add_token_stat': False, 'learn_stats' : True},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbStwolsandtoken': {'kwargs': {'instance_norm': True, 'add_token_stat': False, 'learn_stats' : False},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbStrevin': {'kwargs': {'instance_norm': True, 'revin': True, 'add_token_stat': False, 'learn_stats' : False},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbStwonorm': {'kwargs': {'instance_norm': False, 'add_token_stat': False, 'learn_stats' : False},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbPEfixed': {'kwargs': {'type_pe': 'fixed', 'window_size': param_training_global['window_size']},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbPEtAPE': {'kwargs': {'type_pe': 'tAPE', 'window_size': param_training_global['window_size']},  
                                            'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbPElearn': {'kwargs': {'type_pe': 'learnable', 'window_size': param_training_global['window_size']},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbPEno': {'kwargs': {'type_pe': 'noencoding', 'window_size': param_training_global['window_size']},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbPEAdd': {'kwargs': {'instance_norm': True, 'learn_stats':True},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbEmbedConv': {'kwargs': {'type_embed': 'conv'},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbEmbedLinear': {'kwargs': {'type_embed': 'linear'},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    ,
                    'NILMFormerAbEmbedPatch': {'kwargs': {'window_size': param_training_global['window_size']},  
                                              'param_training_model': {'lr': 1e-4, 'wd': 0, 'training_in_model': False}}
                    }

    #for name_model, values in list_models.items():
    name_model = param_training_global['name_model']
    values     = list_models[param_training_global['name_model']]

    instance_model = get_instance_NILMFormerAb(name_model, c_in=c_in, **values['kwargs'])
    launch_training_one_model(path+name_model + '_' + str(seed), instance_model, values['param_training_model'], param_training_global, tuple_data)
    del instance_model

    return