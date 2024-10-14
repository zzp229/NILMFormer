import torch
import torch.nn as nn

from Helpers.data_processing import *
from Helpers.torch_dataset import *
from Helpers.torch_trainer import *
from Helpers.utils import *

# SOTA Models
from Models.Sota.TSER.ConvNet import ConvNet
from Models.Sota.TSER.ResNet import ResNet
from Models.Sota.TSER.InceptionTime import Inception


def get_instance_Model(name_model, **kwargs):

    if name_model=='ConvNet':
        inst = ConvNet(in_channels=1, nb_class=1, **kwargs)
    elif name_model=='ResNet':
        inst = ResNet(in_channels=1, nb_class=1, **kwargs)
    elif name_model =='Inception':
        inst = Inception(in_channels=1, nb_class=1, **kwargs)
    else:
        raise ValueError('Model name {} unknown'
                         .format(name_model))

    return inst


def launch_training_one_model(path, inst_model, param_training_model, param_training_global, tuple_data):

    train_dataset = TSDatasetScaling(tuple_data[0][0], tuple_data[0][1])
    valid_dataset = TSDatasetScaling(tuple_data[1][0], tuple_data[1][1])
    test_dataset  = TSDatasetScaling(tuple_data[2][0], tuple_data[2][1])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param_training_global['batch_size'], shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1, shuffle=False)

    loss = nn.MSELoss()

    model_trainer = TserTrainer(inst_model,
                                train_loader=train_loader, valid_loader=valid_loader,
                                learning_rate=param_training_model['lr'], weight_decay=param_training_model['wd'],
                                criterion=loss,
                                f_metrics=NILMmetrics(), 
                                patience_es=param_training_global['p_es'], patience_rlr=param_training_global['p_rlr'],
                                n_warmup_epochs=param_training_global['n_warmup_epochs'],
                                verbose=True, plotloss=False, 
                                save_fig=False, path_fig=None,
                                device=param_training_global['device'], all_gpu=param_training_global['all_gpu'],
                                save_checkpoint=True, path_checkpoint=path)

    model_trainer.train(param_training_global['epochs'])
    model_trainer.restore_best_weights()

    model_trainer.evaluate(valid_loader, 
                           scaler=param_training_global['scaler'] if 'scaler' in param_training_global else None,
                           factor_scaling=param_training_global['factor_scaling_app'] if 'factor_scaling_app' in param_training_global else 1, 
                           threshold_small_values=0, save_outputs=True, mask='valid_metrics')

    model_trainer.evaluate(test_loader,
                           scaler=param_training_global['scaler'] if 'scaler' in param_training_global else None,
                           factor_scaling=param_training_global['factor_scaling_app'] if 'factor_scaling_app' in param_training_global else 1, 
                           threshold_small_values=0, save_outputs=True, mask='test_metrics')

    print(model_trainer.log['test_metrics_win'])

    return


def launch_models_experiments(path, tuple_data, param_training_global, seed):

    c_in = 1 + 2 * len(param_training_global['list_exo_variables'])

    list_models = {'ConvNet': {'kwargs': {}, 'param_training_model': {'lr': 1e-3, 'wd': 0}}
                   ,
                   'ResNet': {'kwargs': {}, 'param_training_model': {'lr': 1e-3, 'wd': 0}}
                   ,
                   'Inception': {'kwargs': {}, 'param_training_model': {'lr': 1e-3, 'wd': 0}}
                  }

    name_model = param_training_global['name_model']
    values     = list_models[param_training_global['name_model']]
    instance_model = get_instance_Model(name_model=name_model, **values['kwargs'])
    launch_training_one_model(path+name_model+'_'+str(seed), instance_model, values['param_training_model'], param_training_global, tuple_data)
    del instance_model

    return