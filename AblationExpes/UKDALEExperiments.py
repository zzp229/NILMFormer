import os, sys
import numpy as np

current_dir = os.getcwd()
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Helpers.data_processing import *
from Helpers.torch_dataset import *
from Helpers.torch_trainer import *
from Helpers.utils import *

from RunAblationExpes import *


def launch_cases_experiments(path, param_training_global, seed):

    np.random.seed(seed=seed)

    data_builder = UKDALE_DataBuilder(data_path=parent_dir+'/Data/UKDALE/',
                                      mask_app=param_training_global['app'],
                                      sampling_rate=param_training_global['sampling_rate'],
                                      window_size=param_training_global['window_size'])

    data, st_date = data_builder.get_nilm_dataset(house_indicies=[1, 2, 3, 4, 5])

    if isinstance(param_training_global['window_size'], str):
        param_training_global['window_size'] = data_builder.window_size

    data_train, st_date_train = data_builder.get_nilm_dataset(house_indicies=param_training_global['ind_house_train'])
    data_test,  st_date_test  = data_builder.get_nilm_dataset(house_indicies=param_training_global['ind_house_test'])

    data_train, st_date_train, data_valid, st_date_valid = Split_train_test_NILMDataset(data_train, st_date_train, perc_house_test=0.2, seed=seed)

    scaler = NILMscaler(power_scaling_type=param_training_global['power_scaling_type'], 
                        appliance_scaling_type=param_training_global['appliance_scaling_type'])
    data = scaler.fit_transform(data)
    param_training_global['scaler'] = scaler
    param_training_global['cutoff'] = scaler.appliance_stat2[0]

    data_train = scaler.transform(data_train)
    data_valid = scaler.transform(data_valid)
    data_test  = scaler.transform(data_test)

    tuple_data = (data_train, data_valid, data_test, data, st_date_train, st_date_valid, st_date_test, st_date)

    path_app = create_dir(path+param_training_global['app']+'/')

    launch_models_experiments(path_app, tuple_data, param_training_global, seed)

    return


if __name__ == "__main__":

    window_size   = str(sys.argv[1])
    case          = str(sys.argv[2])
    name_model    = str(sys.argv[3])
    seed          = int(sys.argv[4])

    try :
        window_size = int(window_size)
    except:
        pass

    print('window_size : ', window_size)
    print('case : ', case)
    print('name_model : ', name_model)
    print('seed : ', seed)

    path_results = current_dir + '/Results/'
    _ = create_dir(path_results)

    #============= Base case with all possible houses =============#
    dict_case_param = {'washing_machine':{'app': 'washing_machine',  
                                          'ind_house_train': [1, 3, 4, 5],
                                          'ind_house_valid': [1],
                                          'ind_house_test': [2]},
                       'dishwasher': {'app': 'dishwasher',  
                                      'ind_house_train': [1, 3, 4, 5],
                                      'ind_house_valid': [1],
                                      'ind_house_test': [2]},
                       'kettle': {'app': 'kettle',  
                                  'ind_house_train': [1, 3, 4, 5],
                                  'ind_house_valid': [1],
                                  'ind_house_test': [2]},
                       'microwave': {'app': 'microwave',  
                                     'ind_house_train': [1, 3, 4, 5],
                                     'ind_house_valid': [1],
                                     'ind_house_test': [2]},
                       'fridge': {'app': 'fridge',  
                                  'ind_house_train': [1, 3, 4, 5],
                                  'ind_house_valid': [1],
                                  'ind_house_test': [2]}
                       }

    # ====== List of dict parameters ===== #
    param_training_global = {'name_model': name_model,
                             'sampling_rate': '1min',  'window_size': window_size,
                             'list_exo_variables': ['minute', 'hour', 'dow', 'month'], 'use_temperature': False,
                             'power_scaling_type': 'MaxScaling', 'appliance_scaling_type': 'SameAsPower', 
                             'batch_size': 64, 'epochs': 50, 'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0,
                             'device': 'cuda', 'all_gpu': True}

    path_results = create_dir(path_results+str(param_training_global['sampling_rate'])+'/')
    path_results = create_dir(path_results+str(param_training_global['window_size'])+'/')

    print('Launch training...')
    param_training_global.update(dict_case_param[case])
    launch_cases_experiments(path_results, param_training_global, seed)

