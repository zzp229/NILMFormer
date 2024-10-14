import os, sys
import numpy as np

current_dir = os.getcwd()
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Helpers.data_processing import *
from Helpers.torch_dataset import *
from Helpers.torch_trainer import *
from Helpers.utils import *

from RunTSERExpes import *


def launch_cases_experiments(path, param_training_global, seed):

    np.random.seed(seed=seed)

    data_builder = REFIT_DataBuilder(data_path=parent_dir+'/Data/REFIT/RAW_DATA_CLEAN/',
                                     mask_app=param_training_global['app'],
                                     sampling_rate=param_training_global['sampling_rate'],
                                     window_size=param_training_global['window_size'])
    
    if isinstance(param_training_global['window_size'], str):
        param_training_global['window_size'] = data_builder.window_size

    data, st_date = data_builder.get_nilm_dataset(house_indicies=param_training_global['house_with_app_i'])

    data_train, st_date_train, data_test, st_date_test = Split_train_test_pdl_NILMDataset(data.copy(), st_date.copy(), 
                                                                                          nb_house_test=2, seed=seed)
    data_train, st_date_train, data_valid, st_date_valid = Split_train_test_pdl_NILMDataset(data_train, st_date_train, 
                                                                                            nb_house_test=1, seed=seed)

    scaler = NILMscaler(power_scaling_type=param_training_global['power_scaling_type'],
                        appliance_scaling_type=param_training_global['appliance_scaling_type'])
    data = scaler.fit_transform(data)

    X, y = NILMdataset_to_TSER(data)
    param_training_global['scaler'] = scaler

    data_train = scaler.transform(data_train)
    data_valid = scaler.transform(data_valid)
    data_test  = scaler.transform(data_test)

    X_train, y_train = NILMdataset_to_TSER(data_train)
    X_valid, y_valid = NILMdataset_to_TSER(data_valid)
    X_test,  y_test  = NILMdataset_to_TSER(data_test)

    tuple_data = ((X_train, y_train, st_date_train), 
                  (X_valid, y_valid, st_date_valid), 
                  (X_test,  y_test,  st_date_test), 
                  (X,       y,       st_date))

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

    try :
        window_size = int(window_size)
    except:
        pass

    path_results = current_dir + '/ResultsTSER/'
    _ = create_dir(path_results)

    #============= Base case with all possible houses =============#
    dict_case_param = {'Dishwasher': {'app': 'Dishwasher',  
                                      'house_with_app_i': [1, 2, 3, 5, 6, 7, 9, 10, 13, 15, 16, 18, 20]},
                       'WashingMachine': {'app': 'WashingMachine',  
                                          'house_with_app_i': [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19]},
                       'Kettle': {'app': 'Kettle',  
                                  'house_with_app_i': [2, 3, 4, 5, 6, 7, 9, 12, 13, 15, 19, 20]},
                       'Microwave': {'app': 'Microwave',  
                                     'house_with_app_i': [2, 3, 4, 6, 8, 9, 10, 12, 13, 15, 17, 18, 19, 20]}
                      }

    # ====== List of dict parameters ===== #
    param_training_global = {'name_model': name_model,
                             'sampling_rate': '1min',  'window_size': window_size,
                             'list_exo_variables': ['minute', 'hour', 'dow', 'month'],
                             'power_scaling_type': 'MaxScaling', 'appliance_scaling_type': 'SameAsPower', 
                             'batch_size': 64, 'epochs': 50, 'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0,
                             'device': 'cuda', 'all_gpu': True}

    path_results = create_dir(path_results+str(param_training_global['sampling_rate'])+'/')
    path_results = create_dir(path_results+str(param_training_global['window_size'])+'/')

    print('Launch training...')
    param_training_global.update(dict_case_param[case])
    launch_cases_experiments(path_results, param_training_global, seed)    

