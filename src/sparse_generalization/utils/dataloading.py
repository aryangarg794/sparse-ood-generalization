import dill
import os

from hydra.utils import to_absolute_path

from sparse_generalization.utils.datasets import BasicDataset, OneHotBoxWorld

def get_shapes_datasets(size: int, data_dir: str, one_hot: bool):
    data_path = os.path.join(data_dir, f"shapes_train_{size}.pl")
    data_path = to_absolute_path(data_path)

    with open(data_path, 'rb') as file:
        train_data = dill.load(file)
        file.close()

    test_id_path = os.path.join(data_dir, f"shapes_test.pl")
    with open(test_id_path, 'rb') as file:
        test_id = dill.load(file)
        file.close()
    
    val_id_path = os.path.join(data_dir, f"shapes_val.pl")
    with open(val_id_path, 'rb') as file:
        val_id = dill.load(file)
        file.close()

    test_a_path = os.path.join(data_dir, f"shapes_test_a.pl")
    with open(test_a_path, 'rb') as file:
        test_a = dill.load(file)
        file.close()

    test_b_path = os.path.join(data_dir, f"shapes_test_b.pl")
    with open(test_b_path, 'rb') as file:
        test_b = dill.load(file)
        file.close()

    val_a_path = os.path.join(data_dir, f"shapes_val_a.pl")
    with open(val_a_path, 'rb') as file:
        val_a = dill.load(file)
        file.close()

    val_b_path = os.path.join(data_dir, f"shapes_val_b.pl")
    with open(val_b_path, 'rb') as file:
        val_b = dill.load(file)
        file.close()
        
    dataset = BasicDataset(train_data['X_train'], train_data['Y_train'], one_hot)
    val_dataset_id = BasicDataset(val_id['X_train'], val_id['Y_train'], one_hot)
    val_dataset_a = BasicDataset(val_a['X_test_a'], val_a['Y_test_a'], one_hot)
    val_dataset_b = BasicDataset(val_b['X_test_b'], val_b['Y_test_b'], one_hot)
    test_dataset_id = BasicDataset(test_id['X_train'], test_id['Y_train'], one_hot)
    test_dataset_a = BasicDataset(test_a['X_test_a'], test_a['Y_test_a'], one_hot)
    test_dataset_b = BasicDataset(test_b['X_test_b'], test_b['Y_test_b'], one_hot)
    
    val_sets = [val_dataset_id, val_dataset_a, val_dataset_b]
    test_sets = [test_dataset_id, test_dataset_a, test_dataset_b]
    
    return dataset, val_sets, test_sets


def get_boxworld_datasets(size: int, num_pairs: int, data_dir: str, one_hot: bool = False):
    data_path = os.path.join(data_dir, f"boxworld_v2_train_{size}_pairs{num_pairs}.pl")
    data_path = to_absolute_path(data_path)
    data_cls = OneHotBoxWorld if one_hot else BasicDataset
    
    with open(data_path, 'rb') as file:
        train_data = dill.load(file)
        file.close()
        
    dataset = data_cls(train_data['X_train'], train_data['Y_train'])

    test_id_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_test_id_pairs{num_pairs}.pl"))
    with open(test_id_path, 'rb') as file:
        test_id = dill.load(file)
        file.close()

    val_id_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_val_id_pairs{num_pairs}.pl"))
    with open(val_id_path, 'rb') as file:
        val_id = dill.load(file)
        file.close()

    test_col_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_test_col_pairs{num_pairs}.pl"))
    with open(test_col_path, 'rb') as file:
        test_col = dill.load(file)
        file.close()

    val_col_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_val_col_pairs{num_pairs}.pl"))
    with open(val_col_path, 'rb') as file:
        val_col = dill.load(file)
        file.close()

    test_pair_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_test_pair_pairs{num_pairs}.pl"))
    with open(test_pair_path, 'rb') as file:
        test_pair = dill.load(file)
        file.close()

    val_pair_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_val_pair_pairs{num_pairs}.pl"))
    with open(val_pair_path, 'rb') as file:
        val_pair = dill.load(file)
        file.close()

    test_dist_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_test_dist_pairs{num_pairs}.pl"))
    with open(test_dist_path, 'rb') as file:
        test_dist = dill.load(file)
        file.close()

    val_dist_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_val_dist_pairs{num_pairs}.pl"))
    with open(val_dist_path, 'rb') as file:
        val_dist = dill.load(file)
        file.close()

    test_comb_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_test_comb_pairs{num_pairs}.pl"))
    with open(test_comb_path, 'rb') as file:
        test_comb = dill.load(file)
        file.close()

    val_comb_path = to_absolute_path(os.path.join(data_dir, f"boxworld_v2_val_comb_pairs{num_pairs}.pl"))
    with open(val_comb_path, 'rb') as file:
        val_comb = dill.load(file)
        file.close()

    val_dataset_id = data_cls(val_id['X_train'], val_id['Y_train'])
    test_dataset_id = data_cls(test_id['X_train'], test_id['Y_train'])
    val_dataset_col = data_cls(val_col['X_col'], val_col['Y_col'])
    test_dataset_col = data_cls(test_col['X_col'], test_col['Y_col'])
    val_dataset_pair = data_cls(val_pair['X_pair'], val_pair['Y_pair'])
    test_dataset_pair = data_cls(test_pair['X_pair'], test_pair['Y_pair'])
    val_dataset_dist = data_cls(val_dist['X_dist'], val_dist['Y_dist'])
    test_dataset_dist = data_cls(test_dist['X_dist'], test_dist['Y_dist'])
    val_dataset_comb = data_cls(val_comb['X_comb'], val_comb['Y_comb'])
    test_dataset_comb = data_cls(test_comb['X_comb'], test_comb['Y_comb'])
    
    val_sets = [val_dataset_id, val_dataset_col, val_dataset_pair, val_dataset_dist, val_dataset_comb]
    test_sets = [test_dataset_id, test_dataset_col, test_dataset_pair, test_dataset_dist, test_dataset_comb]
    
    return dataset, val_sets, test_sets