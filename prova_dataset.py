import torch
import random

##### __init__  #####
env_ids = [0, 1, 2, 3, 4, 5]
num_envs = len(env_ids)
num_dof = 8
num_actions = num_dof
use_dataset_percentage = 0.5              # nuovo termine da inserire come parametro nello yaml
base_pos = torch.zeros((num_envs, 3), dtype=torch.float)
base_quat = torch.zeros((num_envs, 4), dtype=torch.float)
base_velocities = torch.zeros((num_envs, 6), dtype=torch.float)    # angolari + lineari
env_origins = torch.zeros((num_envs, 3), requires_grad=False)
dof_pos = torch.zeros((num_envs, num_dof), dtype=torch.float)
dof_vel = torch.zeros((num_envs, num_dof, 3), dtype=torch.float)
default_dof_pos = torch.zeros((num_envs, 8), dtype=torch.float, requires_grad=False)
pos = [0.0, 0.0, 0.35]      # 4
rot = [1.0, 0.0, 0.0, 0.0]  # 3
v_lin = [0.0, 0.0, 0.0]     # 3
v_ang = [0.0, 0.0, 0.0]     # 3
base_init_state = pos + rot + v_lin + v_ang  # 13
# -------------------------------------------------------------------------------------------------------------------- #
# creazione di un dizionario (annidato) = insieme non ordinato di coppie chiave:valore(valore=altro dizionario contenente i test);
# richiede inserimento di almeno 1 test
failure_dataset = { 'dof_pos': {
                        'test1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        'test2': torch.ones(8)
                    },
                    # 'dof_vel': {
                    #    'test1': torch.ones(8),
                    #    'test2': torch.zeros(8)
                    # },         # questo va sistemato perchè per i calcoli di accel e jerk ora questo ha una dimensione in più
                    'base_quat': {
                        'test1': [1, 2, 3, 4],
                        'test2': torch.ones(4) * 2
                    },
                    'base_pos': {
                        'test1': [10, 20, 30],
                        'test2': torch.ones(3) * 3
                    },
                    'base_velocities': {
                        'test1': [0.01, 0.02, 0.03, 0.001, 0.002, 0.003],
                        'test2': torch.ones(6) * 4
                    }
}
# estrazione dei valori delle grandezze definite dentro una lista
lista_dof_pos = list(failure_dataset['dof_pos'].values())    # per accedere a un elemento della lista usare le []
tensore_dof_pos = torch.tensor(lista_dof_pos, dtype=torch.float) # per accedere a un elemento usare le []
#lista_dof_vel = list(failure_dataset['dof_vel'].values())
# tensore_dof_vel = torch.tensor(lista_dof_vel, dtype=torch.float)
lista_base_quat = list(failure_dataset['base_quat'].values())
tensore_base_quat = torch.tensor(lista_base_quat, dtype=torch.float)
lista_base_pos = list(failure_dataset['base_pos'].values())
tensore_base_pos = torch.tensor(lista_base_pos, dtype=torch.float)
lista_base_velocities = list(failure_dataset['base_velocities'].values())
tensore_base_velocities = torch.tensor(lista_base_velocities, dtype=torch.float)
# Lista dei test nel dataset
tests = list(failure_dataset['dof_pos'].keys())

print("failure_dataset  = ", failure_dataset)  # così mostra tutto il dizionario
print("La key dof_pos è ", failure_dataset['dof_pos'])
print("---------------------------------------------")
print("I dof_pos aggiunti sono ", lista_dof_pos)  # così accedo a tutti gli elementi di 'dof_pos' inseriti in una lista
print("visti come tensore invece sono ", tensore_dof_pos)
#print("I dof_vel aggiunti sono ", lista_dof_vel)
print("I base_quat aggiunti sono ", lista_base_quat)

# -------------------------------------------------------------------------------------------------------------------- #

#### get mulinex ####
base_init_state = torch.tensor(base_init_state, dtype=torch.float, requires_grad=False)
mulinex_translation = torch.tensor([0.0, 0.0, 0.35])  # da modificare se modifico C.I.
mulinex_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
mulinex = [mulinex_translation, mulinex_orientation]
dof_names = ["LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE", "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"]
default_dof_pos[:, 0], default_dof_pos[:, 3] = 2.094 , 2.094
default_dof_pos[:, 1], default_dof_pos[:, 2] = -2.094 , -2.094
default_dof_pos[:, 4], default_dof_pos[:, 7] = -1.0472 , -1.0472
default_dof_pos[:, 5], default_dof_pos[:, 5] = 1.0472 , 1.0472


#### reset_idx ####
print('---------------------------------------------------------------')
print('Assegnazione dei valori')
# Funzione per assegnare i valori dal dataset o calcolarli
# def assign_values_0(env_id, use_dataset = False):
#     if use_dataset:
#         # Definiamo le grandezze usando i valori del dataset
#         index = 0  # uso lo stesso indice per ogni lista, index corrisponde al test che stiamo usando, qua è inizializzato
#         # ??? TOGLIERE CICLO FOR E SOSTITUIRE INDEX CON ENV_ID ??? #
#         for key in failure_dataset['dof_pos'].keys():   # va bene usare una qualsiasi chiave perchè dovranno avere tutte lo stesso numero di elementi
#             dof_pos[env_id] = tensore_dof_pos[index]
#             # dof_vel[env_id] = tensore_dof_vel[index]
#             base_quat[env_id] = tensore_base_quat[index]
#             base_pos[env_id] = tensore_base_pos[index]
#             base_velocities[env_id] = tensore_base_velocities[index]
#
#             print('dof_pos per tutti gli env_ids diventano ')
#             print(dof_pos[env_id])
#             index += 1
#     else:  # Calcola i valori come avresti fatto normalmente
#         positions_offset = (1.5 - 0.5) * torch.rand(len(env_ids), num_dof) + 0.5
#         velocities = (0.1 + 0.1) * torch.rand(len(env_ids), num_dof) - 0.1
#         dof_pos[env_ids] = default_dof_pos[env_ids] * positions_offset
#         dof_vel[env_ids, :, -1] = velocities
#         base_pos[env_ids] = base_init_state[0:3]
#         base_pos[env_ids, 0:3] += env_origins[env_ids]
#         base_quat[env_ids] = base_init_state[3:7]
#         base_velocities[env_ids] = base_init_state[7:]
#
#         print('dof_pos per tutti gli env_ids diventano ')
#         print(dof_pos[env_id])
#
#
# assign_values_0(env_id=env_ids, use_dataset=True)
# print('---------------------')
# assign_values_0(env_id=env_ids)


def assign_values(env_ids, use_dataset_percentage, tests):

    """input: use_dataset_percentage è la percentuale di robot che riceveranno valori dal dataset;
    tests è una lista che contiene le key corrispondenti ai nomi dei vari test presenti nel dataset. \n
    ===> La funzione assign_values sceglie casualmente un sottoinsieme di env_ids che useranno i valori dal
    dataset ed assegna i valori ai robot in base a questa selezione.
    Per i robot che useranno il dataset, viene scelto casualmente un test da tests per ogni env_ids"""

    num_envs_using_dataset = int(len(env_ids) * use_dataset_percentage)  # numero
    env_ids_using_dataset = random.sample(env_ids, num_envs_using_dataset)   # identificativo dei robot che useranno il dataset

    for env_id in env_ids:  # itera su tutti i robot vivi
        if env_id in env_ids_using_dataset:  # se il robot è stato scelto per usare dataset
            # Scegli un test casuale dal dataset
            test_key = random.choice(tests)
            test_key_idx = tests.index(test_key)
            dof_pos[env_id] = tensore_dof_pos[test_key_idx]
            # dof_vel[env_id] = tensore_dof_vel[test_key_idx]
            base_quat[env_id] = tensore_base_quat[test_key_idx]
            base_pos[env_id] = tensore_base_pos[test_key_idx]
            base_velocities[env_id] = tensore_base_velocities[test_key_idx]

        else:
            # Calcola i valori come avresti fatto normalmente
            positions_offset = (1.5 - 0.5) * torch.rand(num_dof) + 0.5
            velocities = (0.1 + 0.1) * torch.rand(num_dof) - 0.1
            dof_pos[env_id] = default_dof_pos[env_id] * positions_offset
            dof_vel[env_id, :, -1] = velocities
            base_pos[env_id] = base_init_state[0:3]
            base_pos[env_id, 0:3] += env_origins[env_id]
            base_quat[env_id] = base_init_state[3:7]
            base_velocities[env_id] = base_init_state[7:]

        print('dof_pos per tutti gli env_ids diventano ')
        print(dof_pos)


# Assegna i valori ai robot
assign_values(env_ids, use_dataset_percentage = use_dataset_percentage, tests = tests)