# TODO: save data for accelerated verification in the verification functions
import os
import pickle

from matplotlib import pyplot as plt

# from Compare_different_approaches import experiment_name
from environment.environment_helpers import read_experiment_config, load_env_config
# Importing required functions and classes
from helper_scripts.general_helpers import verify_external_policy_on_specific_env, make_experiment_folder, \
    run_specific_test

environment_settings = read_experiment_config('config/environment_setting.yaml')
print(environment_settings)
DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes


# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')
optimization_type = 'random_walk'
algorithm = 'random_policy'
experiment_name = 'noise_tests'

# save_folder_figures = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Figures')
save_folder_results = make_experiment_folder(optimization_type, algorithm, environment_settings, experiment_name=experiment_name)
save_results_name = os.path.join(save_folder_results, 'Results.pkl')

print(save_results_name)

for _ in range(5):
    rewards_per_task, ep_len_per_task, actions_per_task, states_per_task = run_specific_test(
        env=env,
        policy=None,
        episodes=nr_validation_episodes,
        seed_set=validation_seeds,
        save_results=save_results_name
    )

    save_dict = {'rewards_per_task': rewards_per_task, 'ep_len_per_task': ep_len_per_task,
                 'actions_per_task': actions_per_task, 'states_per_task': states_per_task}

    # with open(save_name_results, "rb") as f:
    #     save_dict = pickle.load(f)
    # Extract data from the dictionary

    rewards_per_task = save_dict.get('rewards_per_task')
    # print('rewards_per_task: ', rewards_per_task)
    ep_len_per_task = save_dict.get('ep_len_per_task')
    print('ep_len_per_task: ', ep_len_per_task)
    actions_per_task = save_dict.get('actions_per_task')
    # print('actions_per_task: ',actions_per_task)
    states_per_task = save_dict.get('states_per_task')
    # print('states_per_task: ', states_per_task)

# Let's plot the episode lengths for simplicity. Adapt this for other metrics as needed.
plt.figure(figsize=(10, 5))
for i, ep_lengths in enumerate(ep_len_per_task):
    plt.plot(ep_lengths, label=f'Task {i+1}')

plt.title('Episode Lengths per Task')
plt.xlabel('Episode')
plt.ylabel('Length of Episode')
plt.legend()
plt.grid(True)
plt.show()