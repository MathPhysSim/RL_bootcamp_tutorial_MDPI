# TODO: save data for accelerated verification in the verification functions


# TODO: think about overall structure of storing results
import os
import pickle
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


# Importing required functions and classes
from helper_scripts.general_helpers import verify_external_policy_on_specific_env, prepare_experiment_folder, \
    run_specific_test
from environment.environment_helpers import read_experiment_config, load_env_config

optimization_type = 'random_walk'
algorithm = 'random_policy'
experiment_name = 'my_fist_noise_test'



environment_settings = read_experiment_config('config/environment_setting.yaml')
DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes

# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')

# save_folder_figures = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Figures')
save_folder_results = prepare_experiment_folder(optimization_type, algorithm, environment_settings, experiment_name=experiment_name)



for _ in range(5):
    # Get the current date and time
    # Get the current date and time
    current_time = datetime.now()
    # Format the date and time as a string to include milliseconds
    filename = current_time.strftime("%Y%m%d_%H%M%S_%f")  # %f will give microsecond precision
    # To get milliseconds, you can slice the microseconds to the first three digits
    filename = filename[:-3]  # This removes the last three digits of microseconds to keep milliseconds
    # Example of
    full_filename = f"experiment_results_{filename}.pkl"
    save_results_name = os.path.join(save_folder_results, full_filename)

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



# Create a figure for plotting
plt.figure(figsize=(10, 5))

# Read each pickle file from the results directory, assuming files are named in a way that preserves order
file_list = sorted([f for f in os.listdir(save_folder_results) if f.startswith('experiment_results_') and f.endswith('.pkl')])
for file_name in file_list:
    with open(os.path.join(save_folder_results, file_name), 'rb') as file:
        data = pickle.load(file)
        ep_len_per_task = data['ep_len_per_task']
        # Plot episode lengths for each task in this specific file
        for i, ep_lengths in enumerate(ep_len_per_task):
            plt.plot(ep_lengths, label=f'Task {i+1} from {file_name}', drawstyle='steps-post', linewidth=2)

plt.title('Episode Lengths per Task')
plt.xlabel('Episode')
plt.ylabel('Length of Episode')
plt.legend()
plt.grid(True)
plt.show()




file_list = sorted(
    [f for f in os.listdir(save_folder_results) if f.startswith('experiment_results_') and f.endswith('.pkl')])
rewards_all = []

# Gather all cumulative rewards from each file
for file_name in file_list:
    with open(os.path.join(save_folder_results, file_name), 'rb') as file:
        data = pickle.load(file)
        rewards_per_task = data['rewards_per_task']

        # Calculate cumulative rewards for each episode in each task
        for rewards in rewards_per_task[0]:
            print(rewards)
            cumulative_rewards = np.cumsum(rewards)
            rewards_all.append(cumulative_rewards)



# Pad shorter reward arrays to have uniform lengths
max_length = max(len(rewards) for rewards in rewards_all)
rewards_all_padded = [np.pad(rewards, (0, max_length - len(rewards)), 'edge') for rewards in rewards_all]

# Convert list to a NumPy array for easier manipulation
rewards_all_array = np.array(rewards_all_padded)

# Calculate mean and standard deviation across all tasks/iterations
mean_rewards = np.mean(rewards_all_array, axis=0)
std_rewards = np.std(rewards_all_array, axis=0)

# Plotting
plt.figure(figsize=(10, 5))
steps = np.arange(max_length)  # assuming steps are 0-indexed and correspond to indices in the array
plt.plot(steps, mean_rewards, label='Mean Cumulative Reward', color='blue')
plt.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2,
                 label='Standard Deviation')
plt.title('Mean and Standard Deviation of Cumulative Rewards Over Steps')
plt.xlabel('Step')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.grid(True)
plt.show()