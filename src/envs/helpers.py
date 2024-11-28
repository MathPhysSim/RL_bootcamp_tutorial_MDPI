from pathlib import Path
import pickle

def load_predefined_task(task_nr, task_location):
    # Check if the file exists
    task_location = Path().cwd().joinpath(task_location)

    with open(task_location, "rb") as input_file:  # Load in tasks
        tasks = pickle.load(input_file)
    return tasks[task_nr]