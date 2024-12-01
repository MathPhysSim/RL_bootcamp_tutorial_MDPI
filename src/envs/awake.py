import math
import random
from enum import Enum
from typing import Optional, Dict, Any

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from cpymad.madx import Madx
from gymnasium import spaces
from matplotlib import pyplot as plt


class Plane(Enum):
    horizontal = 0
    vertical = 1




class DynamicsHelper:
    # init
    def __init__(self):
        self.twiss = self._generate_optics()
        self.response_scale = 0.5
        self.twiss_bpms = self.twiss[self.twiss["keyword"] == "monitor"]
        self.twiss_correctors = self.twiss[self.twiss["keyword"] == "kicker"]

    def _calculate_response(self, bpmsTwiss, correctorsTwiss, plane):
        bpms = bpmsTwiss.index.values.tolist()
        correctors = correctorsTwiss.index.values.tolist()
        bpms.pop(0)
        correctors.pop(-1)
        rmatrix = np.zeros((len(bpms), len(correctors)))
        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if plane == Plane.horizontal:
                    bpm_beta = bpmsTwiss.betx[bpm]
                    corrector_beta = correctorsTwiss.betx[corrector]
                    bpm_mu = bpmsTwiss.mux[bpm]
                    corrector_mu = correctorsTwiss.mux[corrector]
                else:
                    bpm_beta = bpmsTwiss.bety[bpm]
                    corrector_beta = correctorsTwiss.bety[corrector]
                    bpm_mu = bpmsTwiss.muy[bpm]
                    corrector_mu = correctorsTwiss.muy[corrector]

                if bpm_mu > corrector_mu:
                    rmatrix[i][j] = (
                            math.sqrt(bpm_beta * corrector_beta)
                            * math.sin((bpm_mu - corrector_mu) * 2.0 * math.pi)
                            * self.response_scale
                    )
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix

    def generate_optics(self, randomize=True):
        twiss = self._generate_optics(randomize)
        twiss_bpms = twiss[twiss["keyword"] == "monitor"]
        twiss_correctors = twiss[twiss["keyword"] == "kicker"]
        responseH = self._calculate_response(
            twiss_bpms, twiss_correctors, Plane.horizontal
        )
        responseV = self._calculate_response(
            twiss_bpms, twiss_correctors, Plane.vertical
        )
        return responseH, responseV

    def recalculate_response(self):
        responseH = self._calculate_response(self.twiss_bpms, self.twiss_correctors, Plane.horizontal)
        responseV = self._calculate_response(self.twiss_bpms, self.twiss_correctors, Plane.vertical)
        return responseH, responseV

    def _generate_optics(self, randomize=False):
        OPTIONS = ["WARN"]  # ['ECHO', 'WARN', 'INFO', 'DEBUG', 'TWISS_PRINT']
        MADX_OUT = [f"option, -{ele};" for ele in OPTIONS]
        madx = Madx(stdout=False)
        madx.input("\n".join(MADX_OUT))
        tt43_ini = "environment/electron_design.mad"
        madx.call(file=tt43_ini, chdir=True)
        madx.use(sequence="tt43", range="#s/plasma_merge")
        quads = {}
        variation_range = (0.75, 1.25)
        if randomize:
            for ele, value in dict(madx.globals).items():
                if "kq" in ele:
                    # quads[ele] = value * 0.8
                    quads[ele] = value * np.random.uniform(variation_range[0], variation_range[1], size=None)
                    # pass
        madx.globals.update(quads)
        madx.input(
            "initbeta0:beta0,BETX=5,ALFX=0,DX=0,DPX=0,BETY=5,ALFY=0,DY=0.0,DPY=0.0,x=0,px=0,y=0,py=0;"
        )
        twiss_cpymad = madx.twiss(beta0="initbeta0").dframe()

        return twiss_cpymad

    def sample_tasks(self, num_tasks):
        # Generate goals using list comprehension for more concise code
        goals = [self.generate_optics() for _ in range(num_tasks)]

        # Create tasks with goals and corresponding IDs using list comprehension
        tasks = [{"goal": goal, "id": idx} for idx, goal in enumerate(goals)]
        return tasks

    def get_origin_task(self, idx=0):
        # Generate goals using list comprehension for more concise code
        goal = self.generate_optics(randomize=False)
        # Create tasks with goals and corresponding IDs using list comprehension
        task = {"goal": goal, "id": idx}
        return task


class AwakeSteering(gym.Env):
    """
    Gym environment for beam steering using reinforcement learning.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        boundary_conditions: bool = False,
        init_scaling: float = 1.0,
        plane: Any = Plane.horizontal,
        seed: Optional[int] = None,
        twiss: Optional[Any] = None,        # FIXXME: doesn't seem to be used by the AwakeSteering class
        task: Optional[Any] = None,         
        train: bool = False,                # FIXXME: doesn't seem to be used by the AwakeSteering class
        #**kwargs,
        ):
        """
        Initialize the AwakeSteering environment.

        Args:
            twiss: Optional Twiss parameters.
            task: Task dictionary containing 'goal' and 'id'.
            train: Training mode flag.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.__version__ = "1.0"
        
        
        
        # FIXXME
        # Maximum number of steps should be defined by the registry (max_episode_steps).
        # This wraps environment with a TimeLimit Wrapper 
        # self.MAX_TIME = kwargs.get("MAX_TIME", 100)
        
        
        # FIXXME
        # Initializer arguments should be defined in the signature of the initializer.
        # Taking the kwargs dictionary instead is a bit of a abuse complicates the usage if the
        # class as it isn't appearent out of the initializer signature 
        #self.boundary_conditions = kwargs.get("boundary_conditions", False)
        self.boundary_conditions = boundary_conditions
        
        self.state_scale = 1.0

        # FIXXME
        # init_scaling has been defined in the wrapper class only.
        # Consequently, the envirionment doesn't work without a wrapper which huge nogo.
        self.init_scaling = init_scaling


        self.threshold = -0.1  # Corresponds to 1 mm scaled.

        self.current_episode = -1
        self.current_steps = 0

        # FIXXME
        # Initializer arguments should be defined in the signature of the initializer.
        # Taking the kwargs dictionary instead is a bit of a abuse complicates the usage if the
        # class as it isn't appearent out of the initializer signature 
        self.seed(seed)
        self.maml_helper = DynamicsHelper()
        
            
        # FIXXME
        # Initializer arguments should be defined in the signature of the initializer.
        # Taking the kwargs dictionary instead is a bit of a abuse complicates the usage if the
        # class as it isn't appearent out of the initializer signature 
        # self.plane = kwargs.get("plane", Plane.horizontal)
        self.plane = plane

        if task is None:
            task = self.maml_helper.get_origin_task()
        self.reset_task(task)

        self.setup_dimensions()


        # FIXXE
        # self.verification_tasks_loc is not used anywhere in the code and is thus obsolete
        # self.verification_tasks_loc = kwargs.get("verification_tasks_loc", None)


        # FIXXME
        # self.noise_setting is only used in the DoF wrapper class. If a property is only required
        # in the wrapper class than it should be defined there
        # self.noise_setting = kwargs.get('noise_setting', False)

    def setup_dimensions(self):
        """
        Set up the dimensions of the action and observation spaces based on the response matrices.
        """
        num_bpms = len(self.maml_helper.twiss_bpms) - 1
        num_correctors = len(self.maml_helper.twiss_correctors) - 1

        # Define action and observation space limits
        self.high_action = np.ones(num_correctors, dtype=np.float32)
        self.low_action = -self.high_action

        self.high_observation = np.ones(num_bpms, dtype=np.float32) * self.state_scale
        self.low_observation = -self.high_observation

        self.action_space = spaces.Box(
            low=self.low_action, high=self.high_action, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observation, high=self.high_observation, dtype=np.float32
        )

        # Set the response matrix based on the plane
        if self.plane == Plane.horizontal:
            self.rmatrix = self.responseH
        else:
            self.rmatrix = self.responseV

        # Use pseudo-inverse for numerical stability
        self.rmatrix_inverse = np.linalg.pinv(self.rmatrix)

    def step(self, action: np.ndarray):
        """
        Execute one time step within the environment.

        Args:
            action: The action to be taken.

        Returns:
            observation: The agent's observation of the current environment.
            reward: Amount of reward returned after previous action.
            done: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional information about the environment.
        """
        delta_kicks = np.clip(action, self.low_action, self.high_action)
        self.state += self.rmatrix.dot(delta_kicks)

        self.state = np.clip(self.state, self.low_observation, self.high_observation)
        return_state = self.state.copy()

        reward = self._get_reward(return_state)

        self.current_steps += 1
        done = reward > self.threshold
        
        # FIXXME
        # Let TimeLimit Wrappers cut the epispde instead
        truncated = False
        # truncated = self.current_steps >= self.MAX_TIME
        
        
        # FIXXME 
        # It's not a good practice to set done at the end of the episode even
        # though the criteria has not been met.
        # if truncated:
        #     done = True

        # Find all indices where the absolute value of return_state exceeds or equals 1
        violations = np.argwhere(np.abs(return_state) >= 1).flatten()
        if violations.size > 0:
            # Violations detected
            # Take the index of the first violation
            violation_index = violations[0]
            # Set return_state from the point of violation onwards
            # Assign the sign (+1 or -1) of the violating element to the remaining elements
            return_state[violation_index:] = np.sign(return_state[violation_index])
            # Compute a penalty reward based on the modified return_state
            reward = self._get_reward(return_state)
        info = {"task": self._id, 'time': self.current_steps}
        return return_state, reward, done, truncated, info

    def _get_reward(self, observation: np.ndarray) -> float:
        """
        Compute the reward based on the current observation.

        Args:
            observation: The current state observation.

        Returns:
            The computed reward.
        """
        # Negative Euclidean norm (L2 norm) as a penalty
        return -np.sqrt(np.mean(np.square(observation)))

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """
        Reset the environment to an initial state.

        Args:
            seed: Seed for randomness.
            options: Additional options for reset.

        Returns:
            observation: The initial observation.
            info: Additional information.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
            self.observation_space.seed(seed)
        self.is_finalized = False
        self.current_steps = 0
        
        # FIXXME 
        # It doesn't really make much sense for the environment to track wich episode it's in.
        # Suggest to remvove.
        self.current_episode += 1
        
        
        self.state = np.clip(self.observation_space.sample(), -1, 1)*self.init_scaling
        return_state = self.state.copy()
        return return_state, {}

    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment's random number generator(s).

        Args:
            seed: The seed value.

        Returns:
            A list containing the seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def sample_tasks(self, num_tasks: int):
        """
        Sample a list of tasks for meta-learning.

        Args:
            num_tasks: The number of tasks to sample.

        Returns:
            A list of task dictionaries.
        """
        tasks = self.maml_helper.sample_tasks(num_tasks)
        return tasks

    def get_origin_task(self, idx: int = 0):
        """
        Get the original task with default optics.

        Args:
            idx: Task ID.

        Returns:
            The original task dictionary.
        """
        task = self.maml_helper.get_origin_task(idx=idx)
        return task

    def reset_task(self, task: Dict[str, Any]):
        """
        Reset the environment with a new task.

        Args:
            task: The task dictionary containing 'goal' and 'id'.
        """
        self._task = task
        self._goal = task["goal"]
        self._id = task["id"]

        self.responseH = self._goal[0]
        self.responseV = self._goal[1]

    def get_task(self):
        """
        Get the current task.

        Returns:
            The current task dictionary.
        """
        return self._task


class DoFAwakeSteering(AwakeSteering):
    def __init__(
        self,
        DoF: int,
        threshold: float = -0.1,
        action_scale: float = 1.0,
        penalty_scaling: float = 1.0,
        noise_setting: Optional[dict] = None,
        *args,
        **kwargs,
        ):
        
        super().__init__(*args, **kwargs)
        
        self.DoF = DoF
        self.threshold = threshold
        self.action_scale = action_scale
        self.penalty_scaling = penalty_scaling
        
        if noise_setting is not None:
            self.noise_sigma = noise_setting['std_noise']

        # Modify the action and observation spaces
        self._full_action_space = self.action_space
        self.action_space = spaces.Box(
            low=self.action_space.low[:DoF],
            high=self.action_space.high[:DoF],
            dtype=self.action_space.dtype,
        )
        self._full_observation_space = self.observation_space
        self.observation_space = spaces.Box(
            low=self.observation_space.low[:DoF],
            high=self.observation_space.high[:DoF],
            dtype=self.observation_space.dtype,
        )
        
    def swap_spaces(self):
        observation_space = self.observation_space
        action_space = self.action_space
        self.observation_space = self._full_observation_space
        self.action_space = self._full_action_space
        return action_space, observation_space


    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """
        Reset the environment and return the initial observation limited to the specified DoF.

        Args:
            seed: Optional seed for the environment.
            options: Optional dictionary of options.

        Returns:
            observation: The initial observation limited to the specified DoF.
            info: Additional information from the environment.
        """
        
        # Swap observation and action spaces as the base env depends on it
        observation_space, action_space = self.swap_spaces()
        observation, info = super().reset(seed=seed, options=options)
        self.observation_space = observation_space
        self.action_space = action_space
        
        
        observation = observation[: self.DoF]
        return observation, info
    

    def step(self, action: np.ndarray):
        """
        Step the environment with the given action, limited to the specified DoF.

        Args:
            action: The action to take, limited to the DoF.

        Returns:
            observation: The observation after the action, limited to the DoF.
            reward: The reward after the action.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated.
            info: Additional information from the environment.
        """

        observation_space, action_space = self.swap_spaces()
        
        # Initialize a zero-filled array for the full action space
        full_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        full_action[: self.DoF] = action  # Set the first 'DoF' elements with the provided action

        # Execute the action in the environment
        observation, reward, terminated, truncated, info = super().step(full_action)

        self.observation_space = observation_space
        self.action_space = action_space
        
        # Generate observation noise
        if hasattr(self, 'noise_sigma') and self.noise_sigma is not None:
            observation_noise = np.zeros_like(observation)  # Ensure noise shape matches observation
            # Apply Gaussian noise to the first 'DoF' elements of the observation
            noise = np.random.randn(self.DoF) * self.noise_sigma
            observation_noise[:self.DoF] = noise

            # Add noise to the observation
            observation += observation_noise
            observation = np.clip(observation,-1, 1)

        # Focus only on the degrees of freedom for observations
        observation = observation[: self.DoF]

        # Update the reward based on the current observation 
        reward = self._get_reward(observation)

        # Check for termination based on the reward threshold
        terminated = terminated or reward >= self.threshold

        # Check for any violations where the absolute values in observations exceed 1
        violations = np.where(np.abs(observation) >= 1)[0]
        if violations.size > 0:
            # Modify observation from the first violation onward
            first_violation = violations[0]
            observation[first_violation:] = np.sign(observation[first_violation])

            # Recalculate reward after modification
            reward *= self.penalty_scaling

            # Terminate if boundary conditions are set
            terminated = self.boundary_conditions

        return observation, reward, terminated, truncated, info
    
    

    # Optional function, not currently used
    def pot_function(self, x: np.ndarray, k: float = 1000, x0: float = 1) -> np.ndarray:
        """
        Compute a potential function using a modified sigmoid to handle deviations.
        The output scales transformations symmetrically for positive and negative values of x.

        Args:
            x: Input array.
            k: Steepness of the sigmoid.
            x0: Center point of the sigmoid.

        Returns:
            Transformed array with values scaled between 1 and 11.
        """
        # Precompute the exponential terms to use them efficiently.
        exp_pos = np.exp(k * (x - x0))
        exp_neg = np.exp(k * (-x - x0))

        # Calculate the transformation symmetrically for both deviations
        result = (1 - 1 / (1 + exp_pos)) + (1 - 1 / (1 + exp_neg))

        # Scale and shift the output between 1 and 11
        return 1 + 10 * result
    
    
    
    
    
    
    
    

class DoFWrapper(gym.Wrapper):
    """
    Gym Wrapper to limit the environment to a subset of degrees of freedom (DoF).
    This wrapper modifies the action and observation spaces to include only the first 'DoF' elements.
    It also modifies the reward and termination conditions based on the subset of observations.
    """

    def __init__(self, env: gym.Env, DoF: int, **kwargs):
        """
        Initialize the DoFWrapper.

        Args:
            env: The original Gym environment.
            DoF: The number of degrees of freedom to limit the action and observation spaces to.
            **kwargs: Additional keyword arguments, such as 'threshold' and 'boundary_conditions'.
        """
        super().__init__(env)
        self.DoF = DoF
        self.threshold = kwargs.get("threshold", -0.1)
        self.boundary_conditions = kwargs.get("boundary_conditions", False)
        
        
        # FIXXME
        # A wrapper class should never modify members of the wrapped object!
        # If this env.init_scaling should be parameterized, do it in the enviornment class
        # self.env.init_scaling = kwargs.get("init_scaling", 1.0)
        
        
        self.action_scale = kwargs.get("action_scale", 1.0)
        self.penalty_scaling = kwargs.get("penalty_scaling", 1.0)
        noise_setting = kwargs.get("noise_settings", None)
        if noise_setting is not None:
            self.noise_sigma = noise_setting['std_noise']

        # Modify the action and observation spaces
        self.action_space = spaces.Box(
            low=env.action_space.low[:DoF],
            high=env.action_space.high[:DoF],
            dtype=env.action_space.dtype,
        )
        self.observation_space = spaces.Box(
            low=env.observation_space.low[:DoF],
            high=env.observation_space.high[:DoF],
            dtype=env.observation_space.dtype,
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """
        Reset the environment and return the initial observation limited to the specified DoF.

        Args:
            seed: Optional seed for the environment.
            options: Optional dictionary of options.

        Returns:
            observation: The initial observation limited to the specified DoF.
            info: Additional information from the environment.
        """
        observation, info = self.env.reset(seed=seed, options=options)
        observation = observation[: self.DoF]
        return observation, info

    def step(self, action: np.ndarray):
        """
        Step the environment with the given action, limited to the specified DoF.

        Args:
            action: The action to take, limited to the DoF.

        Returns:
            observation: The observation after the action, limited to the DoF.
            reward: The reward after the action.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated.
            info: Additional information from the environment.
        """

        action = action*self.action_scale
        # Initialize a zero-filled array for the full action space
        full_action = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        full_action[: self.DoF] = action  # Set the first 'DoF' elements with the provided action

        # Execute the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(full_action)

        # Generate observation noise
        if hasattr(self, 'noise_sigma') and self.noise_sigma is not None:
            observation_noise = np.zeros_like(observation)  # Ensure noise shape matches observation
            # Apply Gaussian noise to the first 'DoF' elements of the observation
            noise = np.random.randn(self.DoF) * self.noise_sigma
            observation_noise[:self.DoF] = noise

            # Add noise to the observation
            observation += observation_noise
            observation = np.clip(observation,-1, 1)

        # Focus only on the degrees of freedom for observations
        observation = observation[: self.DoF]

        # Update the reward based on the current observation
        
        
        # FIXXME
        # Wrapper methods should not call private methods. Besided we got the reward already 
        # from the step method before. This is pretty much a design flaw and shows that 
        # DoFWrapper is not suited to wrap the environment as it is deals rather with a subset
        # of functionality and should thus be an inherted class from AwakeSteering.
        # As a quick fix we copy the code from AwakeSteering but the inheritance concept need to
        # be redefined!
        # reward = self.env._get_reward(observation)
        reward = -np.sqrt(np.mean(np.square(observation)))

        # Check for termination based on the reward threshold

        # FIXXME 
        # More elegant and easier to read ...
        # if reward >= self.threshold:
        #     terminated = True
        terminated = reward >= self.threshold

        # Check for any violations where the absolute values in observations exceed 1
        violations = np.where(np.abs(observation) >= 1)[0]
        if violations.size > 0:
            # Modify observation from the first violation onward
            first_violation = violations[0]
            observation[first_violation:] = np.sign(observation[first_violation])

            # Recalculate reward after modification
            
            # FIXXME 
            # Why to call the env reward function again here? We have already computed the 
            # reward above. Calling private methods of the env member crashes whe gym envs
            # are wrapped.
            #reward = self.env._get_reward(observation) * self.penalty_scaling
            reward *= self.penalty_scaling

            # Terminate if boundary conditions are set
            terminated = self.boundary_conditions

        return observation, reward, terminated, truncated, info

    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment's random number generator(s).

        Args:
            seed: The seed value.

        Returns:
            A list containing the seed.
        """
        return self.env.seed(seed)

    # Optional function, not currently used
    def pot_function(self, x: np.ndarray, k: float = 1000, x0: float = 1) -> np.ndarray:
        """
        Compute a potential function using a modified sigmoid to handle deviations.
        The output scales transformations symmetrically for positive and negative values of x.

        Args:
            x: Input array.
            k: Steepness of the sigmoid.
            x0: Center point of the sigmoid.

        Returns:
            Transformed array with values scaled between 1 and 11.
        """
        # Precompute the exponential terms to use them efficiently.
        exp_pos = np.exp(k * (x - x0))
        exp_neg = np.exp(k * (-x - x0))

        # Calculate the transformation symmetrically for both deviations
        result = (1 - 1 / (1 + exp_pos)) + (1 - 1 / (1 + exp_neg))

        # Scale and shift the output between 1 and 11
        return 1 + 10 * result



class Awake_Benchmarking_Wrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.invV = None
        self.invH = None
        self.optimal_rewards = None
        self.optimal_actions = None
        self.optimal_states = None

    def reset(self, **kwargs):
        #     print('reset', self.current_steps, self._get_reward(return_value))
        return_initial_state, _ = self.env.reset(**kwargs)

        self.invH, self.invV = np.linalg.inv(self.env.responseH / 100) / 100, np.linalg.inv(
            self.env.responseV / 100) / 100
        self.optimal_states, self.optimal_actions, self.optimal_rewards = self._get_optimal_trajectory(
            return_initial_state)
        return return_initial_state, {}

    def policy_optimal(self, state):
        # invrmatrix = self.invH if self.plane == 'horizontal' else self.invV
        invrmatrix = self.invH
        action = -invrmatrix.dot(state * self.env.state_scale)
        # action = np.clip(action, -1, 1)
        action_abs_max = max(abs(action))
        if action_abs_max > 1:
            action /= action_abs_max
        return action

    def get_k_for_state(self, state):
        # invrmatrix = self.invH if self.plane == 'horizontal' else self.invV
        invrmatrix = self.invH
        k = invrmatrix.dot(state * self.env.unwrapped.state_scale) * self.env.unwrapped.action_scale
        return k

    def get_optimal_trajectory(self):
        return self.optimal_states, self.optimal_actions, self.optimal_rewards

    def _get_optimal_trajectory(self, init_state):
        max_iterations = 25
        states = [init_state]
        actions = []
        # Todo: reward scaling
        rewards = [self.env._get_reward(init_state) * self.env.reward_scale]

        self.env.kicks_0_opt = self.env.kicks_0.copy()
        self.env.kicks_0 = self.get_k_for_state(init_state)
        self.env.is_finalized = False

        for i in range(max_iterations):
            action = self.policy_optimal(states[i])
            actions.append(action)
            state, reward, is_finalized, _, _ = self.env.step(action)

            states.append(state)
            rewards.append(reward)

            if is_finalized:
                break

        if i < max_iterations - 1:
            # nan_state = [np.nan] * self.env.observation_space.shape[-1]
            # nan_action = [np.nan] * self.env.action_space.shape[-1]
            # states[i + 2:] = [nan_state] * (max_iterations - i - 1)
            # actions[i + 1:] = [nan_action] * (max_iterations - i - 1)
            states.append([np.nan] * self.env.observation_space.shape[-1])
            actions.append([np.nan] * self.env.action_space.shape[-1])
            rewards.append(np.nan)

        self.env.kicks_0 = self.env.kicks_0_opt.copy()
        self.env.is_finalized = False
        return states, actions, rewards

    def draw_optimal_trajectories(self, init_state, nr_trajectories=5):
        states_frames, actions_frames, rewards_frames = [], [], []
        len_mean = []

        for i in range(nr_trajectories):
            states, actions, rewards = self._get_optimal_trajectory(init_state)
            states_frames.append(pd.DataFrame(states))
            actions_frames.append(pd.DataFrame(actions))
            rewards_frames.append(pd.DataFrame(rewards))
            # actions end with np.nan to find episode ends
            len_mean.append(len(actions) - 1)

        mean_length = np.mean(len_mean)
        # print(mean_length)

        states_df = pd.concat(states_frames, ignore_index=True)
        actions_df = pd.concat(actions_frames, ignore_index=True)
        rewards_df = pd.concat(rewards_frames, ignore_index=True)

        fig, axs = plt.subplots(3, figsize=(10, 10))
        for df, ax in zip([states_df, rewards_df, actions_df], axs):
            df.plot(ax=ax)

        plt.suptitle(f'Mean Length of Episodes: {mean_length}')
        plt.tight_layout()
        plt.show()
        plt.pause(1)