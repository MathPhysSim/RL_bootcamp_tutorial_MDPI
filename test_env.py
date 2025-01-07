from src.envs.awake import AwakeSteering, DoFWrapper, DoFAwakeSteering

if __name__ == "__main__":
    seed = 1234
    
    env = AwakeSteering()
    obs, info = env.reset(seed=seed)
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action=action)
    print("#"*10, " AwakeSteering ", "#"*10)
    print("obs.shape: ", obs.shape)
    print("action.shape: ", action.shape)
    print("reward: ", reward)
    print("next_obs.shape: ", next_obs.shape)

    wrapper = DoFWrapper(
        env=env,
        DoF=6,                      # Number of degrees of freedom
        boundary_conditions=True,   # Whether to enforce boundary conditions on beam positions.
    	action_scale=1.0,           # Scaling factor applied to actions to control the magnitude of control inputs.
        penalty_scaling=100.0,      # Scaling factor applied to penalties when boundary conditions are violated.
        noise_setting = dict(std_noise=0.5),
    )
    obs, info = wrapper.reset(seed=seed)
    action = wrapper.action_space.sample()
    next_obs, reward, done, truncated, info = wrapper.step(action=action)
    print("#"*10, " DoFWrapper ", "#"*10)
    print("obs.shape: ", obs.shape)
    print("action.shape: ", action.shape)
    print("reward: ", reward)
    print("next_obs.shape: ", next_obs.shape)

    dof_env = DoFAwakeSteering(
        DoF=6,
        boundary_conditions=True,
        action_scale=1.0,
        penalty_scaling=100.0,
        noise_setting=dict(std_noise=0.5),
    )
    
    obs, info = dof_env.reset(seed=seed)
    action = dof_env.action_space.sample()
    next_obs, reward, done, truncated, info = dof_env.step(action=action)
    print("#"*10, " DoFAwakeSteering ", "#"*10)
    print("obs.shape: ", obs.shape)
    print("action.shape: ", action.shape)
    print("reward: ", reward)
    print("next_obs.shape: ", next_obs.shape)
