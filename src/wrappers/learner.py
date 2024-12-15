from typing import Any, Optional, List
from stable_baselines3.common.evaluation import evaluate_policy
        
def sb3_learner(agent, callback: Optional[List]= None, **kwargs) -> Any: 
    if callback:
        callback = list(callback)
       
    agent.learn(callback=callback, **kwargs)
    mean_reward, _ = evaluate_policy(model=agent, env=agent.get_env())
    
    return mean_reward