from typing import Any, List, Mapping, Tuple, Union,TypeVar,Any
from citylearn.citylearn import CityLearnEnv
import numpy as np
from citylearn.wrappers import *
from typing import Any, Mapping, List, Union
from citylearn.citylearn import CityLearnEnv
from citylearn.building import Building
import itertools


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

##only for citylearn 2022 
hour_index = [0,2,4]
index_commun = hour_index + list(range(6,23)) + list(range(27,31))
index_particular_rbc = list(range(23,27)) 
index_particular_sb = list(range(23,27)) + list(range(31,35)) + list(range(39,43)) + list(range(47,51))+ list(range(55,59))
index_all = index_commun + index_particular_sb

def action_space_to_dict(aspace):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    #building_info = env.buildings()
    #building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                #"building_info": building_info,
                "observation": observations}
    return obs_dict

class NormalizedObservationWrapperCustom(ObservationWrapper):
    """Wrapper for observations min-max and periodic normalization.
    
    Temporal observations including `hour`, `day_type` and `month` are periodically normalized using sine/cosine 
    transformations and then all observations are min-max normalized between 0 and 1.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv) -> None:
        super().__init__(env)
        self.env: CityLearnEnv
        self.dataset=[]

    @property
    def shared_observations_norm(self) -> List[str]:
        """Names of common observations across all buildings i.e. observations that have the same value irrespective of the building.
        
        Includes extra three observations added during cyclic transformation of :code:`hour`, :code:`day_type` and :code:`month`.
        """

        shared_observations = []
        periodic_observation_names = list(Building.get_periodic_observation_metadata().keys())

        for o in self.env.shared_observations:
            if o in periodic_observation_names:
                shared_observations += [f'{o}_cos', f'{o}_sin']
            
            else:
                shared_observations.append(o)

        return shared_observations
    
    def observation_names(self) -> List[List[str]]:
        """Names of returned observations.

        Includes extra three observations added during cyclic transformation of :code:`hour`, :code:`day_type` and :code:`month`.

        Notes
        -----
        If `central_agent` is True, a list of 1 sublist containing all building observation names is returned in the same order as `buildings`. 
        The `shared_observations` names are only included in the first building's observation names. If `central_agent` is False, a list of sublists 
        is returned where each sublist is a list of 1 building's observation names and the sublist in the same order as `buildings`.
        """

        if self.env.unwrapped.central_agent:
            observation_names = []

            for i, b in enumerate(self.env.buildings):
                for k, _ in b.observations(normalize=True, periodic_normalization=True).items():
                    if i == 0 or k not in self.shared_observations or k not in observation_names:
                        observation_names.append(k)
                    
                    else:
                        pass

            observation_names = [observation_names]
        
        else:
            observation_names = [list(b.observations(normalize=True, periodic_normalization=True).keys()) for b in self.env.buildings]

        return observation_names


    

    def get_observation_norm(self, observations: List[List[float]]) -> List[List[float]]:
        """Returns normalized observations."""

        if self.env.central_agent:
            norm_observations = []
            shared_observations = []

            for i, b in enumerate(self.env.buildings):
                for k, v in b.observations(normalize=True, periodic_normalization=True).items():
                    if i==0 or k not in self.shared_observations_norm or k not in shared_observations:
                        norm_observations.append(v)

                    else:
                        pass

                    if k in self.shared_observations_norm and k not in shared_observations:
                        shared_observations.append(k)
                    
                    else:
                        pass
            
            norm_observations = [norm_observations]

        else:
            norm_observations = [list(b.observations(normalize=True, periodic_normalization=True).values()) for b in self.env.buildings]
        
        return norm_observations
    
    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Resets the environment with kwargs."""
        obs ,_= self.env.reset(**kwargs)
        #print(obs)
        norm_obs = self.get_observation_norm(obs)
        observation_commun = [norm_obs[0][i] for i in index_commun]
        observation_particular = [[o[i] for i in index_particular_rbc] for o in norm_obs]
        observation_particular = list(itertools.chain(*observation_particular))

        observation = observation_commun + observation_particular
        
        self.current_obs = observation
        #self.dataset = []
        return observation,_
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Steps through the environment with action."""
        obs, reward, done, truncated,info = self.env.step(action)
        #print(obs)
        norm_obs = self.get_observation_norm(obs)
        #print(norm_obs)

        observation_commun = [norm_obs[0][i] for i in index_commun]
        observation_particular = [[o[i] for i in index_particular_rbc] for o in norm_obs]
        observation_particular = list(itertools.chain(*observation_particular))
        # we concatenate the observation
        observation = observation_commun + observation_particular

        
        
        self.dataset.append({
            "observations": self.current_obs,
            "next_observations": observation,  # Assuming next observation is same as current for simplicity
            "actions": action,
            "rewards": sum(reward),
            "dones": done,
            "info": info
        })
        
        self.current_obs = observation
        
        return observation, reward, done, truncated,info

class StableBaselines3WrapperCustom(Wrapper):
    """Wrapper for :code:`stable-baselines3` algorithms.

    Wraps `env` in :py:class:`citylearn.wrappers.StableBaselines3ObservationWrapper`,
    :py:class:`citylearn.wrappers.StableBaselines3ActionWrapper`
    and :py:class:`citylearn.wrappers.StableBaselines3RewardWrapper`.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv):
        env = StableBaselines3ActionWrapper(env)
        env = StableBaselines3RewardWrapper(env)
        env = StableBaselines3ObservationWrapper(env)
        super().__init__(env)
        self.env: CityLearnEnv
        self.dataset = []
        
    @property
    def observation_space(self) -> List[spaces.Box]:
        """Returns observation space for normalized observations."""
        assert self.env.central_agent == True
        low_limit = []
        high_limit = []

        if self.env.central_agent:
            shared_observations = []

            for i, b in enumerate(self.env.buildings):
                s = b.estimate_observation_space(normalize=True)
                o = b.observations(normalize=True, periodic_normalization=True)

                for k, lv, hv in zip(o, s.low, s.high):                    
                    if i == 0 or k not in self.shared_observations or k not in shared_observations:
                        low_limit.append(lv)
                        high_limit.append(hv)

                    else:
                        pass

                    if k in self.shared_observations and k not in shared_observations:
                        shared_observations.append(k)
                    
                    else:
                        pass
            
            observation_space = [spaces.Box(low=np.array(low_limit)[index_all], high=np.array(high_limit)[index_all], dtype=np.float32)]

        else:
            observation_space = [b.estimate_observation_space(normalize=True) for b in self.env.buildings]
        
        return observation_space[0]
    
    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Resets the environment with kwargs."""
        obs,_ = self.env.reset(**kwargs)
        self.current_obs = obs[index_all]
        #
        #self.dataset = []
        return obs[index_all],_
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Steps through the environment with action."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        
        self.dataset.append({
            "observations": self.current_obs,
            "next_observations": obs[index_all],  # Assuming next observation is same as current for simplicity
            "actions": action,
            "rewards": reward,
            "dones": done,
            "info": info
        })
        
        self.current_obs = obs[index_all]
        
        return  obs[index_all], reward, done, truncated, info