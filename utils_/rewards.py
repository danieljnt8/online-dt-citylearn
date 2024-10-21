from citylearn.reward_function import RewardFunction
from citylearn.citylearn import CityLearnEnv
from typing import List, Mapping, Tuple,Union
import numpy as np

class CustomReward(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        """

        super().__init__(env)
        self.reward_name = "firstCR"

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        reward_list = []

        for b in self.env_metadata.buildings:
            cost = b.net_electricity_consumption_cost[-1]
            battery_capacity = b.electrical_storage.capacity_history[0]
            battery_soc = b.electrical_storage.soc[-1]/battery_capacity
            penalty = -(1.0 + np.sign(cost)*battery_soc)
            reward = penalty*abs(cost)
            reward_list.append(reward)

        reward = [sum(reward_list)]

        return reward