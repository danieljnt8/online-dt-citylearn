from tqdm.auto import trange, tqdm
import numpy as np
def segment_v2(states, actions, rewards, terminals):
    assert len(states) == len(terminals)
    
    trajectories = []
    episode = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones":[]
    }

    for t in trange(len(terminals), desc="Segmenting"):
        episode["observations"].append(states[t])
        episode["actions"].append(actions[t])
        episode["rewards"].append(rewards[t])
        episode["dones"].append(terminals[t])

        if terminals[t]:
            # Convert lists to numpy arrays
            episode["observations"] = np.array(episode["observations"])
            episode["actions"] = np.array(episode["actions"])
            episode["rewards"] = np.array(episode["rewards"])
            episode["dones"] = np.array(episode["dones"])
            # Append the current episode to the trajectories list
            trajectories.append(episode)
            # Reset episode
            episode = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones":[]
            }

    # If there are any remaining observations, actions, and rewards in the current episode
    if episode["observations"]:
        episode["observations"] = np.array(episode["observations"])
        episode["actions"] = np.array(episode["actions"])
        episode["rewards"] = np.array(episode["rewards"])
        episode["dones"] = np.array(episode["dones"])
        trajectories.append(episode)
    
    trajectories_lens = [len(episode["observations"]) for episode in trajectories]

    return trajectories, trajectories_lens
