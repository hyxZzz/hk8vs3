# Reward and Action Diagnostics

The helper script [`utils/reward_diagnostics.py`](../utils/reward_diagnostics.py) summarises the training logs stored in `reward.txt` and `action.txt`.
Running it on the supplied data yields the following key observations:

* **Reward distribution** – Rewards span from −31.42 to 4.89 with a mean of −1.95. The heavy lower tail is evident from the 5th percentile dropping to −8.86 while the median sits at −0.32.
* **Action usage** – 10 actions account for the majority of selections, led by action 7 (18,022 steps, 63% positive rewards on those steps, mean reward −1.01) and action 9 (10,896 steps, 64% positive, mean reward −0.58). Every frequently used action still averages a negative reward, indicating consistently large penalties.
* **Problematic actions** – Actions 1, 21, 26, 28, 4, and 24 exhibit the worst mean rewards (−3.77 or lower) despite thousands of selections, confirming that sharp negative updates dominate the return distribution.

These metrics motivate the updated reward shaping in this change set: boosting positive credit when the closest threat distance opens up, re-scaling danger-driven bonuses, and stiffening launch penalties so that historically bad actions are discouraged while distance-making manoeuvres can overcome the background penalties that previously dominated the returns.
