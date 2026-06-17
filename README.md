# COMP2025_memory-augmented_q-learning

A comparative study of short-term and long-term memory mechanisms in tabular Q-learning.

This project is developed for the **COMP2050 Artificial Intelligence Programming Project**.

---

## Project Overview

Standard tabular Q-learning learns action values from repeated interaction with an environment. However, it does not explicitly remember recently visited states or successful trajectories from previous episodes.

This project investigates whether adding simple memory mechanisms can affect:

* Learning speed
* Final policy performance
* Stability across random seeds
* Repeated-state behaviour
* Exploration and exploitation
* Robustness to larger and stochastic environments

Four agents are compared:

1. Standard Q-Learning
2. Short-Term Memory Q-Learning
3. Long-Term Memory Q-Learning
4. Combined Memory Q-Learning

The objective is not to assume that memory will always improve Q-learning. The experiments also examine cases where memory mechanisms reduce exploration, introduce bias, or perform worse than the baseline.

---

## Research Questions

This project investigates the following research questions:

* **RQ1:** Does short-term memory reduce repeated-state behaviour and improve learning efficiency?
* **RQ2:** Does long-term memory based on successful state-action trajectories improve the use of past experience?
* **RQ3:** Does combining short-term and long-term memory perform better than using either mechanism independently?
* **RQ4:** How do environment size and transition stochasticity affect the usefulness of memory?

---

## Algorithms

### 1. Standard Q-Learning

The baseline agent uses tabular Q-learning with epsilon-greedy exploration.

The Q-learning update rule is:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max Q(s', a') - Q(s, a)]
```

where:

* `s` is the current state
* `a` is the selected action
* `r` is the received reward
* `s'` is the next state
* `alpha` is the learning rate
* `gamma` is the discount factor

With probability `epsilon`, the agent selects a random action. Otherwise, it selects the action with the highest Q-value.

---

### 2. Short-Term Memory Q-Learning

The short-term memory agent stores recently visited states within the current episode.

Example:

```python
from collections import deque

recent_states = deque(maxlen=10)
```

When the agent moves into a recently visited state, a configurable repetition penalty is applied:

```text
adjusted_reward = environment_reward - repetition_penalty
```

The purpose of short-term memory is to discourage repetitive behaviour such as:

```text
A -> B -> A -> B -> A
```

Main short-term memory parameters:

* Memory size
* Repetition penalty

The short-term memory is reset at the beginning of every episode.

---

### 3. Long-Term Memory Q-Learning

The long-term memory agent stores state-action pairs that appeared in successful trajectories.

For example, after a successful episode:

```text
state 0, action RIGHT
state 1, action DOWN
state 5, action DOWN
state 9, action RIGHT
```

The corresponding state-action memory values are increased.

During action selection, the agent combines the learned Q-value with the long-term memory bonus:

```text
action_score = Q(state, action) + memory_strength * memory_bonus
```

This allows actions associated with previous successful trajectories to receive additional preference.

Main long-term memory parameters:

* Memory strength
* Credit assignment strategy
* Optional memory decay

Unlike short-term memory, long-term memory is preserved across episodes.

---

### 4. Combined Memory Q-Learning

The combined agent uses both memory mechanisms.

Its action score can be represented as:

```text
action_score =
    Q-value
    + long-term memory bonus
    - short-term repetition penalty
```

The three components have different purposes:

* The Q-table represents knowledge learned through temporal-difference updates.
* Short-term memory discourages repeated behaviour within the current episode.
* Long-term memory encourages state-action pairs associated with previous success.

---

## Environments

The experiments use environments from the Gymnasium library and an optional custom grid-world map.

| Environment |   Size | Transition Type             | Purpose                    |
| ----------- | -----: | --------------------------- | -------------------------- |
| FrozenLake  |    4x4 | Deterministic               | Baseline validation        |
| FrozenLake  |    8x8 | Deterministic               | Scalability evaluation     |
| FrozenLake  |    8x8 | Stochastic                  | Robustness evaluation      |
| Custom Maze | Custom | Deterministic or stochastic | Loop and dead-end analysis |

In deterministic FrozenLake:

```python
is_slippery = False
```

In stochastic FrozenLake:

```python
is_slippery = True
```

The custom map is designed to contain additional loops, misleading paths, or dead ends so that the effects of memory can be observed more clearly.

---

## Project Structure

```text
memory_augmented_qlearning/
│
├── README.md
├── requirements.txt
├── run_experiments.py
├── run_evaluation.py
│
├── configs/
│   ├── environments.yaml
│   └── experiments.yaml
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── qlearning.py
│   │   ├── stm_qlearning.py
│   │   ├── ltm_qlearning.py
│   │   └── combined_qlearning.py
│   │
│   ├── environments/
│   │   └── custom_maps.py
│   │
│   ├── memory/
│   │   ├── short_term.py
│   │   └── long_term.py
│   │
│   ├── training.py
│   ├── evaluation.py
│   ├── metrics.py
│   └── utils.py
│
├── experiments/
│   ├── hyperparameter_search.py
│   └── final_experiments.py
│
├── analysis/
│   ├── statistical_tests.py
│   └── generate_figures.py
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
├── report/
│   └── report.tex
│
└── statement/
    └── statement.tex
```

The final structure may be simplified depending on the implementation progress.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd memory_augmented_qlearning
```

Replace `<repository-url>` with the URL of this repository.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Suggested `requirements.txt` content:

```text
gymnasium
numpy
pandas
matplotlib
scipy
pyyaml
tqdm
```

---

## Running the Project

### Run a baseline experiment

```bash
python run_experiments.py --agent qlearning --environment frozenlake_4x4
```

### Run the short-term memory agent

```bash
python run_experiments.py --agent stm --environment frozenlake_4x4
```

### Run the long-term memory agent

```bash
python run_experiments.py --agent ltm --environment frozenlake_4x4
```

### Run the combined memory agent

```bash
python run_experiments.py --agent combined --environment frozenlake_4x4
```

### Run all final experiments

```bash
python experiments/final_experiments.py
```

### Generate figures

```bash
python analysis/generate_figures.py
```

### Run statistical analysis

```bash
python analysis/statistical_tests.py
```

These commands may be updated as the implementation develops.

---

## Experimental Protocol

All four algorithms are evaluated under the same experimental conditions.

Recommended initial configuration:

```text
Training episodes: 10,000
Evaluation episodes: 500
Number of final seeds: 30

Learning rate: 0.1
Discount factor: 0.99

Initial epsilon: 1.0
Minimum epsilon: 0.05
```

During final evaluation, the learned policy is evaluated greedily:

```text
epsilon = 0
```

The following settings must remain identical across algorithms:

* Environment
* Number of training episodes
* Number of evaluation episodes
* Random seeds
* Learning rate
* Discount factor
* Epsilon schedule
* Maximum episode length
* Initial Q-table
* Terminal conditions

Only the memory mechanism should change between the agents.

---

## Metrics

### Primary Metrics

#### Final Success Rate

```text
successful evaluation episodes
-------------------------------
total evaluation episodes
```

#### Learning-Curve Area Under the Curve

The area under the learning curve measures overall learning efficiency rather than only final performance.

#### Success Rate at Training Checkpoints

Example checkpoints:

```text
500 episodes
1,000 episodes
2,000 episodes
5,000 episodes
10,000 episodes
```

---

### Secondary Metrics

The project also records:

* Average episodic return
* Average episode length
* Repeated-state rate
* Number of unique states visited
* Convergence speed
* Training time
* Policy stability

The repeated-state rate is defined as:

```text
transitions into recently visited states
----------------------------------------
total number of transitions
```

---

## Planned Experiments

### Experiment 1: Baseline Validation

Verify that standard Q-learning can learn a successful policy in deterministic FrozenLake 4x4.

This experiment is used to:

* Validate the Q-learning implementation
* Test reward logging
* Test policy evaluation
* Confirm reproducibility
* Identify implementation errors

---

### Experiment 2: Main Agent Comparison

Compare the four agents:

* Standard Q-learning
* Short-Term Memory Q-learning
* Long-Term Memory Q-learning
* Combined Memory Q-learning

The comparison focuses on:

* Final success rate
* Learning efficiency
* Repeated-state behaviour
* Episode length
* Stability across random seeds

---

### Experiment 3: Environment Difficulty

Evaluate all four agents on:

* FrozenLake 4x4 deterministic
* FrozenLake 8x8 deterministic
* FrozenLake 8x8 stochastic
* A custom map containing loops or dead ends

This experiment studies whether memory mechanisms generalize across different problem settings.

---

### Experiment 4: Hyperparameter Sensitivity

Initial short-term memory settings:

```text
Memory size: 5, 10, 20
Repetition penalty: 0.01, 0.05, 0.10
```

Initial long-term memory settings:

```text
Memory strength: 0.05, 0.10, 0.30
```

Hyperparameter experiments should initially use approximately 5 to 10 seeds.

After selecting reasonable configurations, the final comparison should use 30 seeds.

---

### Experiment 5: Ablation Study

The four agents form an ablation study:

| Method                     | Short-Term Memory | Long-Term Memory |
| -------------------------- | ----------------: | ---------------: |
| Standard Q-learning        |                No |               No |
| STM Q-learning             |               Yes |               No |
| LTM Q-learning             |                No |              Yes |
| Combined Memory Q-learning |               Yes |              Yes |

This design helps identify the individual contribution of each memory mechanism.

---

## Statistical Analysis

Each final experiment should be repeated using 30 random seeds.

The results should include:

* Mean
* Standard deviation
* 95% confidence interval
* Pairwise statistical comparisons
* Effect sizes

A possible statistical procedure is:

1. Use the same random seeds for all agents.
2. Compare agents using paired statistical tests.
3. Use the Wilcoxon signed-rank test when normality cannot be assumed.
4. Apply Holm correction when performing multiple comparisons.
5. Report effect sizes in addition to p-values.

Statistical significance should not be treated as the only evidence of improvement. The magnitude and consistency of the differences should also be discussed.

---

## Expected Outputs

The project should generate:

* Raw per-episode results
* Raw per-seed evaluation results
* Processed summary tables
* Learning curves
* Final success-rate plots
* Repeated-state plots
* Episode-length plots
* State-visitation heatmaps
* Hyperparameter sensitivity plots
* Statistical comparison tables

Example output directories:

```text
results/raw/
results/processed/
results/figures/
```

Each result record should contain information such as:

```text
algorithm
environment
seed
episode
reward
success
episode_length
repeated_state_count
epsilon
execution_time
```

Raw results should be saved before aggregation so that figures can be regenerated without rerunning the experiments.

---

## Reproducibility

Each experiment should record:

* Algorithm name
* Environment name
* Environment settings
* Random seed
* Number of episodes
* Learning rate
* Discount factor
* Epsilon schedule
* Memory parameters
* Training metrics
* Evaluation metrics
* Execution time

Random seeds should be set for:

* Python
* NumPy
* Gymnasium environment
* Action space

Example:

```python
import random
import numpy as np

random.seed(seed)
np.random.seed(seed)

state, info = env.reset(seed=seed)
env.action_space.seed(seed)
```

---

## Figures and Tables

Planned report figures include:

* **Figure 1:** Architecture of the memory-augmented Q-learning framework
* **Figure 2:** Training success rate across episodes
* **Figure 3:** Final success rate across environments
* **Figure 4:** Repeated-state rate
* **Figure 5:** State-visitation heatmaps
* **Figure 6:** Hyperparameter sensitivity analysis

Planned report tables include:

* **Table 1:** Environment configurations
* **Table 2:** Algorithm hyperparameters
* **Table 3:** Final results as mean and standard deviation
* **Table 4:** Statistical comparisons and effect sizes

Learning curves should be averaged across multiple seeds and displayed with uncertainty bands.

---

## Development Plan

### Phase 1: Baseline

* Implement standard tabular Q-learning
* Test FrozenLake 4x4 deterministic
* Verify that the agent can reach the goal
* Save training and evaluation results
* Confirm reproducibility

### Phase 2: Experimental Pipeline

* Create reusable training functions
* Create separate evaluation functions
* Add CSV result logging
* Add multiple-seed execution
* Add metric calculation

### Phase 3: Short-Term Memory

* Store recently visited states
* Detect state repetition
* Apply the repetition penalty
* Record repeated-state metrics
* Compare against the baseline

### Phase 4: Long-Term Memory

* Store successful state-action trajectories
* Implement memory normalization
* Add the memory bonus during action selection
* Test whether memory reduces exploration too strongly

### Phase 5: Combined Agent

* Integrate short-term and long-term memory
* Verify that both components operate independently
* Compare the combined agent with individual memory agents

### Phase 6: Pilot Experiments

* Run approximately 5 seeds
* Inspect learning curves
* Detect implementation errors
* Select reasonable hyperparameter ranges
* Confirm that training episodes are sufficient

### Phase 7: Final Experiments

* Run 30 seeds
* Evaluate all algorithms
* Evaluate all final environments
* Save raw results
* Generate summary tables

### Phase 8: Analysis and Reporting

* Generate figures
* Perform statistical tests
* Interpret successful and unsuccessful results
* Write the final report
* Write the implementation statement

---

## Current Development Checklist

* [ ] Create the repository structure
* [ ] Create the Python virtual environment
* [ ] Add `requirements.txt`
* [ ] Implement standard Q-learning
* [ ] Validate FrozenLake 4x4 deterministic
* [ ] Add experiment logging
* [ ] Add policy evaluation
* [ ] Add fixed random seeds
* [ ] Add repeated-state tracking
* [ ] Implement the short-term memory agent
* [ ] Implement the long-term memory agent
* [ ] Implement the combined memory agent
* [ ] Add FrozenLake 8x8 deterministic
* [ ] Add FrozenLake 8x8 stochastic
* [ ] Create a custom loop or dead-end map
* [ ] Run pilot experiments
* [ ] Select final hyperparameters
* [ ] Run the final 30-seed experiments
* [ ] Perform statistical analysis
* [ ] Generate figures and tables
* [ ] Write `report.pdf`
* [ ] Write `statement.pdf`
* [ ] Package the source code as `code.zip`

---

## Possible Limitations

The project investigates several possible limitations:

* Short-term penalties may discourage revisits that are necessary to reach the goal.
* Long-term memory may create premature exploitation.
* A successful trajectory may have occurred because of chance in a stochastic environment.
* Memory mechanisms may overfit to a particular map.
* Strong memory bonuses may reduce exploration.
* FrozenLake uses sparse rewards, so average reward and success rate may provide similar information.
* Tabular Q-learning does not scale efficiently to large or continuous state spaces.

These limitations should be discussed even when the proposed methods outperform the baseline.

---

## Academic Integrity and AI Usage

The initial standard Q-learning baseline may be developed with AI assistance.

The main independently designed and implemented components are intended to include:

* Short-term memory mechanism
* Long-term state-action memory
* Combined memory agent
* Custom environment maps
* Repeated-state metrics
* Experimental pipeline
* Hyperparameter analysis
* Statistical analysis
* Interpretation of results

All externally sourced, AI-generated, and independently implemented code must be identified accurately through source-code comments and described in `statement.pdf`.

Example comments:

```python
# AI-assisted initial implementation of standard Q-learning.
```

```python
# Independently designed and implemented short-term memory mechanism.
```

The final declaration must reflect the actual development process.

---

## Final Submission

The final submission must include:

```text
report.pdf
code.zip
statement.pdf
```

Before submission, verify that:

* All figures have numbers and captions.
* All tables have numbers and captions.
* All external sources are referenced.
* AI-assisted code is declared.
* Independently implemented sections are identified.
* Experimental results can be reproduced.
* Unsuccessful results are reported honestly.
* The code runs in a clean Python environment.

---

## Author

**Name:** Do Phuong An
**Student ID:** V202401391
**Course:** COMP2050 Artificial Intelligence
**Project:** Programming Project
**Academic Year:** 2025–2026
