Flow Matching Policy Training with SAC
======================================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-overview.png
   :align: center
   :width: 80%

   SAC-Flow overview.

Train a **Flow Matching** policy network with **SAC (Soft Actor-Critic)** in simulation or on a real robot. The method combines maximum-entropy reinforcement learning with generative flow matching models.

Paper: `SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling <https://arxiv.org/abs/2509.25756>`_

Overview
--------

Train a Flow Matching policy with SAC — in ManiSkill simulation or on a real Franka (peg insertion).

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      SAC · RLPD

   .. grid-item-card:: Models
      :text-align: center

      Flow Matching policy

   .. grid-item-card:: Environments / Data
      :text-align: center

      ManiSkill · Franka

   .. grid-item-card:: Training
      :text-align: center

      Sim & Real

| **You'll do:** install (sim or real) → pick a config → launch → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` (sim) or :doc:`franka` (real hardware).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 32 28 24

   * - Setting
     - Environment & task
     - Observation
     - Action
   * - Simulation
     - ManiSkill3 — ``PickCube-v1``
     - Joint angles + object state
     - 4-dim: 3D position + gripper
   * - Real world
     - Franka Panda + RealSense — peg insertion
     - RGB + proprioception
     - 6-dim end-effector pose (legacy) or 7-dim pose + gripper (BC fine-tuning)

How SAC-Flow Works
------------------

**Core Algorithm Components**

1.  **SAC (Soft Actor-Critic)**

    -   Learns Q-values through the Bellman equation and entropy regularization.

    -   Uses a **Flow Matching** network as the Actor policy.

    -   Learns a temperature parameter to balance exploration and exploitation.

2.  **Flow Matching Policy**

    -   **Velocity Network Parameterization**: Treats the K-step sampling of the flow policy as an RNN, replacing the velocity network in the flow policy with a recurrent modern Transformer architecture to solve training stability issues.

    -   **Log-Likelihood Calculation**: Adds Gaussian noise + corresponding drift correction in each sampling step to ensure the terminal action distribution remains unchanged, while decomposing the path density into a product of single-step Gaussian likelihoods, thereby obtaining a differentiable :math:`\log p_{\theta}(A|s)` .

3. **RLPD (Reinforcement Learning with Prior Data)**

   - A variant of SAC that combines offline data and online data for training.

   - To accelerate training in the real world, SAC-Flow can also be used with RLPD using pre-collected offline data as a demonstration buffer.

Installation
------------

For running in a simulation environment, please refer to :doc:`../../start/installation` for installation.

For running on real hardware, please refer to :doc:`franka` for installation and hardware configuration.

Run It
------

Pretrained Flow-T: GELLO demonstrations to online SACFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf also provides an opt-in, PyTorch-only path that reuses the same
``FlowPolicy`` and ``FlowTActor`` for behavior cloning and online SACFlow:

.. code-block:: text

   realworld_collect_data_gello.yaml
     ├─ collected_data/ → RollingLeRobotDataset → Flow BC
     └─ demos/          → TrajectoryReplayBuffer → online SACFlow

The two entry configurations are:

- ``examples/sft/config/franka_gello_flow_bc.yaml`` for Rectified Flow or
  experimental Improved MeanFlow supervision;
- ``examples/embodiment/config/franka_sacflow_online_finetune.yaml`` for a
  frozen-anchor warm-up followed by online SACFlow.

The checked-in examples select Improved MeanFlow. To train and fine-tune
Rectified Flow instead, change both
``flow_matching.objective: rectified_flow`` and
``flow_sampling.profile: rf_sacflow_sde_v1``, and set
``flow_sampling.flow_sde.field_source: instantaneous_velocity``. Also remove
the iMF-only ``noise_std_range`` and ``safe_initial_time`` keys from the
``flow_sde`` block. Objective or checkpoint mismatches are rejected before
weights are loaded.

Both configurations use ``flow_actor_type: FlowTActor``, a 19-dimensional
state, one action chunk, and a 7-dimensional GELLO action including the
gripper. The portable BC artifact is
``actor/flow_actor/model_state_dict/full_weights.pt`` plus
``actor/flow_actor/flow_actor_manifest.json``. A full online resume through
``runner.resume_dir`` takes precedence over actor-only initialization.

Improved MeanFlow keeps its native direction (``t=1`` noise to ``t=0`` action)
and conditions its interval field on two independently encoded times. Its ODE,
flow-noise, and flow-SDE samplers evaluate only the adjacent interval field
``U(z, t_from, t_to)``. The Flow-SDE transition follows the local OpenPI
MeanFlow formula, but the Flow-T implementation neither imports nor modifies
``openpi_action_model.py``. Online rollout is stochastic; evaluation is always
the deterministic ODE.

The online example initializes a fresh critic, hard-copies it to the target,
starts alpha at 0.2, and samples the online and demonstration buffers 50/50.
During the first 10,000 online transitions the critic trains normally while
the actor receives only frozen-anchor action regularization and alpha remains
fixed. Afterwards the actor loss is the SACFlow path-density objective plus
the same anchor regularizer. The live actor and anchor use common initial and
per-step noise when their actions are compared.

This iMF integration and the Franka deployment are engineering extensions.
The SACFlow paper's Appendix-F experiments were performed in simulation, so
validate checkpoint identity, action limits, stochastic-path scale, and the
warm-up phase in a dummy or hardware-in-the-loop setup before enabling robot
motion.

Launch the stages after replacing all data, checkpoint, robot, and target-pose
placeholders:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data_gello
   bash examples/sft/run_vla_sft.sh franka_gello_flow_bc
   bash examples/embodiment/run_embodiment.sh franka_sacflow_online_finetune

**1. Configuration Files**

RLinf provides default configuration files for both simulation and real-world environments:

-   **Simulation (ManiSkill)**: ``examples/embodiment/config/maniskill_sac_flow_state.yaml``
-   **Real World (Franka)**: ``examples/embodiment/config/realworld_sac_flow_image.yaml``

**2. Key Parameter Configuration**

**2.1 Model Parameters (Model)**

.. code:: yaml

   actor:
     model:
       model_type: "flow_policy"
       # Input type: 'state' (simulation) or 'mixed' (real world, image+state)
       input_type: "state"

       # Flow Matching related parameters
       denoising_steps: 4  # Number of denoising steps for action generation
       d_model: 256        # Transformer dimension
       n_head: 4           # Number of attention heads
       n_layers: 2         # Number of layers
       use_batch_norm: False  # Whether to use Batch Normalization
       batch_norm_momentum: 0.99  # Batch Normalization momentum
       flow_actor_type: "JaxFlowTActor"  # JAX style "JaxFlowTActor" or torch style "FlowTActor". "JaxFlowTActor" supports the following noise std settings:
       noise_std_head: False  # Whether to use a separate head to predict noise std, otherwise use fixed std
       # Noise std used during inference (rollout) can be smaller than during training to balance exploration and exploitation
       log_std_min_train: -5  # Min log std during training (if using noise_std_head)
       log_std_max_train: 2   # Max log std during training (if using noise_std_head)
       log_std_min_rollout: -20  # Min log std during rollout (if using noise_std_head)
       log_std_max_rollout: 0    # Max log std during rollout (if using noise_std_head)
       noise_std_train: 0.3  # Fixed noise std during training (if not using noise_std_head)
       noise_std_rollout: 0.02  # Fixed noise std during rollout (if not using noise_std_head)


**2.2 Algorithm Parameters (Algorithm)**

.. code:: yaml

   algorithm:
      # SAC Hyperparameters
      gamma: 0.96          # Discount factor
      tau: 0.005           # Target network soft update coefficient
      entropy_tuning:
         alpha_type: softplus # Entropy coefficient parameterization
         initial_alpha: 0.01  # Initial entropy coefficient
         target_entropy: -4
         optim:
            lr: 3.0e-4     # Entropy coefficient learning rate
            lr_scheduler: torch_constant
            clip_grad: 10.0
      critic_actor_ratio: 4  # Ratio of Critic to Actor training steps

      # Training and Interaction Frequency
      update_epoch: 30     # Number of training steps after each interaction

**2.3 Cluster and Hardware Configuration (Cluster)**

For real-world training, use a multi-node configuration, deploying the Actor/Policy on a GPU server and the Env/Robot on a control machine (NUC/Industrial PC). Specific configurations can be found in :doc:`franka`.


**3. Launch Commands**

**Simulation Training (ManiSkill)**

Launch simulation training on a single machine:

::

   bash examples/embodiment/run_embodiment.sh maniskill_sac_flow_state

**Real World Training (Franka)**

Launch real-world training in a distributed environment (needs to be run on the master node with cluster configured):

::

   bash examples/embodiment/run_realworld_async.sh realworld_sac_flow_image

Visualization and Results
-------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs

**2. Key Monitoring Metrics**

For metric definitions, see :doc:`Training metrics <../../reference/metrics>`. SAC-relevant metrics:

- **Environment Metrics**:

  - ``env/episode_len``: The actual number of environment steps in the episode
  - ``env/return``: Total return of the episode
  - ``env/reward``: Step-level reward from the environment
  - ``env/success_once``: Flag indicating at least one success in the episode (0 or 1)

- **Training Metrics**:

  - ``train/sac/critic_loss``: Loss of the Q-function
  - ``train/critic/grad_norm``: Gradient norm of the Q-function

  - ``train/sac/actor_loss``: Policy loss
  - ``train/actor/entropy``: Policy entropy
  - ``train/actor/grad_norm``: Gradient norm of the policy

  - ``train/sac/alpha_loss``: Loss of the temperature parameter
  - ``train/sac/alpha``: Value of the temperature parameter
  - ``train/alpha/grad_norm``: Gradient norm of the temperature parameter

  - ``train/replay_buffer/size``: Current size of the replay buffer
  - ``train/replay_buffer/max_reward``: Maximum reward stored in the replay buffer
  - ``train/replay_buffer/min_reward``: Minimum reward stored in the replay buffer
  - ``train/replay_buffer/mean_reward``: Mean reward stored in the replay buffer
  - ``train/replay_buffer/std_reward``: Standard deviation of rewards stored in the replay buffer
  - ``train/replay_buffer/utilization``: Utilization of the replay buffer

Real World Results
~~~~~~~~~~~~~~~~~~
Below are the demo video (accelerated) and training curve for the SAC-Flow algorithm on the Peg Insertion task. Within 30 minutes of training, the robot learns a policy capable of consistently completing the task.

.. raw:: html

  <div style="flex: 0.8; text-align: center;">
      <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-success-rate.png" style="width: 100%;"/>
      <p><em>Training Curve</em></p>
    </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-peg-insertion.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>Peg Insertion</em></p>
  </div>
