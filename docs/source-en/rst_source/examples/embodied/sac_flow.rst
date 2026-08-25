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

Flow-T BC Action Chunks
~~~~~~~~~~~~~~~~~~~~~~~~

The PyTorch ``FlowPolicy`` and ``FlowTActor`` use one contract for a single
action and for a future-action chunk. Set ``actor.model.action_horizon`` to a
positive integer; the public action, target, and sample shapes are always
``[B,H,A]``, where ``A`` is ``actor.model.action_dim``. ``H=1`` retains the
horizon dimension. The GELLO example below uses ``A=7``.

The GELLO dataset creates future windows within one episode. It returns a
prefix ``action_valid_mask: [B,H]`` and writes exact zeros into invalid tail
slots. The mask is used only by the loss and evaluation metrics. It is never
passed to the field network or sampler, so a policy cannot observe the target
episode's remaining length.

Internally, ``FlowTActor`` flattens the chunk to a chunk-major ``[B,H*A]`` flow
state. All horizons use ``tanh_latent`` and identity latent normalization.
Only valid coordinates contribute to the RF/iMF loss, gradients, norms, and
per-horizon metrics; zero padding never enters a metric denominator.

A minimal model contract is:

.. code-block:: yaml

   actor:
     model:
       flow_actor_type: FlowTActor
       action_dim: 7
       action_horizon: 8
       flow_matching:
         implementation: pytorch_flow_t
         objective: improved_meanflow
         action_transform: tanh_latent
         action_chunking:
           action_layout: chunk_major
           latent_normalization: identity
       flow_sampling:
         evaluation:
           method: flow_ode
           num_steps: 10

Static BC evaluation supports both ``flow_ode`` and ``flow_sde`` for every
horizon. For iMF SDE, configure a positive ``noise_level``,
``0 < std_min <= std_max``, and ``safe_initial_time > 1 - 1/num_steps``:

.. code-block:: yaml

   flow_sampling:
     evaluation:
       method: flow_sde
       num_steps: 10
     flow_sde:
       noise_level: 0.1
       noise_std_range: [0.005, 0.05]
       safe_initial_time: 0.99
       joint_path_logprob: true

RF SDE uses the existing corrected-drift kernel and accepts ``noise_level``
only; ``noise_std_range`` and ``safe_initial_time`` are iMF-only. Explicit
initial noise has shape ``[B,H,A]`` and explicit SDE step noise has shape
``[B,N,H,A]``. Reusing both tensors reproduces the complete path.

.. warning::

   ``H>1`` currently supports offline BC and static ODE/SDE sampling only.
   Online SACFlow, rollout, replay, discounting, and real-robot chunk execution
   remain restricted to ``action_horizon: 1``.

Train the checked-in GELLO example after setting its data and encoder paths:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh franka_gello_flow_bc

Evaluate a portable actor on validation episodes with ODE:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python \
     examples/sft/evaluate_franka_gello_flow_bc.py \
     --checkpoint /path/to/global_step_N \
     --data-root /path/to/collected_data \
     --split val \
     --sampler-method flow_ode \
     --num-steps 10

The same checkpoint can be evaluated with SDE without changing its weights:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python \
     examples/sft/evaluate_franka_gello_flow_bc.py \
     --checkpoint /path/to/global_step_N \
     --data-root /path/to/collected_data \
     --split val \
     --sampler-method flow_sde \
     --num-steps 10 \
     --sde-noise-level 0.1 \
     --sde-noise-std-range 0.005 0.05 \
     --sde-safe-initial-time 0.99 \
     --initial-noise-seed 1234 \
     --step-noise-seed 1235

The report records the actual sampler, steps, SDE parameters, and random seeds.
Aggregate and per-horizon metrics ignore padded slots. The all-valid full-chunk
view includes only samples where ``action_valid_mask.all(dim=1)``. Best-of-K
selects one whole valid chunk per observation rather than combining different
samples across horizons.

Portable artifacts contain ``flow_actor_manifest.json`` and
``model_state_dict/full_weights.pt``. The manifest accepts only
``schema: unified_flow_t_bc`` and records the objective/time contract, action
horizon, internal flow width, chunk layout, identity normalization, zero
padding, and loss-only mask role. ODE/SDE choice is runtime configuration and
is not a weight-compatibility field. Old manifests, cross-horizon loads, and
projection inflation are rejected.

Online SACFlow remains an H=1 workflow. Produce a separate compatible H=1
artifact with the same image/state/action and objective contract before using
``examples/embodiment/config/franka_sacflow_online_finetune.yaml``.

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
