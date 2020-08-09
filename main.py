#https://github.com/bulletphysics/bullet3/tree/2f09c7c2240420cf37b246a8a94f780604c24801/examples/pybullet/gym/pybullet_robots/panda
#https://arxiv.org/pdf/1802.09464.pdf
import pybullet as p
import pybullet_data as pd
from box import Box
import yaml

import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.trajectories import trajectory
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from pybulletArmEnv.pyenvArm import PandaEnv

with open('./pybulletArmEnv/env_config.yaml', 'r') as ymlfile:
    cfg = Box(yaml.safe_load(ymlfile))

def agent_setup(env):
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=(100,)
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=10000)

    return agent, replay_buffer


def pybullet_setup(time_step=1./240):
    p.connect(p.DIRECT)
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
                                 cameraTargetPosition=[0.35, -0.13, 0])
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setTimeStep(time_step)
    p.setGravity(0, -9.8, 0)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for i in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)



if __name__ == '__main__':
    pybullet_setup()
    env = PandaEnv(p, [0, 0, 0])
    tf_env = tf_py_environment.TFPyEnvironment(env)

    agent, replay_buffer = agent_setup(tf_env)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(tf_env, agent.policy, 5)
    returns = [avg_return]

    for _ in range(3000):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(1):
            collect_step(tf_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % 20 == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % 5 == 0:
            avg_return = compute_avg_return(env, agent.policy, 5)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

