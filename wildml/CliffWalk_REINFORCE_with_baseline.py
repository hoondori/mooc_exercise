
import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections
import plotting

matplotlib.use('TkAgg')

#matplotlib.style.use('ggplot')

from envs.cliff_walking import CliffWalkingEnv

env = CliffWalkingEnv()

class PolicyEstimator():
    """
    Policy Function approximator
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")  # s_t
            self.action = tf.placeholder(dtype=tf.int32, name="action")  # a_t
            self.target = tf.placeholder(dtype=tf.float32, name="target") # advantage = G_t - baseline

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot,0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer()
            )
            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)   # pi(a_t|s_t)

            self.loss = -tf.log(self.picked_action_prob) * self.target  # J = -log(pi(a_t|s_t)*advantage
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step()
            )

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state: state } )

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {
            self.state: state, self.target:target, self.action: action
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")  # s_t
            self.target = tf.placeholder(dtype=tf.float32, name="target") # td_target

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot,0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step()
            )

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state } )

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {
            self.state: state, self.target:target
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def reinforce(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm
    :param env: OpenAI environment
    :param estimator_policy: Policy approx func to be optimized
    :param estimator_value: Value approx func used as a baseline
    :param num_episodes: Number of episodes to run for
    :param discount_factor:
    :return:
    """

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        state = env.reset()

        episode = []

        # Generate sampled episode following current policy
        for t in itertools.count():

            # take a step according to current policy
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done
            ))

            # Update statistics
            stats.episode_lengths[i_episode] += t
            stats.episode_rewards[i_episode] += reward

            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode+1, num_episodes, stats.episode_rewards[i_episode-1]))

            if done:
                break

            state = next_state

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            total_return =  sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
            estimator_value.update(transition.state, total_return)
            baseline_value = estimator_value.predict(transition.state)
            advantage = total_return - baseline_value
            estimator_policy.update(transition.state, advantage, transition.action)
    return stats

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = reinforce(env, policy_estimator, value_estimator, 2000, discount_factor=1.0)

plotting.plot_episode_stats(stats, smoothing_window=25)