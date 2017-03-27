import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections
import plotting
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.kernel_approximation import RBFSampler

env = gym.envs.make("MountainCarContinuous-v0")
print env.observation_space.sample()
print env.reset()

# Feature Preprocessing: normalize state to zero mean and unit variance
observation_examples = np.array([env.observation_space.sample() for x in range(100000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featureized representation
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100)),
])
featurizer.fit(scaler.transform(observation_examples))

def featurize_state(state):
    """
    Return the featurized representation for a state
    :param state:
    :return:
    """

    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]

# Deterministic Policy
class PolicyEstimator():
    """
    Policy Function approximator
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")  # s_t
            self.action = tf.placeholder(dtype=tf.float32, name="action")  # a_t
            self.target = tf.placeholder(dtype=tf.float32, name="target") # advantage = G_t - baseline

            # This is just linear classifier
            self.mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state,0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer()
            )
            self.mu = tf.squeeze(self.mu)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer()
            )

            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5

            # define distribution of actions
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

            # pick up action from distribution and clip by min/max
            self.action = self.normal_dist._sample_n(1)
            self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])

            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            self.loss -= 1e-1 * self.normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step()
            )

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.action, { self.state: state } )

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
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
            self.state = tf.placeholder(tf.float32, [400], "state")  # s_t
            self.target = tf.placeholder(dtype=tf.float32, name="target") # td_target

            # This is just table lookup estimator
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state,0),
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
        state = featurize_state(state)
        return sess.run(self.value_estimate, { self.state: state } )

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {
            self.state: state, self.target:target
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor-Critic Algorithm : optimize the policy using policy gradient
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

            env.render()

            # take a step according to current deterministic policy
            action = estimator_policy.predict(state)
            next_state, reward, done, _ = env.step(action)

            # keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done
            ))

            # Update statistics
            stats.episode_lengths[i_episode] += t
            stats.episode_rewards[i_episode] += reward

            # Calculate TD target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor*value_next
            td_error = td_target - estimator_value.predict(state)

            # update the value estimator
            estimator_value.update(state, td_target)

            # update the policy estimator
            # using td error as our advantage estimate
            estimator_policy.update(state, td_error, action)


            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode+1, num_episodes, stats.episode_rewards[i_episode-1]))

            if done:
                break

            state = next_state

    return stats


tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(learning_rate=0.001)
value_estimator = ValueEstimator(learning_rate=0.1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = actor_critic(env, policy_estimator, value_estimator, 50, discount_factor=0.95)

plotting.plot_episode_stats(stats, smoothing_window=25)









