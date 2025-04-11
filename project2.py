import os
import signal
import logging
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.summary import create_file_writer

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Signal handler for graceful exit
def signal_handler(sig, frame):
    logging.info("Interrupt received, saving model and exiting...")
    if 'agent' in globals():
        agent.save_model("project_RL/models/ppo_agent_interrupt")
        with open("project_RL/checkpoint_metadata.json", "w") as f:
            json.dump({
                "last_episode": getattr(train_agent, 'last_episode', 0),
                "epsilon": agent.epsilon,
                "status": "interrupted",
                "timestamp": str(datetime.now())
            }, f, indent=4)
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Directory setup
os.makedirs("project_RL/data", exist_ok=True)
os.makedirs("project_RL/models", exist_ok=True)
os.makedirs("project_RL/checkpoints", exist_ok=True)
os.makedirs("project_RL/logs", exist_ok=True)  # TensorBoard logs
os.makedirs("project_RL/graphs", exist_ok=True)  # Final graphs

# TensorBoard writer
log_dir = "project_RL/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = create_file_writer(log_dir)

# Load environment data
env_data_path = "/fab3/btech/2022/vipul.pareek22b/Project/project_RL/data/rl_environment_data.json"
static_allocation_path = "/fab3/btech/2022/vipul.pareek22b/Project/project_RL/data/static_allocation.npy"

with open(env_data_path, 'r') as f:
    env_data = json.load(f)

static_allocation = np.load(static_allocation_path)

# Regular Ambulance Environment
class AmbulanceEnv:
    def __init__(self):
        self.n_clusters = env_data['n_clusters']
        self.n_ambulances = env_data['n_ambulances']
        self.static_allocation = static_allocation
        self.incident_rates = np.array(env_data['incident_rates'])
        self.travel_times = np.array(env_data['travel_times'])
        self.state_size = self.n_clusters + self.n_ambulances + self.n_clusters + self.n_ambulances + 1
        self.action_size = self.n_clusters
        self.h1, self.h2, self.h3 = 1.5, -0.1, -0.002  # Adjusted reward weights (h2 increased)
        self.reset()

    def reset(self):
        self.ambulance_locations = np.concatenate([np.full(count, i, dtype=int) for i, count in enumerate(self.static_allocation)])
        self.waiting_incidents = np.zeros(self.n_clusters)
        self.waiting_times = np.zeros(self.n_clusters)
        self.ambulance_fatigue = np.zeros(self.n_ambulances)
        self.step_count = 0
        self.hour = 0
        self.state = np.concatenate([
            self.waiting_incidents / 10.0,
            self.ambulance_locations.astype(float) / self.n_clusters,
            self.waiting_times / 60.0,
            self.ambulance_fatigue / 10.0,
            [self.hour / 24.0]
        ])
        return self.state

    def step(self, action):
        reward = 0
        peak_factor = 2.0 if self.hour in {17, 18, 19, 20} else 1.0
        new_incidents = np.random.poisson(self.incident_rates * peak_factor)
        self.waiting_incidents += new_incidents
        self.waiting_times += self.waiting_incidents

        for amb, cluster in enumerate(action):
            self.ambulance_locations[amb] = cluster
            self.ambulance_fatigue[amb] = min(self.ambulance_fatigue[amb] + 1, 10)
            m_kt = min(self.waiting_incidents[cluster], 1)
            w_kt = self.waiting_times[cluster]
            T_ki = self.travel_times[amb, cluster]
            reward += self.h1 * m_kt + self.h2 * w_kt + self.h3 * T_ki

        cluster_counts = np.bincount(self.ambulance_locations, minlength=self.n_clusters)
        self.waiting_incidents -= np.minimum(self.waiting_incidents, cluster_counts)
        self.waiting_times = self.waiting_incidents * 5
        self.step_count += 1
        self.hour = (self.hour + 1) % 24

        self.state = np.concatenate([
            self.waiting_incidents / 10.0,
            self.ambulance_locations.astype(float) / self.n_clusters,
            self.waiting_times / 60.0,
            self.ambulance_fatigue / 10.0,
            [self.hour / 24.0]
        ])
        done = self.step_count >= 200
        info = {
            'waiting_time': np.sum(self.waiting_times),
            'answered_requests': np.sum(np.minimum(new_incidents, cluster_counts)),
            'total_requests': np.sum(new_incidents)
        }
        return self.state, reward, done, info

# Chaotic Ambulance Environment (kept for evaluation only)
class ChaoticAmbulanceEnv:
    def __init__(self):
        self.n_clusters = env_data['n_clusters']
        self.n_ambulances = env_data['n_ambulances']
        self.static_allocation = static_allocation
        self.incident_rates = np.random.uniform(0.5, 2.0, self.n_clusters)
        self.travel_times = np.array(env_data['travel_times'])
        self.state_size = self.n_clusters + self.n_ambulances + self.n_clusters + self.n_ambulances + 1
        self.action_size = self.n_clusters
        self.h1, self.h2, self.h3 = 1.5, -0.1, -0.002  # Adjusted reward weights (h2 increased)
        self.reset()

    def reset(self):
        self.ambulance_locations = np.random.randint(0, self.n_clusters, self.n_ambulances)
        self.waiting_incidents = np.zeros(self.n_clusters)
        self.waiting_times = np.zeros(self.n_clusters)
        self.ambulance_fatigue = np.zeros(self.n_ambulances)
        self.step_count = 0
        self.hour = 0
        self.state = np.concatenate([
            self.waiting_incidents / 10.0,
            self.ambulance_locations.astype(float) / self.n_clusters,
            self.waiting_times / 60.0,
            self.ambulance_fatigue / 10.0,
            [self.hour / 24.0]
        ])
        return self.state

    def step(self, action):
        reward = 0
        peak_factor = np.random.uniform(1.0, 3.0)
        new_incidents = np.random.poisson(self.incident_rates * peak_factor)
        self.waiting_incidents += new_incidents
        self.waiting_times += self.waiting_incidents

        for amb, cluster in enumerate(action):
            self.ambulance_locations[amb] = cluster
            self.ambulance_fatigue[amb] = min(self.ambulance_fatigue[amb] + 1, 10)
            m_kt = min(self.waiting_incidents[cluster], 1)
            w_kt = self.waiting_times[cluster]
            T_ki = self.travel_times[amb, cluster]
            reward += self.h1 * m_kt + self.h2 * w_kt + self.h3 * T_ki

        cluster_counts = np.bincount(self.ambulance_locations, minlength=self.n_clusters)
        self.waiting_incidents -= np.minimum(self.waiting_incidents, cluster_counts)
        self.waiting_times = self.waiting_incidents * 5
        self.step_count += 1
        self.hour = (self.hour + 1) % 24

        self.state = np.concatenate([
            self.waiting_incidents / 10.0,
            self.ambulance_locations.astype(float) / self.n_clusters,
            self.waiting_times / 60.0,
            self.ambulance_fatigue / 10.0,
            [self.hour / 24.0]
        ])
        done = self.step_count >= 200
        info = {
            'waiting_time': np.sum(self.waiting_times),
            'answered_requests': np.sum(np.minimum(new_incidents, cluster_counts)),
            'total_requests': np.sum(new_incidents)
        }
        return self.state, reward, done, info

# PPO Agent
class PPOAgent:
    def __init__(self, state_size, action_size, n_ambulances):
        self.state_size = state_size
        self.action_size = action_size
        self.n_ambulances = n_ambulances
        self.memory = []
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.batch_size = 256  # Increased for stability
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Adjusted for slower exploration d

        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)  # Lowered learning rate
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)  # Lowered learning rate

        self.load_latest_checkpoint()
        logging.info("PPO Agent initialized")

    def _build_actor(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_size * self.n_ambulances)
        ])
        return model

    def _build_critic(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])
        return model

    def load_latest_checkpoint(self):
        checkpoint_dir = "project_RL/checkpoints"
        metadata_file = "project_RL/checkpoint_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            last_episode = metadata.get("last_episode", -1)
            epsilon = metadata.get("epsilon", 1.0)
            checkpoint_path = f"{checkpoint_dir}/ppo_checkpoint_epoch_{last_episode}"
            if os.path.exists(f"{checkpoint_path}_actor.keras") and os.path.exists(f"{checkpoint_path}_critic.keras"):
                self.actor = tf.keras.models.load_model(f"{checkpoint_path}_actor.keras")
                self.critic = tf.keras.models.load_model(f"{checkpoint_path}_critic.keras")
                self.epsilon = epsilon
                logging.info(f"Loaded checkpoint from episode {last_episode}")
                return last_episode
        logging.info("No checkpoint found, starting fresh")
        return -1

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size, size=self.n_ambulances), np.zeros(self.n_ambulances)
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.actor(state)
        logits = tf.reshape(logits, [self.n_ambulances, self.action_size])
        probs = tf.nn.softmax(logits)
        actions = tf.random.categorical(logits, 1)[:, 0].numpy()
        log_probs = tf.reduce_sum(tf.math.log(probs + 1e-10) * tf.one_hot(actions, self.action_size), axis=1)
        return actions, log_probs.numpy()

    def decay_epsilon(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns

    def update(self):
        if not self.memory:
            return 0, 0, 0

        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)

        values = self.critic(states)[:, 0].numpy()
        next_values = self.critic(next_states)[:, 0].numpy()
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_log_probs, advantages, returns))
        dataset = dataset.shuffle(buffer_size=len(self.memory)).batch(self.batch_size)

        total_actor_loss = 0
        total_critic_loss = 0
        max_kl = 0
        for _ in range(4):
            for batch in dataset:
                s, a, olp, adv, ret = batch
                with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                    logits = self.actor(s)
                    logits = tf.reshape(logits, [-1, self.n_ambulances, self.action_size])
                    probs = tf.nn.softmax(logits)
                    probs_chosen = tf.gather(probs, a, axis=2, batch_dims=2)
                    new_log_probs = tf.reduce_sum(tf.math.log(probs_chosen + 1e-10), axis=1)
                    ratio = tf.exp(new_log_probs - tf.reduce_sum(olp, axis=1))
                    surr1 = ratio * adv
                    surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    values = self.critic(s)[:, 0]
                    critic_loss = tf.reduce_mean(tf.square(ret - values))
                    kl_div = tf.reduce_mean(tf.reduce_sum(olp, axis=1) - new_log_probs)

                actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                total_actor_loss += actor_loss.numpy()
                total_critic_loss += critic_loss.numpy()
                max_kl = max(max_kl, kl_div.numpy())

        self.memory = []
        return total_actor_loss / 4, total_critic_loss / 4, max_kl

    def save_model(self, filepath):
        self.actor.save(filepath + "_actor.keras")
        self.critic.save(filepath + "_critic.keras")
        logging.info(f"Models saved to {filepath}")

def train_agent(regular_env, chaotic_env, agent, episodes):
    print("=== Starting Training Phase ===", flush=True)
    start_episode = agent.load_latest_checkpoint() + 1
    train_agent.last_episode = start_episode - 1
    scores = []

    # Curriculum learning weights (commented out, using only regular env as requested)
    curriculum_phases = [
        (2500, 1.0, 0.0),  # Episodes 0-2500: 100% regular
        (5000, 0.75, 0.25),  # Episodes 2501-5000: 75% regular, 25% chaotic
        (7500, 0.6, 0.4),   # Episodes 5001-7500: 50% regular, 50% chaotic
        (10000, 0.5, 0.5)   # Episodes 7501-10000: 50% regular, 50% chaotic
    ]

    for e in tqdm(range(start_episode, episodes), desc="Training", unit="episode"):
        # Curriculum logic (disabled, using only regular environment)
        reg_prob = 1.0
        for phase_end, reg_weight, chaotic_weight in curriculum_phases:
            if e <= phase_end:
                reg_prob = reg_weight
                break
        env = regular_env if np.random.rand() < reg_prob else chaotic_env
        # env = regular_env  #Train only on regular environment

        state = env.reset()
        total_reward = 0
        total_waiting_time = 0
        total_answered = 0
        total_requests = 0
        done = False
        steps = 0

        while not done and steps < 200:
            action, log_prob = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.append((state, action, reward, next_state, done, log_prob))
            state = next_state
            total_reward += reward
            total_waiting_time += info['waiting_time']
            total_answered += info['answered_requests']
            total_requests += info['total_requests']
            steps += 1

        actor_loss, critic_loss, kl_div = agent.update()
        epsilon = agent.decay_epsilon(e)
        scores.append(total_reward)
        now_time = total_waiting_time / max(1, total_requests)
        nrar = total_answered / max(1, total_requests) * 100
        train_agent.last_episode = e

        if e % 10 == 0 or e == episodes - 1:
            agent.save_model(f"project_RL/checkpoints/ppo_checkpoint_epoch_{e}")
            with open("project_RL/checkpoint_metadata.json", "w") as f:
                json.dump({
                    "last_episode": e,
                    "epsilon": epsilon,
                    "status": "running",
                    "timestamp": str(datetime.now())
                }, f, indent=4)

        if e % 50 == 0 or e == episodes - 1:
            print(f"Episode {e}: Reward={total_reward:.2f}, NOW Time={now_time:.2f}, NRAR={nrar:.2f}%, "
                  f"KL={kl_div:.4f}, Epsilon={epsilon:.4f}, Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}",
                  flush=True)
            with summary_writer.as_default():
                tf.summary.scalar("Reward", total_reward, step=e)
                tf.summary.scalar("NOW Time", now_time, step=e)
                tf.summary.scalar("NRAR", nrar, step=e)
                tf.summary.scalar("KL Divergence", kl_div, step=e)
                tf.summary.scalar("Epsilon", epsilon, step=e)
                tf.summary.scalar("Actor Loss", actor_loss, step=e)
                tf.summary.scalar("Critic Loss", critic_loss, step=e)

    agent.save_model("project_RL/models/ppo_agent_final")
    with open("project_RL/checkpoint_metadata.json", "w") as f:
        json.dump({
            "last_episode": episodes - 1,
            "epsilon": agent.epsilon,
            "status": "completed",
            "timestamp": str(datetime.now())
        }, f, indent=4)
    return scores

def evaluate_agent(env, agent, episodes, env_type):
    print(f"=== Starting {env_type} Environment Evaluation for PPO Agent ===", flush=True)
    rewards = []
    waiting_times = []
    answered_requests = []
    total_requests_list = []

    for _ in tqdm(range(episodes), desc=f"Evaluating PPO on {env_type}", unit="episode"):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        episode_waiting_time = 0
        episode_answered = 0
        episode_requests = 0
        while not done and steps < 200:
            action, _ = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            episode_waiting_time += info['waiting_time']
            episode_answered += info['answered_requests']
            episode_requests += info['total_requests']
            steps += 1
        
        rewards.append(episode_reward)
        waiting_times.append(episode_waiting_time)
        answered_requests.append(episode_answered)
        total_requests_list.append(episode_requests)

    avg_reward = np.mean(rewards)
    reward_variance = np.var(rewards)
    now_time = np.sum(waiting_times) / max(1, np.sum(total_requests_list))
    nrar = np.sum(answered_requests) / max(1, np.sum(total_requests_list)) * 100
    now_time_variance = np.var([wt / max(1, tr) for wt, tr in zip(waiting_times, total_requests_list)])
    nrar_variance = np.var([ar / max(1, tr) * 100 for ar, tr in zip(answered_requests, total_requests_list)])

    print(f"{env_type} PPO Agent - Reward: {avg_reward:.2f}, NOW Time: {now_time:.2f}, NRAR: {nrar:.2f}%", flush=True)
    with summary_writer.as_default():
        tf.summary.scalar(f"{env_type}/PPO Reward", avg_reward, step=episodes)
        tf.summary.scalar(f"{env_type}/PPO NOW Time", now_time, step=episodes)
        tf.summary.scalar(f"{env_type}/PPO NRAR", nrar, step=episodes)
    return avg_reward, now_time, nrar, reward_variance, now_time_variance, nrar_variance

def evaluate_baseline(env, episodes, policy, env_type):
    print(f"=== Starting {env_type} Environment Evaluation for {policy} Policy ===", flush=True)
    rewards = []
    waiting_times = []
    answered_requests = []
    total_requests_list = []

    for _ in tqdm(range(episodes), desc=f"Evaluating {policy} on {env_type}", unit="episode"):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        episode_waiting_time = 0
        episode_answered = 0
        episode_requests = 0
        while not done and steps < 200:
            if policy == 'Random':
                action = np.random.randint(env.action_size, size=env.n_ambulances)
            elif policy == 'Location-Based':
                action = np.argmin(env.travel_times, axis=1)
            elif policy == 'Time-Based':
                action = np.full(env.n_ambulances, np.argmax(env.waiting_times))
            elif policy == 'Request-Based':
                action = np.full(env.n_ambulances, np.argmax(env.waiting_incidents))
            elif policy == 'Static':
                action = np.concatenate([np.full(count, i) for i, count in enumerate(env.static_allocation)])

            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            episode_waiting_time += info['waiting_time']
            episode_answered += info['answered_requests']
            episode_requests += info['total_requests']
            steps += 1
        
        rewards.append(episode_reward)
        waiting_times.append(episode_waiting_time)
        answered_requests.append(episode_answered)
        total_requests_list.append(episode_requests)

    avg_reward = np.mean(rewards)
    reward_variance = np.var(rewards)
    now_time = np.sum(waiting_times) / max(1, np.sum(total_requests_list))
    nrar = np.sum(answered_requests) / max(1, np.sum(total_requests_list)) * 100
    now_time_variance = np.var([wt / max(1, tr) for wt, tr in zip(waiting_times, total_requests_list)])
    nrar_variance = np.var([ar / max(1, tr) * 100 for ar, tr in zip(answered_requests, total_requests_list)])

    print(f"{env_type} {policy} - Reward: {avg_reward:.2f}, NOW Time: {now_time:.2f}, NRAR: {nrar:.2f}%", flush=True)
    with summary_writer.as_default():
        tf.summary.scalar(f"{env_type}/{policy} Reward", avg_reward, step=episodes)
        tf.summary.scalar(f"{env_type}/{policy} NOW Time", now_time, step=episodes)
        tf.summary.scalar(f"{env_type}/{policy} NRAR", nrar, step=episodes)
    return avg_reward, now_time, nrar, reward_variance, now_time_variance, nrar_variance

def plot_results(scores, regular_results, chaotic_results):
    print("=== Generating Final Graphs ===", flush=True)
    
    # Training Reward Plot
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig("project_RL/graphs/training_reward.png")
    plt.close()

    # Bar Chart Comparisons
    policies = list(regular_results.keys())
    regular_rewards = [regular_results[p]["Reward"] for p in policies]
    regular_now_times = [regular_results[p]["NOW Time"] for p in policies]
    regular_nrars = [regular_results[p]["NRAR"] for p in policies]
    chaotic_rewards = [chaotic_results[p]["Reward"] for p in policies]
    chaotic_now_times = [chaotic_results[p]["NOW Time"] for p in policies]
    chaotic_nrars = [chaotic_results[p]["NRAR"] for p in policies]

    # Regular Environment Bar Charts
    plt.figure(figsize=(12, 6))
    plt.bar(policies, regular_rewards)
    plt.xlabel('Policy')
    plt.ylabel('Reward')
    plt.title('Regular Environment: Reward Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("project_RL/graphs/regular_reward_comparison.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(policies, regular_now_times)
    plt.xlabel('Policy')
    plt.ylabel('NOW Time (min/request)')
    plt.title('Regular Environment: NOW Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("project_RL/graphs/regular_now_time_comparison.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(policies, regular_nrars)
    plt.xlabel('Policy')
    plt.ylabel('NRAR (%)')
    plt.title('Regular Environment: NRAR Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("project_RL/graphs/regular_nrar_comparison.png")
    plt.close()

    # Chaotic Environment Bar Charts
    plt.figure(figsize=(12, 6))
    plt.bar(policies, chaotic_rewards)
    plt.xlabel('Policy')
    plt.ylabel('Reward')
    plt.title('Chaotic Environment: Reward Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("project_RL/graphs/chaotic_reward_comparison.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(policies, chaotic_now_times)
    plt.xlabel('Policy')
    plt.ylabel('NOW Time (min/request)')
    plt.title('Chaotic Environment: NOW Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("project_RL/graphs/chaotic_now_time_comparison.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(policies, chaotic_nrars)
    plt.xlabel('Policy')
    plt.ylabel('NRAR (%)')
    plt.title('Chaotic Environment: NRAR Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("project_RL/graphs/chaotic_nrar_comparison.png")
    plt.close()

if __name__ == "__main__":
    start_time = datetime.now()

    # Create environments
    regular_env = AmbulanceEnv()
    chaotic_env = ChaoticAmbulanceEnv()
    agent = PPOAgent(regular_env.state_size, regular_env.action_size, regular_env.n_ambulances)

    # Training on regular environment only, extended to 10,000 episodes
    scores = train_agent(regular_env, chaotic_env, agent, episodes=10000)
    training_results = {
        "training_scores": scores,
        "episodes": 10000,
        "avg_training_reward": float(np.mean(scores)),
        "reward_variance": float(np.var(scores)),
        "hyperparameters": {
            "gamma": agent.gamma,
            "gae_lambda": agent.gae_lambda,
            "clip_ratio": agent.clip_ratio,
            "batch_size": agent.batch_size,
            "epsilon_start": 1.0,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay,
            "learning_rate_actor": 2e-5,
            "learning_rate_critic": 2e-5
        },
        "start_time": str(start_time),
        "end_time_training": str(datetime.now())
    }

    # Evaluation on regular environment
    agent_reward, agent_now_time, agent_nrar, agent_reward_var, agent_now_var, agent_nrar_var = evaluate_agent(regular_env, agent, episodes=500, env_type="Regular")
    baselines = {
        'Random': 'Random',
        'Location-Based': 'Location-Based',
        'Time-Based': 'Time-Based',
        'Request-Based': 'Request-Based',
        'Static': 'Static'
    }
    regular_results = {
        "PPO Agent": {
            "Reward": agent_reward,
            "NOW Time": agent_now_time,
            "NRAR": agent_nrar,
            "Reward Variance": agent_reward_var,
            "NOW Time Variance": agent_now_var,
            "NRAR Variance": agent_nrar_var
        }
    }
    for name, policy in baselines.items():
        reward, now_time, nrar, reward_var, now_var, nrar_var = evaluate_baseline(regular_env, 500, policy, "Regular")
        regular_results[name] = {
            "Reward": reward,
            "NOW Time": now_time,
            "NRAR": nrar,
            "Reward Variance": reward_var,
            "NOW Time Variance": now_var,
            "NRAR Variance": nrar_var
        }

    # Evaluation on chaotic environment
    chaotic_agent_reward, chaotic_now_time, chaotic_nrar, chaotic_reward_var, chaotic_now_var, chaotic_nrar_var = evaluate_agent(chaotic_env, agent, episodes=500, env_type="Chaotic")
    chaotic_results = {
        "PPO Agent": {
            "Reward": chaotic_agent_reward,
            "NOW Time": chaotic_now_time,
            "NRAR": chaotic_nrar,
            "Reward Variance": chaotic_reward_var,
            "NOW Time Variance": chaotic_now_var,
            "NRAR Variance": chaotic_nrar_var
        }
    }
    for name, policy in baselines.items():
        reward, now_time, nrar, reward_var, now_var, nrar_var = evaluate_baseline(chaotic_env, 500, policy, "Chaotic")
        chaotic_results[name] = {
            "Reward": reward,
            "NOW Time": now_time,
            "NRAR": nrar,
            "Reward Variance": reward_var,
            "NOW Time Variance": now_var,
            "NRAR Variance": nrar_var
        }

    # Generate and save plots
    plot_results(scores, regular_results, chaotic_results)

    # Save results
    print("=== Starting Result Saving Phase ===", flush=True)
    final_json_results = {
        "training": training_results,
        "regular_evaluation": regular_results,
        "chaotic_evaluation": chaotic_results,
        "metadata": {
            "tensorflow_version": tf.__version__,
            "execution_end_time": str(datetime.now()),
            "total_duration": str(datetime.now() - start_time)
        }
    }
    with open("project_RL/results.json", "w") as f:
        json.dump(final_json_results, f, indent=4)
    logging.info("Detailed evaluation results saved to project_RL/results.json")

    with open("project_RL/final_results.txt", "w") as f:
        f.write("=== Ambulance Dispatch RL Project - Final Results ===\n")
        f.write(f"Execution Start Time: {start_time}\n")
        f.write(f"Execution End Time: {datetime.now()}\n")
        f.write(f"Total Duration: {datetime.now() - start_time}\n\n")
        
        f.write("=== Training Summary ===\n")
        f.write(f"Episodes: {training_results['episodes']}\n")
        f.write(f"Average Training Reward: {training_results['avg_training_reward']:.2f}\n")
        f.write(f"Training Reward Variance: {training_results['reward_variance']:.2f}\n\n")
        
        f.write("=== Regular Environment Evaluation ===\n")
        for name, metrics in regular_results.items():
            f.write(f"{name}:\n")
            f.write(f"  Reward: {metrics['Reward']:.2f} (Variance: {metrics['Reward Variance']:.2f})\n")
            f.write(f"  NOW Time: {metrics['NOW Time']:.2f} min/request (Variance: {metrics['NOW Time Variance']:.2f})\n")
            f.write(f"  NRAR: {metrics['NRAR']:.2f}% (Variance: {metrics['NRAR Variance']:.2f})\n")
        
        f.write("\n=== Chaotic Environment Evaluation ===\n")
        for name, metrics in chaotic_results.items():
            f.write(f"{name}:\n")
            f.write(f"  Reward: {metrics['Reward']:.2f} (Variance: {metrics['Reward Variance']:.2f})\n")
            f.write(f"  NOW Time: {metrics['NOW Time']:.2f} min/request (Variance: {metrics['NOW Time Variance']:.2f})\n")
            f.write(f"  NRAR: {metrics['NRAR']:.2f}% (Variance: {metrics['NRAR Variance']:.2f})\n")
        
        f.write("\n=== Hyperparameters ===\n")
        for key, value in training_results["hyperparameters"].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n=== System Info ===\n")
        f.write(f"TensorFlow Version: {tf.__version__}\n")
    logging.info("Human-readable final results saved to project_RL/final_results.txt")
    print("=== Execution Completed ===", flush=True)
    print(f"TensorBoard logs saved to {log_dir}. Run 'tensorboard --logdir project_RL/logs' to view.", flush=True)