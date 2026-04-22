"""
Deep Q-Network (DQN) implementation in PyTorch for ALE/Breakout-v5.

This module intentionally avoids high-level RL libraries so every piece of
the algorithm is visible and editable for learning purposes.

Key components:
  - QNetwork     : CNN DeepMind que mapea 4 fotogramas grises → Q(s,a)
  - ReplayBuffer : stores (s, a, r, s', done) transitions for experience replay
  - DQNAgent     : the training loop, ε-greedy policy, and target-network sync
"""

import cv2
import torch
import random

import numpy       as np
import gymnasium   as gym
import torch.nn    as nn
import torch.optim as optim

from collections import deque
from pathlib     import Path
from typing      import Self

# ── Frame preprocessing ──────────────────────────────────────────────

FRAME_H     = 84
FRAME_W     = 84
FRAME_STACK = 4
ARCH_TAG    = "cnn_breakout_v1"


def preprocess_frame(obs: np.ndarray) -> np.ndarray:
    """Convierte un fotograma RGB (210×160×3) a gris normalizado (84×84)."""
    gray    = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    """Apila los últimos `k` fotogramas. Estado resultante: (k, 84, 84)."""

    def __init__(self, k: int = FRAME_STACK) -> None:
        self.k = k
        self._frames: deque[np.ndarray] = deque(maxlen=k)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        frame = preprocess_frame(obs)
        for _ in range(self.k):
            self._frames.append(frame)
        return self._get()

    def step(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(preprocess_frame(obs))
        return self._get()

    def _get(self) -> np.ndarray:
        return np.stack(list(self._frames), axis=0)


# ── Neural network ────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Arquitectura CNN DeepMind (Nature 2015) para Breakout.
    Entrada: (batch, 4, 84, 84) → Salida: (batch, action_dim)
    """
    def __init__(self, action_dim: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            # Primera Capa Conv: 4 canales → 32 filtros
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Segunda Capa Conv: 32 → 64 filtros
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Tercera Capa Conv: 64 → 64 filtros
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            # Flatten + Capas Fully Connected
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paso hacia adelante: predice los valores Q para el estado x."""
        return self.network(x)

# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions for experience replay."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        # Guardar como uint8 para ahorrar RAM (~4× menos)
        self.buffer.append((
            (state      * 255).astype(np.uint8),
            action,
            reward,
            (next_state * 255).astype(np.uint8),
            done,
        ))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent:
    """
    Deep Q-Network agent implemented from scratch.

    Hyperparameters are intentionally exposed as constructor args so you
    can experiment with them directly.
    """

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9999,
        batch_size: int = 32,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1_000,
        learn_every: int = 4,
        min_buffer: int = 10_000,
    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_every = learn_every
        self.min_buffer = min_buffer
        self.training_episodes = 0
        self.total_steps = 0

        env = gym.make(env_id)
        self.action_dim = int(env.action_space.n)
        env.close()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Usando GPU CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("ADVERTENCIA: GPU no encontrada, usando CPU (entrenamiento lento)")

        self.q_net = QNetwork(self.action_dim).to(self.device)
        self.target_net = QNetwork(self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_capacity)

        # Frame stacker interno (usado en predict / sim / render)
        self._frame_stack = FrameStack()

    # ── policy ────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(t)
            return int(q_values.argmax(dim=1).item())

    def predict(self, obs, *, deterministic=True):
        if len(self._frame_stack._frames) < self._frame_stack.k:
            state = self._frame_stack.reset(obs)
        else:
            state = self._frame_stack.step(obs)
        return self.select_action(state, deterministic=deterministic), None

    def reset_predict(self):
        """Llamar al inicio de cada episodio para reiniciar el frame stack."""
        self._frame_stack._frames.clear()

    # ── learning step ─────────────────────────────────────────────────

    def _learn(self) -> float:
        """Sample a mini-batch from the buffer and perform one gradient step.

        Returns the batch loss value.
        """
        if len(self.buffer) < self.min_buffer:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states_u8, actions, rewards, next_states_u8, dones = zip(*batch)

        # De uint8 → float32 normalizado en [0, 1]
        states_t      = torch.FloatTensor(np.array(states_u8,      dtype=np.float32) / 255.0).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states_u8, dtype=np.float32) / 255.0).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a) from the online network
        current_q = self.q_net(states_t).gather(1, actions_t)

        # Double DQN: q_net elige la mejor acción, target_net la evalúa
        with torch.no_grad():
            best_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, best_actions)

        # Bellman target: r + γ * max Q(s', a') * (1 - done)
        target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # ── training loop ─────────────────────────────────────────────────

    def train(self, total_episodes: int = 10_000, log_interval: int = 50) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []
        frame_stack = FrameStack()

        for episode in range(1, total_episodes + 1):
            raw_obs, _ = env.reset()
            state = frame_stack.reset(raw_obs)
            total_reward = 0.0
            done = False

            # Environment loop
            while not done:
                # Select action
                action = self.select_action(state)
                # Take action
                raw_next, reward, terminated, truncated, _ = env.step(action)
                # Preprocess next state
                next_state = frame_stack.step(raw_next)
                # Update state
                done = terminated or truncated
                # Update buffer
                self.buffer.push(state, action, float(reward), next_state, done)

                self.total_steps += 1

                # Update Q-network
                if self.total_steps % self.learn_every == 0:
                    self._learn()

                # Sync target network
                if self.total_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                # Update state and total reward
                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % log_interval == 0:
                avg  = np.mean(rewards_history[-log_interval:])
                best = max(rewards_history)
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Best Reward: {best:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer):,}"
                )

        env.close()
        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "arch": ARCH_TAG,
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "total_steps": self.total_steps,
            "env_id": self.env_id,
            "action_dim": self.action_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "learn_every": self.learn_every,
            "min_buffer": self.min_buffer,
        }
        torch.save(data, path)
        print(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)

        if data.get("arch") != ARCH_TAG:
            raise ValueError(
                f"El archivo {path} usa la arquitectura '{data.get('arch')}', "
                f"esperada '{ARCH_TAG}'. Bórralo con: rlgames delete dqn"
            )

        agent = cls(
            data["env_id"],
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
            batch_size=data["batch_size"],
            target_update_freq=data["target_update_freq"],
            learn_every=data.get("learn_every", 4),
            min_buffer=data.get("min_buffer", 10_000),
        )
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["target_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.training_episodes = data["training_episodes"]
        agent.total_steps = data.get("total_steps", 0)
        return agent

    def info(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"DQN agent for {self.env_id}\n"
            f"  Episodes trained  : {self.training_episodes}\n"
            f"  Total steps       : {self.total_steps:,}\n"
            f"  Network params    : {params:,}\n"
            f"  Epsilon           : {self.epsilon:.4f}\n"
            f"  LR / Gamma        : {self.lr} / {self.gamma}\n"
            f"  Batch size        : {self.batch_size}\n"
            f"  Learn every       : {self.learn_every} steps\n"
            f"  Target update     : every {self.target_update_freq} steps\n"
            f"  Min buffer        : {self.min_buffer}\n"
            f"  Device            : {self.device}"
        )
