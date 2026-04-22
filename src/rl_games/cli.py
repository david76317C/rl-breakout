# Dependencias

import ale_py
import argparse
import numpy     as np
import gymnasium as gym

from importlib.metadata import version
from pathlib            import Path

gym.register_envs(ale_py)

# Parámetros

ENV_ID        = 'ALE/Breakout-v5'
SAVE_DIR      = Path('saves')
AGENT_CHOICES = ('dqn',)
VERSION       = version('rl_games')

# Funciones Generales

def _save_path(agent_type: str) -> Path:
    return SAVE_DIR / "dqn_breakout.pt"

def _load_agent(agent_type: str):
    from rl_games.agents.dqn import DQNAgent
    return DQNAgent.load(_save_path(agent_type))

# Comandos

ACTION_NAMES = {0: 'NOOP', 1: 'FIRE', 2: 'Derecha', 3: 'Izquierda'}

def _fmt_action(action: int) -> str:
    name = ACTION_NAMES.get(action, "?")
    return f"{action} ({name})"

def cmd_inspect(args: argparse.Namespace) -> None:

    env_id = args.env or ENV_ID
    env    = gym.make(id=env_id)

    print(f"Ambiente                    : {env_id}")
    print(f"Espacio de observaciones    : {env.observation_space}")
    print(f"Espacio de acciones         : {env.action_space}")
    print(f"Acciones disponibles        : {ACTION_NAMES}")

    n = args.steps
    obs, info = env.reset()

    print(f"\n-- Política aleatoria ({n} pasos) --\n")
    for step in range(1, n + 1):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f" Paso {step} | Recompensa {reward} | ¿Terminado? {done} | Acción tomada: {_fmt_action(action)} ")

        obs = next_obs
        if done:
            print("!!! Episodio terminado. Reiniciando... !!!\n")
            obs, info = env.reset()
            print(f'Vuelve a la posición inicial')
    env.close()

def cmd_init(args: argparse.Namespace) -> None:

    path = _save_path(args.agent)
    if path.exists():
        print(f"Ya existe un archivo de guardado en {path}. Ejecuta rlgames delete {args.agent} primero.")
        return

    from rl_games.agents.dqn import DQNAgent
    agent = DQNAgent(ENV_ID)
    agent.save(path)
    print(f"Agente dqn inicializado y guardado correctamente")

def cmd_train(args: argparse.Namespace) -> None:

    path = _save_path(args.agent)
    from rl_games.agents.dqn import DQNAgent
    agent = DQNAgent.load(path) if path.exists() else DQNAgent(ENV_ID)
    agent.train(total_episodes=args.episodes)
    agent.save(path)
    print("Entrenamiento Completado")

def cmd_delete(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    if path.exists():
        path.unlink()
        print(f"Eliminado {path}")
    else:
        print(f"Ningún archivo guardado encontrado en {path}")

def cmd_load(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    if not path.exists():
        print(f"Ningún archivo guardado encontrado en {path}")
        return

    agent = _load_agent(args.agent)
    print(agent.info())

    if args.eval:
        print("\n Evaluando (10 episodios) ...")
        env = gym.make(ENV_ID)
        rewards = []
        for _ in range(10):
            obs, _ = env.reset()
            agent.reset_predict()
            done, total = False, 0.0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
            rewards.append(total)
        env.close()
        print(f" Recompensa promedio: {np.mean(rewards):.2f}")

def cmd_sim(args: argparse.Namespace) -> None:

    path = _save_path(args.agent)
    if not path.exists():
        print(f"Ningún archivo guardado encontrado en {path}")
        return

    agent = _load_agent(args.agent)
    env   = gym.make(ENV_ID)

    all_rewards: list[float] = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        agent.reset_predict()
        done = False
        total_reward = 0.0
        step = 0

        print(f"== Episodio {ep}/{args.episodes} ==\n")

        limit = args.steps

        while not done:
            step     += 1
            action, _ = agent.predict(obs, deterministic=True)
            action    = int(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if limit is None or step <= limit:
                print(f" Paso {step} | Recompensa {reward} | Recompensa Total {total_reward} | ¿Terminado? {done} | Acción tomada: {_fmt_action(action)} ")

            obs = next_obs

        if limit is not None and step > limit:
            print(f"  ... (Quedan {step - limit} pasos) ...")

        outcome = "Tiempo agotado" if truncated else "Game Over"
        print(f"\n  Resultado: {outcome} | Pasos: {step} | Recompensa Total {total_reward} \n")
        all_rewards.append(total_reward)

    env.close()

    if len(all_rewards) > 1:
        print(
            f"Resumen después de {len(all_rewards)} episodios: "
            f"Recompensa promedio: {np.mean(all_rewards):.2f}"
        )

def cmd_render(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    if not path.exists():
        print(f"Ningún archivo guardado encontrado en {path}")
        return

    agent = _load_agent(args.agent)
    env = gym.make(ENV_ID, render_mode="human")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        agent.reset_predict()
        done   = False
        total_reward = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episodio {ep}/{args.episodes} | Recompensa: {total_reward:.2f}")

    env.close()

def cmd_version(_args: argparse.Namespace) -> None:
    print(f"rl_games {VERSION}")

def cmd_list(_args: argparse.Namespace) -> None:
    print("Agentes disponibles:\n")
    for agent in AGENT_CHOICES:
        path = _save_path(agent)
        status = "Guardado" if path.exists() else "No guardado"
        print(f"  {agent:<14} [{status}]  {path}")

# ── argument parser ──────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlgames",
        description="Train and evaluate RL agents on ALE/Breakout-v5",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # version
    p = sub.add_parser("version", help="Muestra la versión de la paquetería")
    p.set_defaults(func=cmd_version)

    # list
    p = sub.add_parser("list", help="Lista de agentes disponibles y su status de guardado")
    p.set_defaults(func=cmd_list)

    # inspect
    p = sub.add_parser(
        "inspect",
        help="Inspección del ambiente: muestra el espacio estado/acción y una muestra de transiciones",
    )
    p.add_argument("--env", type=str, default=None, help=f"Gymnasium env ID (default: {ENV_ID})")
    p.add_argument("--steps", type=int, default=5, help="Pasos aleatorios para el muestreo (default: 5)")
    p.set_defaults(func=cmd_inspect)

    # init
    p = sub.add_parser("init", help="Inicializando un nuevo agente y guardándolo")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.set_defaults(func=cmd_init)

    # train
    p = sub.add_parser("train", help="Entrenando un nuevo agente y guardando sus resultados")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=10_000, help="Episodios de entrenamiento (default: 10k)")
    p.set_defaults(func=cmd_train)

    # delete
    p = sub.add_parser("delete", help="Eliminar el agente guardado")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.set_defaults(func=cmd_delete)

    # load
    p = sub.add_parser("load", help="Cargando un agente guardado y mostrando su información")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--eval", action="store_true", help="Evaluación rápida de 10 episodios")
    p.set_defaults(func=cmd_load)

    # sim
    p = sub.add_parser("sim", help="Simulando episodios con agente entrenado")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=1, help="Número de episodios para simular")
    p.add_argument("--steps", type=int, default=None, help="Límite de los primeros N pasos por episodio (default: mostrar todo)")
    p.add_argument("--verbose", action="store_true", help="Imprimir cada paso con su vector total de estados")
    p.set_defaults(func=cmd_sim)

    # render
    p = sub.add_parser("render", help="Renderizar episodios usando agente guardado (Ventana Gráfica)")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=1, help="Número de episodios para renderizar")
    p.set_defaults(func=cmd_render)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
