from nn import MLP
from envs import HalfCheetahDirEnv
from utils import ReplayBuffer
import torch.optim as O
import torch.distributions as D
import hydra
from hydra.utils import get_original_cwd
import json
from collections import namedtuple
import pickle
import torch
from typing import List
import higher
from itertools import count


from utils import Experience
from losses import policy_loss_on_batch, vf_loss_on_batch


def rollout_policy(policy: MLP, env, render: bool = False) -> List[Experience]:
    trajectory = []
    state = env.reset()
    if render:
        env.render()
    done = False
    total_reward = 0
    episode_t = 0
    success = False
    policy.eval()
    current_device = list(policy.parameters())[-1].device
    while not done:
        with torch.no_grad():
            action_sigma = 0.2
            action = policy(torch.tensor(state).to(current_device).float()).squeeze()

            action_dist = D.Normal(action, torch.empty_like(action).fill_(action_sigma))
            log_prob = action_dist.log_prob(action).to("cpu").numpy().sum()

            np_action = action.squeeze().cpu().numpy()
            np_action = np_action.clip(min=env.action_space.low, max=env.action_space.high)

        next_state, reward, done, info_dict = env.step(np_action)

        if "success" in info_dict and info_dict["success"]:
            success = True

        if render:
            env.render()
        trajectory.append(Experience(state, np_action, next_state, reward, done))
        state = next_state
        total_reward += reward
        episode_t += 1
        if episode_t >= env._max_episode_steps or done:
            break

    return trajectory, total_reward, success


def build_networks_and_buffers(args, env, task_config):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy_head = [32, 1] if args.advantage_head_coef is not None else None
    policy = MLP(
        [obs_dim] + [args.net_width] * args.net_depth + [action_dim],
        final_activation=torch.tanh,
        extra_head_layers=policy_head,
        w_linear=args.weight_transform,
    ).to(args.device)

    vf = MLP(
        [obs_dim] + [args.net_width] * args.net_depth + [1],
        w_linear=args.weight_transform,
    ).to(args.device)

    buffer_paths = [
        task_config.train_buffer_paths.format(idx) for idx in task_config.train_tasks
    ]

    buffers = [
        ReplayBuffer(
            args.inner_buffer_size,
            obs_dim,
            action_dim,
            discount_factor=0.99,
            immutable=True,
            load_from=buffer_paths[i],
        )
        for i, task in enumerate(task_config.train_tasks)
    ]

    return policy, vf, buffers


def get_env(args, task_config):
    tasks = []
    for task_idx in range(task_config.total_tasks):
        with open(task_config.task_paths.format(task_idx), "rb") as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f"Unexpected task info: {task_info}"
            tasks.append(task_info[0])

    if args.advantage_head_coef == 0:
        args.advantage_head_coef = None

    return HalfCheetahDirEnv(tasks, include_goal=False)


def get_opts_and_lrs(args, policy, vf):
    policy_opt = O.Adam(policy.parameters(), lr=args.outer_policy_lr)
    vf_opt = O.Adam(vf.parameters(), lr=args.outer_value_lr)
    policy_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_policy_lr).to(args.device))
        for p in policy.parameters()
    ]
    vf_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_value_lr).to(args.device))
        for p in vf.parameters()
    ]

    return policy_opt, vf_opt, policy_lrs, vf_lrs


@hydra.main(config_path="config", config_name="config.yaml")
def run(args):
    with open(f"{get_original_cwd()}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )

    env = get_env(args, task_config)
    policy, vf, task_buffers = build_networks_and_buffers(args, env, task_config)
    policy_opt, vf_opt, policy_lrs, vf_lrs = get_opts_and_lrs(args, policy, vf)

    for train_step_idx in count(start=1):
        for i, (train_task_idx, task_buffer) in enumerate(
            zip(task_config.train_tasks, task_buffers)
        ):
            inner_batch = task_buffer.sample(
                args.inner_batch_size, return_dict=True, device=args.device
            )
            outer_batch = task_buffer.sample(
                args.outer_batch_size, return_dict=True, device=args.device
            )

            # Adapt value function
            opt = O.SGD([{"params": p, "lr": None} for p in vf.parameters()])
            with higher.innerloop_ctx(
                vf, opt, override={"lr": vf_lrs}, copy_initial_weights=False
            ) as (f_vf, diff_value_opt):
                loss = vf_loss_on_batch(f_vf, inner_batch, inner=True)
                diff_value_opt.step(loss)

                meta_vf_loss = vf_loss_on_batch(f_vf, outer_batch)
                total_vf_loss = meta_vf_loss / len(task_config.train_tasks)
                total_vf_loss.backward()

            # Adapt policy using adapted value function
            adapted_vf = f_vf
            opt = O.SGD([{"params": p, "lr": None} for p in policy.parameters()])
            with higher.innerloop_ctx(
                policy, opt, override={"lr": policy_lrs}, copy_initial_weights=False
            ) as (f_policy, diff_policy_opt):
                loss = policy_loss_on_batch(
                    f_policy,
                    adapted_vf,
                    inner_batch,
                    args.advantage_head_coef,
                    inner=True,
                )

                diff_policy_opt.step(loss)

                meta_policy_loss = policy_loss_on_batch(
                    f_policy, adapted_vf, outer_batch, args.advantage_head_coef
                )

                (meta_policy_loss / len(task_config.train_tasks)).backward()

                # Sample adapted policy trajectory, add to replay buffer i [L12]
                if train_step_idx % args.rollout_interval == 0:
                    adapted_trajectory, adapted_reward, success = rollout_policy(
                        f_policy, env
                    )


if __name__ == "__main__":
    run()
