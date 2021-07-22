import torch
import torch.distributions as D
import torch.nn.functional as F

def vf_loss_on_batch(vf, batch, inner: bool = False):
    value_estimates = vf(batch["obs"])
    targets = batch["mc_rewards"]

    return (value_estimates - targets).pow(2).mean()


def policy_loss_on_batch(policy, vf, batch, adv_coef: float, inner: bool = False):
    with torch.no_grad():
        value_estimates = vf(batch["obs"])
        action_value_estimates = batch["mc_rewards"]

        advantages = (action_value_estimates - value_estimates).squeeze(-1)
        normalized_advantages = (advantages - advantages.mean()) / advantages.std()
        weights = normalized_advantages.clamp(max=3).exp()

    original_action = batch["actions"]
    action_mu, advantage_prediction = policy(batch["obs"], batch["actions"])
    action_sigma = torch.empty_like(action_mu).fill_(0.2)
    action_distribution = D.Normal(action_mu, action_sigma)
    action_log_probs = action_distribution.log_prob(batch["actions"]).sum(-1)

    losses = -(action_log_probs * weights)

    adv_prediction_loss = None
    if inner:
        adv_prediction_loss = adv_coef *  (advantage_prediction.squeeze() - advantages) ** 2
        losses = losses + adv_prediction_loss
        adv_prediction_loss = adv_prediction_loss.mean()

    return losses.mean()
