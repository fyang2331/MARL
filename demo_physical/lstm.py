import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.policies import ActorCriticPolicy


# 定义自定义的LSTM网络

class CustomLSTM(nn.Module):

    def __init__(self, obs_space, action_space, features_dim=64):
        super().__init__()
        n_input_channels = obs_space.shape[0]
        n_actions = action_space.n
        self.features_extractor = BaseFeaturesExtractor(
            obs_space,
            features_dim=features_dim
        )

        self.lstm = nn.LSTM(features_dim, features_dim)

        self.actor = nn.Linear(features_dim, n_actions)

        self.critic = nn.Linear(features_dim, 1)

    def forward(self, obs, h_in=None, c_in=None, mask=None):
        features = self.features_extractor(obs)

        features = features.view((-1,) + features.shape[-3:])

        features, (h_out, c_out) = self.lstm(features, (h_in, c_in))

        action_logits = self.actor(features)

        values = self.critic(features)

        return action_logits, values, (h_out, c_out)


# 定义自定义的Policy类，包含自定义的LSTM网络

class CustomPolicy(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs)
        self.features_extractor = CustomLSTM(observation_space, action_space)

