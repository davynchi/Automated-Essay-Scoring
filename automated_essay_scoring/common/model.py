from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .constants import (
    BASE_PATH_TO_SAVE_FINETUNED,
    CHECKPOINT_POSTFIX,
    DEVICE,
    MODEL_UNIT_CONFIG_NAME,
    NAMES_OF_MODELS,
)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CustomModel_mean_pooling(nn.Module):
    def __init__(self, cfg, checkpoints_names=None, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                NAMES_OF_MODELS[cfg.model_key], output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path, weights_only=False)

        self.config.update(cfg.base.model_config)

        if checkpoints_names is None:
            if pretrained:
                self.model = AutoModel.from_pretrained(
                    NAMES_OF_MODELS[cfg.model_key], config=self.config
                )
            else:
                self.model = AutoModel.from_config(self.config)
        else:
            self.model = AutoModel.from_pretrained(
                BASE_PATH_TO_SAVE_FINETUNED
                / (NAMES_OF_MODELS[cfg.model_key].replace("/", "-") + CHECKPOINT_POSTFIX)
                / checkpoints_names[cfg.model_key]
            )

        self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.base.target_size)
        self._init_weights(self.fc)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs["attention_mask"])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        feature = self.layer_norm1(feature)
        output = self.fc(feature)
        return output


class CustomModel_attention(nn.Module):
    def __init__(self, cfg, checkpoints_names=None, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                NAMES_OF_MODELS[cfg.model_key], output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path, weights_only=False)

        self.config.update(cfg.base.model_config)

        if checkpoints_names is None:
            if pretrained:
                self.model = AutoModel.from_pretrained(
                    NAMES_OF_MODELS[cfg.model_key], config=self.config
                )
            else:
                self.model = AutoModel.from_config(self.config)
        else:
            self.model = AutoModel.from_pretrained(
                BASE_PATH_TO_SAVE_FINETUNED
                / (NAMES_OF_MODELS[cfg.model_key].replace("/", "-") + CHECKPOINT_POSTFIX)
                / checkpoints_names[cfg.model_key]
            )

        self.model.gradient_checkpointing_enable()
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.base.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


class CustomModel_lstm(nn.Module):
    def __init__(self, cfg, checkpoints_names, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                NAMES_OF_MODELS[cfg.model_key], output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path, weights_only=False)

        self.config.update(cfg.base.model_config)

        if checkpoints_names is None:
            if pretrained:
                self.model = AutoModel.from_pretrained(
                    NAMES_OF_MODELS[cfg.model_key], config=self.config
                )
            else:
                self.model = AutoModel.from_config(self.config)
        else:
            self.model = AutoModel.from_pretrained(
                BASE_PATH_TO_SAVE_FINETUNED
                / (NAMES_OF_MODELS[cfg.model_key].replace("/", "-") + CHECKPOINT_POSTFIX)
                / checkpoints_names[cfg.model_key]
            )

        self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.base.target_size)
        self._init_weights(self.fc)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size)
        self.lstm = nn.LSTM(
            self.config.hidden_size,
            (self.config.hidden_size) // 2,
            num_layers=2,
            dropout=self.config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True,
        )
        self._init_weights(self.lstm)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature, _ = self.lstm(last_hidden_states)
        feature = self.pool(feature, inputs["attention_mask"])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        feature = self.layer_norm1(feature)
        output = self.fc(feature)
        return output


def create_model(cfg, fold, checkpoints_names=None):
    if checkpoints_names is None:
        path_to_config = Path(cfg.path) / MODEL_UNIT_CONFIG_NAME
        if cfg.head == "mean_pooling":
            model = CustomModel_mean_pooling(
                cfg, checkpoints_names, config_path=path_to_config, pretrained=False
            )
        elif cfg.head == "attention":
            model = CustomModel_attention(
                cfg, checkpoints_names, config_path=path_to_config, pretrained=False
            )
        elif cfg.head == "lstm":
            model = CustomModel_lstm(
                cfg, checkpoints_names, config_path=path_to_config, pretrained=False
            )
        state = torch.load(
            Path(cfg.path)
            / f"{NAMES_OF_MODELS[cfg.model_key].replace('/', '-')}_fold{fold}_best.pth",
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        model.load_state_dict(state["model"])
    else:
        if cfg.head == "mean_pooling":
            model = CustomModel_mean_pooling(
                cfg, checkpoints_names, config_path=None, pretrained=True
            )
        elif cfg.head == "attention":
            model = CustomModel_attention(
                cfg, checkpoints_names, config_path=None, pretrained=True
            )
        elif cfg.head == "lstm":
            model = CustomModel_lstm(
                cfg, checkpoints_names, config_path=None, pretrained=True
            )
        torch.save(model.config, Path(cfg.path) / "config.pth")
        model.to(DEVICE)

    return model
