from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..common.constants import (
    BEST_CHECKPOINT_POSTFIX,
    MODEL_UNIT_CONFIG_NAME,
    NAMES_OF_MODELS,
    OUTPUT_DIR_FINETUNED,
)


class MeanPooling(nn.Module):
    """Mean pooling layer that averages token embeddings using attention mask.

    Computes the mean of the last hidden states over the sequence length,
    taking the attention mask into account.

    Methods:
        forward(last_hidden_state, attention_mask): Return mean pooled embeddings.
    """

    def __init__(self):
        """Initialize the MeanPooling module."""
        super().__init__()

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean-pooled embeddings.

        Args:
            last_hidden_state (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (torch.Tensor): Bool or int mask of shape (batch_size, seq_len),
                where 1 indicates valid tokens.

        Returns:
            torch.Tensor: Mean-pooled embeddings of shape (batch_size, hidden_size).
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class CustomModel(nn.Module):
    """Base class for custom essay-scoring models.

    Handles loading or initializing a Transformer backbone, saving its config,
    and provides common forward logic. Subclasses must implement `feature`.

    Args:
        cfg: Configuration object with attributes:
            - `model_key` (str): Key to lookup pre-trained model name.
            - `base.model_config` (dict): Overrides for model config.
            - `path` (str or Path): Output path for saving config.
        load_from_existed (bool): If True, load model weights from an existing checkpoint.
        path_to_finetuned_models (Path | None): Directory with fine-tuned weights.
        config_path (Path | None): Path to a saved model config to load.
        pretrained (bool): If True, load pretrained weights when `load_from_existed`.

    Attributes:
        config: Model configuration (`transformers.PretrainedConfig`).
        model: Transformer backbone (`transformers.PreTrainedModel`).
    """

    def __init__(
        self,
        cfg,
        load_from_existed: bool,
        path_to_finetuned_models: Path | None = None,
        config_path: Path | None = None,
        pretrained: bool = False,
    ):
        super().__init__()
        # Resolve config path
        if config_path is not None and not Path(config_path).is_file():
            config_path = None

        # Load or create config
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                NAMES_OF_MODELS[cfg.model_key], output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path, weights_only=False)
        self.config.update(cfg.base.model_config)

        # Load model weights
        if load_from_existed:
            if pretrained:
                self.model = AutoModel.from_pretrained(
                    NAMES_OF_MODELS[cfg.model_key], config=self.config
                )
            else:
                self.model = AutoModel.from_config(self.config)
        else:
            base_dir = (
                Path(path_to_finetuned_models)
                if path_to_finetuned_models
                else OUTPUT_DIR_FINETUNED
            )
            ckpt_dir = base_dir / (
                NAMES_OF_MODELS[cfg.model_key].replace("/", "-") + BEST_CHECKPOINT_POSTFIX
            )
            self.model = AutoModel.from_pretrained(ckpt_dir)

        # Save config if newly created
        if config_path is None:
            save_path = Path(cfg.path) / MODEL_UNIT_CONFIG_NAME
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.config, save_path)

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights using Normal distribution or zeros.

        Args:
            module (nn.Module): Module to initialize. Supports Linear, Embedding, LayerNorm.
        """
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

    @abstractmethod
    def feature(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from inputs using the backbone and pooling/attention.

        Args:
            inputs (dict[str, torch.Tensor]): Tokenized input batch.

        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, hidden_size).
        """
        ...

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute final model output from inputs.

        Applies `feature`, optional layer norm, and the final linear layer.

        Args:
            inputs (dict[str, torch.Tensor]): Tokenized input batch.

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, target_size).
        """
        feat = self.feature(inputs)
        if hasattr(self, "layer_norm1"):
            feat = self.layer_norm1(feat)
        return self.fc(feat)


class CustomModelMeanPooling(CustomModel):
    """Essay scoring model using mean-pooling over token embeddings."""

    def __init__(
        self,
        cfg,
        load_from_existed: bool,
        path_to_finetuned_models: Path | None = None,
        config_path: Path | None = None,
        pretrained: bool = False,
    ):
        """Initialize using mean-pooling head.

        Args and attributes see `CustomModel`.
        """
        super().__init__(
            cfg, load_from_existed, path_to_finetuned_models, config_path, pretrained
        )
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, cfg.base.target_size)
        self._init_weights(self.fc)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size)

    def feature(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract mean-pooled features from the backbone.

        Args:
            inputs (dict[str, torch.Tensor]): Tokenized input batch.

        Returns:
            torch.Tensor: Mean-pooled features of shape (batch_size, hidden_size).
        """
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # type: ignore[attr-defined]
        return self.pool(last_hidden, inputs["attention_mask"])


class CustomModelAttention(CustomModel):
    """Essay scoring model using learnable attention over token embeddings."""

    def __init__(
        self,
        cfg,
        load_from_existed: bool,
        path_to_finetuned_models: Path | None = None,
        config_path: Path | None = None,
        pretrained: bool = False,
    ):
        """Initialize using attention-based pooling head.

        Args and attributes see `CustomModel`.
        """
        super().__init__(
            cfg, load_from_existed, path_to_finetuned_models, config_path, pretrained
        )
        self.fc = nn.Linear(self.config.hidden_size, cfg.base.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )
        self._init_weights(self.attention)

    def feature(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract attention-pooled features from the backbone.

        Args:
            inputs (dict[str, torch.Tensor]): Tokenized input batch.

        Returns:
            torch.Tensor: Attention-pooled features of shape (batch_size, hidden_size).
        """
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # type: ignore[attr-defined]
        weights = self.attention(last_hidden)
        return torch.sum(weights * last_hidden, dim=1)


class CustomModelLSTM(CustomModel):
    """Essay scoring model combining LSTM and mean-pooling."""

    def __init__(
        self,
        cfg,
        load_from_existed: bool,
        path_to_finetuned_models: Path | None = None,
        config_path: Path | None = None,
        pretrained: bool = False,
    ):
        """Initialize using LSTM head followed by mean-pooling and layer norm.

        Args and attributes see `CustomModel`.
        """
        super().__init__(
            cfg, load_from_existed, path_to_finetuned_models, config_path, pretrained
        )
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, cfg.base.target_size)
        self._init_weights(self.fc)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size // 2,
            num_layers=2,
            dropout=self.config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True,
        )
        self._init_weights(self.lstm)

    def feature(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract LSTM-pooled features from the backbone.

        Args:
            inputs (dict[str, torch.Tensor]): Tokenized input batch.

        Returns:
            torch.Tensor: LSTM+mean-pooled features of shape (batch_size, hidden_size).
        """
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # type: ignore[attr-defined]
        lstm_out, _ = self.lstm(last_hidden)
        return self.pool(lstm_out, inputs["attention_mask"])


# Mapping from head names to model classes
HEAD_CLASS_MAPPING: dict[str, type[CustomModel]] = {
    "mean_pooling": CustomModelMeanPooling,
    "attention": CustomModelAttention,
    "lstm": CustomModelLSTM,
}
