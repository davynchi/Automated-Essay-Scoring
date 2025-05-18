import torch
from torch.utils.data import Dataset


class LALDataset(Dataset):
    """Custom Dataset for essay scoring.

    In train mode returns a tuple `(inputs, score, score_s)`,
    in inference mode returns only `inputs`.

    Args:
        cfg: Configuration object with attributes:
            - `base.target_cols` (list of primary target column names)
            - `base.modif_target_cols` (list of secondary target column names)
            - `max_len` (maximum token sequence length).
        df (pd.DataFrame): DataFrame containing at least:
            - `"full_text"` column with essay texts,
            - columns listed in `cfg.base.target_cols` and `cfg.base.modif_target_cols` if `is_train` is True.
        tokenizer: Tokenizer with an `encode_plus` method.
        is_train (bool): Whether the dataset is used for training (labels will be returned).

    Attributes:
        texts (np.ndarray): Array of essay texts.
        labels (np.ndarray): Array of primary labels (only if `is_train`).
        labels2 (np.ndarray): Array of secondary labels (only if `is_train`).
    """

    def __init__(self, cfg, df, tokenizer, is_train: bool):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.df = df
        self.texts = df["full_text"].values
        self.is_train = is_train
        if self.is_train:
            self.labels = df[cfg.base.target_cols].values
            self.labels2 = df[cfg.base.modif_target_cols].values

    def _prepare_input(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize and convert a single text to tensors.

        Uses `tokenizer.encode_plus` with padding/truncation to `cfg.max_len`.

        Args:
            text (str): Raw essay text.

        Returns:
            Dict[str, torch.Tensor]: Mapping of input names
            (e.g. `"input_ids"`, `"attention_mask"`) to long tensors.
        """
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.cfg.max_len,
            truncation=True,
            return_tensors=None,
        )
        return {k: torch.tensor(v, dtype=torch.long) for k, v in inputs.items()}

    def __len__(self) -> int:
        """Total number of examples."""
        return len(self.texts)

    def __getitem__(self, idx: int):
        """Get one example by index.

        Args:
            idx (int): Index of the example.

        Returns:
            If `is_train` is True:
                Tuple[
                    Dict[str, torch.Tensor],  # tokenized inputs
                    torch.FloatTensor,        # primary score(s)
                    torch.FloatTensor         # secondary score(s)
                ]
            If `is_train` is False:
                Dict[str, torch.Tensor]      # tokenized inputs only
        """
        inputs = self._prepare_input(self.texts[idx])
        if self.is_train:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            label2 = torch.tensor(self.labels2[idx], dtype=torch.float)
            return inputs, label, label2
        return inputs


def collate(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Trim batch tensors to the maximum actual sequence length.

    Reduces padding by slicing all tensors in the batch
    to the max sum of `attention_mask` along sequence dimension.

    Args:
        inputs (Dict[str, torch.Tensor]):
            Batch of tensors from `LALDataset._prepare_input`,
            e.g. keys `"input_ids"`, `"attention_mask"`, shape `(batch_size, seq_len)`.

    Returns:
        Dict[str, torch.Tensor]: Same as input but each tensor
        truncated to shape `(batch_size, max_actual_len)`.
    """
    # Determine the longest actual sequence in the batch
    max_len = int(inputs["attention_mask"].sum(dim=1).max().item())
    # Trim each tensor to that length
    return {k: v[:, :max_len] for k, v in inputs.items()}
