import torch
from torch.utils.data import Dataset


class LALDataset(Dataset):
    def __init__(self, cfg, df, tokenizer, is_train):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.texts = df["full_text"].values
        self.is_train = is_train
        if self.is_train:
            self.labels = df[cfg.base.target_cols].values
            self.labels2 = df[cfg.base.modif_target_cols].values

    def _prepare_input(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.cfg.max_len,
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self._prepare_input(self.texts[item])
        if self.is_train:
            label = torch.tensor(self.labels[item], dtype=torch.float)
            label2 = torch.tensor(self.labels2[item], dtype=torch.float)
            return inputs, label, label2
        else:
            return inputs


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, _ in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs
