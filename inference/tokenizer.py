from transformers import AutoTokenizer

from .make_cfg_list import CFG_LIST


def tokenize_text(test):
    for i in range(len(CFG_LIST)):
        tokenizer = AutoTokenizer.from_pretrained(CFG_LIST[i].path)
        tokenizer.add_special_tokens({"additional_special_tokens": ["[BR]"]})
        CFG_LIST[i].tokenizer = tokenizer

    def text_encode(text):
        return len(tokenizer.encode(text))

    test["length"] = test["full_text"].map(text_encode)
    test = test.sort_values("length", ascending=True).reset_index(drop=True)
