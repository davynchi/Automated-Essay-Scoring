from transformers import AutoTokenizer

from ..common.cfg import CFG
from ..common.constants import OUTPUT_DIR


def add_tokenizer_to_cfg():
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(OUTPUT_DIR)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[BR]"]})
    CFG.tokenizer = tokenizer
