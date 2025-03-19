import pandas as pd

from ..common.constants import INPUT_DIR_INFERENCE
from ..common.utils import modify_texts


def load_data():
    test = pd.read_csv(f"{INPUT_DIR_INFERENCE}test.csv")
    submission = pd.read_csv(f"{INPUT_DIR_INFERENCE}sample_submission.csv")
    modify_texts(test["full_text"])
    return test, submission


# print(f"test.shape: {test.shape}")
# display(test.head())
# print(f"submission.shape: {submission.shape}")
# display(submission.head())
