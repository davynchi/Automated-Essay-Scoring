import pandas as pd

from ..common.basic_definitions import modify_texts
from ..common.constants import INPUT_DIR_INFERENCE


def load_data():
    test = pd.read_csv(f"{INPUT_DIR_INFERENCE}test.csv")
    submission = pd.read_csv(f"{INPUT_DIR_INFERENCE}sample_submission.csv")
    modify_texts(test["full_text"])
    return test, submission


# print(f"test.shape: {test.shape}")
# display(test.head())
# print(f"submission.shape: {submission.shape}")
# display(submission.head())
