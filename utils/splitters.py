import pandas as pd
from pandas import DataFrame
import random as rnd
from sklearn.model_selection import train_test_split


def shuffle_dataframe(data: DataFrame, seed: int = None):
    """This function shuffles the rows of a DataFrame"""
    # return data.sample(frac=1, random_state=seed).reset_index(drop=True)
    if seed is not None:
        rnd.seed(seed)
    indices = list(data.index)
    rnd.shuffle(indices)
    return data.iloc[indices].reset_index(drop=True)


def split_dataframe(data: DataFrame, percentage: float) -> tuple[DataFrame, DataFrame]:
    """This function splits a DataFrame into two parts according to a percentage"""
    percentage = max(0, min(1, percentage))
    split_index = int(len(data) * percentage)
    part1 = data.iloc[:split_index].reset_index(drop=True)
    part2 = data.iloc[split_index:].reset_index(drop=True)
    return part1, part2


def train_test_split_for_unbalanced_dataframes(
    data: DataFrame,
    test_percentage: float,
    column_name: str,
    seed: int = None,
):
    """This function splits a DataFrame into 2 new DataFrames according
    to a percentage"""
    test_percentage = max(0, min(1, test_percentage))
    data = shuffle_dataframe(data, seed)
    grouped = data.groupby(column_name)
    splitted = [split_dataframe(group, 1 - test_percentage) for _, group in grouped]
    train = pd.concat([group[0] for group in splitted]).reset_index(drop=True)
    test = pd.concat([group[1] for group in splitted]).reset_index(drop=True)
    return train, test


def reduce_the_other_category(
    data: DataFrame,
    to_percentage: float,
    seed: int = None,
):
    other_rows = data[data["metaclass"] == "OTHER"].reset_index(drop=True)
    rest = data[data["metaclass"] != "OTHER"].reset_index(drop=True)
    new_other, _ = train_test_split_for_unbalanced_dataframes(
        other_rows,
        test_percentage=1 - to_percentage,
        column_name="classification",
        seed=seed,
    )
    result = pd.concat([new_other, rest]).reset_index(drop=True)
    return result
