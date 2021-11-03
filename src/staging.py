import pandas as pd
from itertools import product
import numpy as np


def create_dataset(df: pd.DataFrame) -> pd.DataFrame:

    index_cols = ["shop_id", "item_id", "date_block_num"]

    # For every month we create a grid from all shops/items combinations from that month
    grid = []
    for block_num in df["date_block_num"].unique():
        cur_shops = df[df["date_block_num"] == block_num]["shop_id"].unique()
        cur_items = df[df["date_block_num"] == block_num]["item_id"].unique()
        grid.append(
            np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype="int32")
        )

    # turn the grid into pandas dataframe
    grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

    # get aggregated values for (shop_id, item_id, month)
    gb = df.groupby(index_cols, as_index=False)["item_cnt_day"].agg({"target": "sum"})

    # join aggregated data to the grid
    all_data = pd.merge(grid, gb, how="left", on=index_cols).fillna(0)
    # sort the data
    all_data.sort_values(["date_block_num", "shop_id", "item_id"], inplace=True)

    return all_data
