import pandas as pd
from typing import List, Literal


class MeanEncoding:
    def __init__(self, categorical_features: List[str], target_name: str):
        self.categorical_features = categorical_features
        self.target_name = target_name

    def fit(self, df_train: pd.DataFrame):
        self.mean_encoding = (
            df_train.groupby(self.categorical_features)[self.target_name]
            .mean()
            .to_frame()
            .reset_index()
        )

    def transform(self, df: pd.DataFrame, mode: Literal["train", "test"]):
        if mode == "train":
            cumsum = (
                df.groupby(self.categorical_features)[self.target_name].cumsum()
                - df[self.target_name]
            )
            cumcount = df.groupby(self.categorical_features).cumcount()
            return cumsum / cumcount
        elif mode == "test":
            assert hasattr(
                self, "mean_encoding"
            ), "Model hasn't been fitted yet. Please fit it before predicting"
            df_subset = df[self.categorical_features].copy()
            df_subset = df_subset.merge(
                self.mean_encoding, how="left", on=self.categorical_features
            )
            return df_subset[self.target_name]
        else:
            raise Exception('mode must be one of ["train","test"]')


class PreviousFeatures:
    def __init__(
        self,
        feat_name: str,
        unique_identifiers: List[str],
        k: int,
        month_feat_name: str = "date_block_num",
    ):
        self.month_feat_name = month_feat_name
        self.feat_name = feat_name
        self.unique_identifiers = unique_identifiers
        self.k = k

    def fit(self, df: pd.DataFrame):
        df_shifted = df[
            [self.month_feat_name, self.feat_name] + self.unique_identifiers
        ].copy()
        df_shifted[self.month_feat_name] = df_shifted[self.month_feat_name] + self.k
        self.new_feat_name = self.feat_name + "_-" + str(self.k)
        df_shifted = df_shifted.rename(columns={self.feat_name: self.new_feat_name})
        self.df_shifted = df_shifted

    def transform(self, df: pd.DataFrame):
        assert hasattr(
            self, "df_shifted"
        ), "Model hasn't been fitted yet. Please fit it before predicting"
        df_subset = df[self.unique_identifiers + [self.month_feat_name]].copy()
        df_subset = df_subset.merge(
            self.df_shifted,
            how="left",
            on=self.unique_identifiers + [self.month_feat_name],
        )
        return df_subset[self.new_feat_name]


class PastAggregations:
    def __init__(
        self,
        groupby_cols: List[str],
        target_name: str,
        aggregations: List[str] = ["min", "max", "mean"],
        k=1,
        month_feat_name: str = "date_block_num",
    ):
        self.groupby_cols = groupby_cols
        self.target_name = target_name
        self.aggregations = aggregations
        self.k = k
        self.month_feat_name = month_feat_name

    def fit(
        self,
        raw_sales: pd.DataFrame,
    ):
        agg_df = (
            raw_sales.groupby(self.groupby_cols)[self.target_name]
            .agg(self.aggregations)
            .reset_index()
        )
        post_fix = "-".join(
            [x for x in self.groupby_cols if x not in self.month_feat_name]
        )
        rename_dict = {
            agg: self.target_name + "_" + agg + "-by-" + post_fix + "-" + str(self.k)
            for agg in self.aggregations
        }
        agg_df = agg_df.rename(columns=rename_dict)
        agg_df[self.month_feat_name] += self.k
        self.new_columns = rename_dict.values()
        self.agg_df = agg_df

    def transform(
        self,
        df: pd.DataFrame,
    ):
        df = df.merge(self.agg_df, how="left", on=self.groupby_cols)

        return df[self.new_columns]


def compare_sets(set1, set2):
    union = set1.union(set2)
    set_1_unique = set1.difference(set2)
    set_2_unique = set2.difference(set1)

    print(f"set 1 size: {len(set1)}")
    print(f"set 2 size: {len(set2)}")
    print(f"Union size: {len(union)}")
    print(f"set 1 unique elements: {len(set_1_unique)}")
    print(f"set 2 unique elements: {len(set_2_unique)}")
    return
