import pandas as pd
from typing import List


def mean_encode(
    df: pd.DataFrame, categorical_features: List[str], target_name: str
) -> pd.Series:
    cumsum = df.groupby(categorical_features)[target_name].cumsum() - df[target_name]
    cumcount = df.groupby(categorical_features).cumcount()
    return cumsum / cumcount


def add_prev_k_months_feature(
    df: pd.DataFrame,
    k_list: List[int],
    feat_name: str,
    unique_identifiers: List[str],
    month_feat_name: str = "date_block_num",
):
    for k in k_list:
        df = _add_prev_k_months_feature_single_k(
            df, k, feat_name, unique_identifiers, month_feat_name
        )
    return df


def _add_prev_k_months_feature_single_k(
    df: pd.DataFrame,
    k: int,
    feat_name: str,
    unique_identifiers: List[str],
    month_feat_name: str = "date_block_num",
):
    df_shifted = df[[month_feat_name, feat_name] + unique_identifiers].copy()
    df_shifted[month_feat_name] = df_shifted[month_feat_name] - k
    new_feat_name = feat_name + "_-" + str(k)
    df_shifted = df_shifted.rename(columns={feat_name: new_feat_name})
    df = df.merge(df_shifted, how="left", on=unique_identifiers + [month_feat_name])
    print(
        f"{100*round(df[new_feat_name].isnull().mean(),3)}% of {new_feat_name} are null"
    )
    return df


def add_aggregations(
    df: pd.DataFrame,
    raw_sales: pd.DataFrame,
    groupby_cols: List[str],
    target_name: str,
    aggregations: List[str] = ["min", "max", "mean"],
    k=1,
    month_feat_name: str = "date_block_num",
) -> pd.DataFrame:
    agg_df = (
        raw_sales.groupby(groupby_cols)[target_name].agg(aggregations).reset_index()
    )
    post_fix = "-".join([x for x in groupby_cols if x not in month_feat_name])
    rename_dict = {
        agg: target_name + "_" + agg + "-by-" + post_fix + "-" + str(k)
        for agg in aggregations
    }
    agg_df = agg_df.rename(columns=rename_dict)
    agg_df[month_feat_name] -= k
    df = df.merge(agg_df, how="left", on=groupby_cols)

    return df
