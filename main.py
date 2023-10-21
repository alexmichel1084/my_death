import polars as pl
from sklearn.preprocessing import TargetEncoder

df = pl.read_csv("./datasets/bank.csv", separator=';')
print(df.dtypes)
print(df.head)
print(df.schema)
columns = df.columns
# for col in columns:
#    print(df[col].unique())
# print(df["y"].unique())
# print(df["poutcome"].unique())
# print(df["previos"].unique())
# print(df["pdays"].unique())

# df['y'] = df['y'].map({'yes': 1, 'no': 0})
# df = df.lazy()
df = df.with_columns(
    pl.col("y")
    .map_dict({"no": 0, "yes": 1}, default="unknown")
    .alias("labels")
)
df = df.drop(['y'])
print(df)

for i in range(len(columns)):
    if df.dtypes[i] == pl.Utf8:
        print(df.dtypes[i])
        lf = df.select([columns[i], 'labels'])
        print(lf.group_by(columns[i]).agg(pl.col("labels").sum()))

        print(lf)

        values = lf[columns[i]].unique()
        print(values)
        means = {}
        for value in values:
            print(value)
            x_len = len(lf.filter((pl.col(columns[i]) == value)))
            print(x_len)
            print( lf.filter((pl.col(columns[i]) == value)).select(['labels']))
            # x_sum = pl.sum( lf.filter((pl.col(columns[i]) == value)).select(['labels']) )
            means[value] = x_sum / x_len
        df = df.with_columns(
            pl.col(columns[i])
            .map_dict(means, default="unknown")
            .alias(f"{columns[i]}_encoded")
        ).drop([columns[i]])

print(df)



def count_encoding(df):
    # data = df.select(pl.col([column_name])).join(count_encoded_data,
    #                                              on=column_name,
    #                                              how="left")
    data = df
    for counter, columns in enumerate(df.columns):
        # if df.dtypes[counter]
        # print(columns)
        count_encoded_data = df.group_by(columns).agg(
            pl.count().alias(f"{columns}_encoded"))
        data = data.join(count_encoded_data,
                         on=columns,
                         how="left").drop(columns)
    # df_encoded = data.select(pl.col([f"{column_name}_encoded"]))
    return data
