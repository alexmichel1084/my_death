import polars as pl

df = pl.read_csv("datasets/bank.csv", separator=';')
columns = df.columns
df = df.lazy()


# rename labels
df = df.with_columns(
    pl.col("y")
    .map_dict({"no": 0, "yes": 1}, default="unknown")
    .alias("labels")
    .cast(pl.Int32)  # it is string in default
)
df = df.drop(['y'])



# Target Encoding
dtype_mistakes = ['education', 'loan', 'contact', 'poutcome']
not_utf8 = []
for i in range(len(columns) - 1):
    # print(f" columns {columns[i]} dtype {df.dtypes[i]} ")
    if df.dtypes[i] == pl.Utf8 or columns[i] in dtype_mistakes:

        means = df.select([columns[i], 'labels']) \
            .group_by(columns[i]) \
            .agg(pl.col("labels")
                 .mean()) \
            .collect().to_dict()

        means = dict(zip(means[columns[i]], means["labels"]))

        # print(means)
        df = df.with_columns(
            pl.col(columns[i])
            .map_dict(means, default="unknown")
            .alias(f"{columns[i]}_encoded")

        )
        df = df.drop([columns[i]])
    else:
        not_utf8.append(columns[i])



# rewrite month
mouth = df.select(['month', 'labels']) \
    .group_by('month') \
    .agg(pl.col("labels")
         .mean()) \
    .collect().to_dict()

mouth = dict(zip(mouth['month'], mouth["labels"]))

mouth['jan'] = 0
mouth['feb'] = 1
mouth['mar'] = 2
mouth['apr'] = 3
mouth['may'] = 4
mouth['jun'] = 5
mouth['jul'] = 6
mouth['aug'] = 7
mouth['sep'] = 8
mouth['oct'] = 9
mouth['nov'] = 10
mouth['dec'] = 11
df = df.with_columns(
    pl.col('month')
    .map_dict(mouth, default="unknown")
    .alias("month_encoded")
)
df = df.drop(['month'])
not_utf8.remove('month')

# replace none
for column in not_utf8:
    mean_value = df.select(pl.mean(column)).collect()[column][0]
    df = df.with_columns(
        pl.col(column).cast(pl.Float64).fill_null(mean_value)
        # Можно убрать, но тогда все значения в датасете будут Float64
        # .cast(pl.Int64)
    )
df = df.cast(pl.Float64)


# z-score
z_score_columns = list(filter(lambda x: not x.endswith('_encoded'), df.columns))
z_score_columns.remove("labels")
z_score_df = df.select(z_score_columns)
means = z_score_df.mean()
stds = z_score_df.std()
for column in z_score_columns:
    mean = means.collect()[column][0]
    std = stds.collect()[column][0]
    df = df.with_columns(
            ((pl.col(column) - mean / std).alias(f"{column}_normalized"))
        )
    df = df.drop(column)


corr_matrix = df.collect().corr().to_dict()

correlated = []
for i, column in enumerate(corr_matrix):
    counter = 0
    for coef in corr_matrix[column]:
        if float(coef) > 0.5:# and float(coef) != 1:
            correlated.append((i, counter))
        counter += 1

# print(correlated)
df.drop(['pdays_normalized'])
# previous_normalized, pdays_normalized are corelated

df.collect().write_csv("datasets/prepared_data.csv")




