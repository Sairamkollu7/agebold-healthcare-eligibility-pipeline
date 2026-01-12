import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, initcap, to_date, regexp_replace, lit
)

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def read_partner_file(spark, file_path, delimiter):
    return (
        spark.read
        .option("header", True)
        .option("delimiter", delimiter)
        .csv(file_path)
    )

def standardize_dataframe(df, config):
    # Rename columns based on mapping
    for src_col, target_col in config["column_mapping"].items():
        df = df.withColumnRenamed(src_col, target_col)

    # Apply transformations
    df = (
        df
        .withColumn("first_name", initcap(col("first_name")))
        .withColumn("last_name", initcap(col("last_name")))
        .withColumn("email", lower(col("email")))
        .withColumn("dob", to_date(col("dob"), config["dob_format"]))
        .withColumn(
            "phone",
            regexp_replace(col("phone"), r"[^0-9]", "")
        )
        .withColumn(
            "phone",
            regexp_replace(col("phone"),
                            r"(\d{3})(\d{3})(\d{4})",
                            r"$1-$2-$3")
        )
        .withColumn("partner_code", lit(config["partner_code"]))
    )

    # Bonus: Drop rows where external_id is missing
    df = df.filter(col("external_id").isNotNull())

    return df.select(
        "external_id",
        "first_name",
        "last_name",
        "dob",
        "email",
        "phone",
        "partner_code"
    )

def main():
    spark = (
        SparkSession.builder
        .appName("AgeBold Eligibility Pipeline")
        .getOrCreate()
    )

    config = load_config("config/partners_config.json")
    unified_df = None

    for partner, cfg in config.items():
        file_path = f"data/{'acme.txt' if partner == 'acme_health' else 'bettercare.csv'}"
        raw_df = read_partner_file(spark, file_path, cfg["delimiter"])
        standardized_df = standardize_dataframe(raw_df, cfg)

        unified_df = (
            standardized_df
            if unified_df is None
            else unified_df.unionByName(standardized_df)
        )

    unified_df.write.mode("overwrite").parquet("output/unified_eligibility")
    unified_df.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
