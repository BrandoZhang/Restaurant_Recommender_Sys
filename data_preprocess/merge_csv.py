import pandas, os

review_csv_path = os.path.join(".", "review.csv")
user_csv_path = os.path.join(".", "user.csv")
business_csv_path = os.path.join(".", "business.csv")
merged_csv_path = os.path.join(".", "merged.csv")

review_df = pandas.read_csv(review_csv_path)
user_df = pandas.read_csv(user_csv_path)
business_df = pandas.read_csv(business_csv_path)

merged_df = review_df.merge(user_df, on="user_id").merge(business_df, on="business_id")

merged_df.to_csv(merged_csv_path)
