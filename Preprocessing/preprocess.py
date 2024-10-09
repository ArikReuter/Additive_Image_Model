from sklearn.impute import SimpleImputer
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

class DataProcessor:
    def __init__(self, df_loc, image_loc, save_loc):
        self.df_loc = df_loc
        self.image_loc = image_loc
        self.save_loc = save_loc
        self.target_df = pd.read_excel(self.df_loc)
        self.images = torch.load(self.image_loc)
        self.res_lis = []
        self.valid_idx_lis = []

    def clean_price_column(self):
        for i, s in enumerate(self.target_df["price"]):
            try:
                if isinstance(s, str):
                    s = s[1:].replace(",", "")
                    self.res_lis.append(float(s))
                    self.valid_idx_lis.append(i)
            except Exception as e:
                print(f"Error processing price: {s}, {e}")

    def filter_outliers(self):
        targets = torch.tensor(self.res_lis)
        self.images = self.images[self.valid_idx_lis]
        
        minimum = torch.quantile(targets, 0.05)
        maximum = torch.quantile(targets, 1 - 0.05)
        valid_idx2 = torch.logical_and(targets > minimum, targets < maximum)

        self.targets = torch.log(targets[valid_idx2])
        self.images = self.images[valid_idx2]

    def plot_histogram(self):
        plt.hist(self.targets, bins=100)
        plt.show()

    def preprocess_dataframe(self):
        imputer = SimpleImputer(strategy="median")
        columns_to_impute = [
            "accommodates",
            "bedrooms",
            "number_of_reviews",
            "review_scores_value",
            "minimum_nights",
        ]
        
        imputer.fit(self.target_df[columns_to_impute])
        self.target_df[columns_to_impute] = imputer.transform(self.target_df[columns_to_impute])
        self.target_df = self.target_df.reset_index(drop=True)
        self.target_df = self.target_df[self.valid_idx2.numpy()]

    def prepare_train_data(self):
        self.train_df = self.target_df.reset_index(drop=True)
        self.train_df = self.train_df[
            [
                "room_type",
                "accommodates",
                "bedrooms",
                "minimum_nights",
                "number_of_reviews",
                "review_scores_value",
                "host_identity_verified",
            ]
        ]
        self.train_df_unscaled = deepcopy(self.train_df)

    def normalize_df(self, df):
        return (df - df.mean()) / df.std()

    def min_max_df(self, df):
        min_max_scaler = MinMaxScaler()
        return min_max_scaler.fit_transform(df)

    def transform_data(self, df):
        num_cols = [dt.kind != "O" for dt in df.dtypes]
        df[df.columns.values[num_cols]] = self.normalize_df(df[df.columns.values[num_cols]])
        return pd.get_dummies(df)

    def save_data(self):
        self.train_df_unscaled = pd.get_dummies(self.train_df_unscaled)
        self.train_df = self.transform_data(self.train_df)
        self.train_df.to_excel(self.save_loc)

    def process(self):
        self.clean_price_column()
        self.filter_outliers()
        self.plot_histogram()
        self.preprocess_dataframe()
        self.prepare_train_data()
        self.save_data()


# Usage
df_loc = "path_to_excel_file"
image_loc = "path_to_image_file"
save_loc = "path_to_save_file"

processor = DataProcessor(df_loc, image_loc, save_loc)
processor.process()
