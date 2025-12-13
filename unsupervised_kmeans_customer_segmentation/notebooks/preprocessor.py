
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Preprocessor: 
    def __init__(self, df):
        self.df = df.copy()
        
    def handle_missing_values(self):
        self.df.fillna(0, inplace=True)
      
    def categorical_values(self):
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        fixed = enc.fit_transform(self.df[['Gender']])
        
        self.df.drop(columns=['Gender'], inplace=True)
        
        cat_df = pd.DataFrame(
            fixed,
            index=self.df.index,
            columns=enc.get_feature_names_out(['Gender'])
        )
        return cat_df  
    
    def nomalize_values(self, columns):
        st = StandardScaler()
        norm = st.fit_transform(self.df[columns])
        norm_df = pd.DataFrame(norm, index=self.df.index, columns=columns)
        return norm_df        
        
    def transform(self): 
        if 'CustomerID' in self.df.columns:
            self.df.drop(columns=['CustomerID'], inplace=True)

        self.handle_missing_values()

        num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        norm = self.nomalize_values(num_cols)

        cat = self.categorical_values()

        self.df = pd.concat([norm, cat], axis=1)

        return self.df


p = Preprocessor(df)
X = p.transform()
print(X.head())
print(X.shape)
