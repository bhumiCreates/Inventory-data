from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:bhumi@localhost:5432/inventory")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(db_path: str):
    engine = create_engine(
    "postgresql://postgres:bhumi@localhost:5432/postgres"
    )
    engine.connect()
    tables = pd.read_sql_query("""
    SELECT tablename 
    FROM tables 
    WHERE schemaname = 'public';
    """, engine)
    tables
    
    df=pd.read_sql_query("select * from vendor_invoice", engine)

    return df


def prepare_feature(df: pd.DataFrame):
    X=df[["dollars","quantity"]]
    Y=df[["freight"]]
    return X,Y


def split_data(X,Y,test_size=0.2,random_state=42):
    return train_test_split(
        X,Y,test_size=test_size, random_state=random_state
    )
    