from sqlalchemy import create_engine
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_invoice_data():
    engine = create_engine("postgresql://postgres:bhumi@localhost:5432/postgres")
    engine.connect()

    tables = pd.read_sql_query(
        """
    SELECT tablename 
    FROM tables 
    WHERE schemaname = 'public';
    """,
        engine,
    )
    tables

    df = pd.read_sql_query(
        """
    With purchase_agg AS (
    Select
    p.PONumber,
    COUNT(DISTINCT p.brand) AS total_brands,
    sum(p.quantity) as total_item_quantity,
    sum(p.dollars) as total_item_dollars,
    Avg(p.ReceivingDate::date - p.PODate::date) as avg_receiving_delay
    from purchases p 
    group by p.PONumber
    )

    SELECT
    vi.PONumber,
    vi.quantity AS invoice_quantity,
    vi.dollars AS invoice_dollars,
    vi.freight,
    (vi.InvoiceDate::Date - vi.PODate::Date) AS days_po_to_invoice,
    (vi.PayDate::Date - vi.InvoiceDate::Date ) AS days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay

    from vendor_invoice vi
    LEFT JOIN purchase_agg pa
    ON vi.PONumber = pa.PONumber
        """,
        engine,
    )

    engine.dispose()
    return df
    
def create_invoice_risk_label(row):

    if abs(row["invoice_dollars"]-row["total_item_dollars"])>5:
        return 1
    
    if row["avg_receiving_delay"]>10:
        return 1
    
    return 0

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df

def split_data(df, features, target):
    X=df[features]
    Y=df[target]

    return train_test_split(
    X,Y, test_size=0.2, random_state=42
    )


def scale_feature(X_train, X_test,scaler_path):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler,'models/scaler.pkl')
    return X_train_scaled, X_test_scaled
