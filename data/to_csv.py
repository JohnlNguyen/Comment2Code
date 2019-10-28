
import pickle
import pandas as pd

y_preds, y_true = [], []
with open('./valid_shared_transformer.pkl', 'rb') as f:
    data = pickle.load(f)
    y_preds, y_true = data


pd.DataFrame(list(zip(y_true, y_preds)), columns=['y_true', 'y_pred']).to_csv('./ba_valid.csv', index=False)
