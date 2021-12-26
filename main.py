import pandas as pd

data = pd.read_pickle('inDelphi/test_FORECasT_inDelphi.pkl')

del_feat = pd.DataFrame(data["del_features"])
