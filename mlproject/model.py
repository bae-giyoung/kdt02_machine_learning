import pandas as pd
from konlpy.tag import Okt # 토크나이저

train_df = pd.read_table("../data/ratings_train.txt")
test_df = pd.read_table("../data/ratings_test.txt")

train_df = train_df.fillna(" ")
test_df = test_df.fillna(" ")

okt = Okt()
text = "한글 자연어 처리는 재밌다. 이제부터 열심히 해야지 ㅎㅎㅎㅎㅎㅎ"

print(okt.morphs(text))