import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("amazon_review.csv")
df = df_
df.head()

#Part 1

#Step 1

df["overall"].mean()

#Step 2

df.info()
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()
df["days"] = (current_date - df["reviewTime"]).dt.days
A = df["days"].quantile([0.25,0.5,0.75])


df.loc[df["days"]<=A[0.25],"overall"].mean()*32/100 + \
df.loc[(df["days"]>A[0.25]) & (df["days"]<=A[0.5]),"overall"].mean()*26/100 +\
df.loc[(df["days"]>A[0.5]) & (df["days"]<=A[0.75]),"overall"].mean()*24/100 + \
df.loc[(df["days"]>A[0.75]),"overall"].mean()*18/100

#Step 3

B = df["day_diff"].quantile([0.25,0.5,0.75])

def time_based_weighted_average(dataframe,w1=36,w2=28,w3=20,w4=16):
    return dataframe.loc[df["day_diff"]<=B[0.25],"overall"].mean()*w1/100 + \
dataframe.loc[(df["day_diff"]>B[0.25]) & (df["day_diff"]<=B[0.5]),"overall"].mean()*w2/100 +\
dataframe.loc[(df["day_diff"]>B[0.5]) & (df["day_diff"]<=B[0.75]),"overall"].mean()*w3/100 + \
dataframe.loc[(df["day_diff"]>B[0.75]),"overall"].mean()*w4/100

time_based_weighted_average(df)

#Step 4

df.loc[df["days"]<=A[0.25],"overall"].mean()

df.loc[(df["days"]>A[0.25]) & (df["days"]<=A[0.5]),"overall"].mean()

df.loc[(df["days"]>A[0.5]) & (df["days"]<=A[0.75]),"overall"].mean()

df.loc[(df["days"]>A[0.75]),"overall"].mean()

#Elde edilen puan ortalamaları bizlere açıkça gösteriyor ki, son zaman diliminde verilen puanlar geçmiş günlere oranla artış
#göstermiştir. Bunun nedeni üründe yapılan güncelleştirme, iyileştirme vs olabilir.

#Part 2

#Step 1
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

no = df[df["helpful_no"] == 1].shape[0]
yes = df[df["helpful_yes"] == 1].shape[0]

#Step 2
score_pos_neg_diff = yes - no
df["score_pos_neg_diff"] = score_pos_neg_diff

score_average_rating = yes / (yes+no) 
df["score_average_rating"] = score_average_rating

def wilson_lower_bound(yes, no, confidence=0.95):
    n = yes + no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = wilson_lower_bound(yes,no)

#Step 3

df_new = df.sort_values("wilson_lower_bound",ascending=False)

df_new["reviewText"].head(20)

#Sıralama sonucu getirilen yorumlar incelendiğinde genel olarak üründen memnuniyet duyulmakta. Yani yapılmış olunan bu sıra-
#lama genel hatlarıyla tutarlı ve sosyal ispat durumuna ters olmayacak şekilde doğru bir değerlendirme olmuş durumda.
