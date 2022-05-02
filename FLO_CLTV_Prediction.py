# Görev 1: Veriyi Hazırlama
# Adım1: flo_data_20K.csvverisiniokuyunuz.

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
#pd.set_option('display.width', 500)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("/Users/eminebozkurt/Desktop/vbo/week3/hw2/FLO_CLTV_Tahmini/flo_data_20k.csv")
df_.head()
df = df_.copy()
df.head()



# Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable): # kendisine girilen değişken için eşik değer belirler.
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round((quartile3 + 1.5 * interquantile_range), 0)
    low_limit = round((quartile1 - 1.5 * interquantile_range), 0)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



# Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"
# değişkenlerinin aykırı değerleri varsa baskılayanız.

outlier_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_online")

df.describe().T

outlier_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "order_num_total_ever_offline")

outlier_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")

# df.sort_values(by="order_num_total_ever_online", ascending=False)[["order_num_total_ever_online"]].head(10)



# Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

# master_id: Eşsiz müşteri numarası
# order_channel:Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

df.head()
df["order_num_total_ever_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes
df.loc[:, ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]] = df.loc[:, df.columns.str.contains("date")].astype("datetime64[ns]")


def convert_date(df):
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col])

df.dtypes
# df = df_.copy()
df.head()
convert_date(df)


# Görev 2: CLTV Veri Yapısının Oluşturulması
# Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()# Timestamp('2021-05-30 00:00:00')

df["last_order_date"].max() # 2021-05-30

analysis_date = dt.datetime(2021, 6, 1)

# Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df = pd.DataFrame({"customer_id": df["master_id"],
             "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T_weekly": ((analysis_date - df["first_order_date"]).dt.days)/7,
             "frequency": df["order_num_total_ever_omnichannel"],
             "monetary_cltv_avg": df["customer_value_total_ever_omnichannel"] / df["customer_value_total_ever_omnichannel"]})

cltv_df.head()


# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
# Adım 1: BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["New_Exp_Sales_3_Month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly']).sort_values(ascending=False).head(10)


# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["New_Exp_Sales_6_Month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly']).sort_values(ascending=False).head(10)

# Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary_cltv_avg'])


# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
# • Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_df.head()

cltv_df["cltv_6_months"] = cltv
cltv_df.sort_values("cltv_6_months", ascending=False)[:20]


# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df.head()

cltv_df["segment"] = pd.qcut(cltv_df["cltv_6_months"], 4, labels=["D", "C", "B", "A"])

# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.
cltv_df.head()
cltv_df.describe().T

cltv_df.groupby("segment").agg({"recency_cltv_weekly": "sum",
                                "cltv_6_months": "sum",
                                "T_weekly": "sum",
                                "frequency": "sum",
                                "exp_average_value": "sum",
                                "New_Exp_Sales_3_Month": "sum",
                                "New_Exp_Sales_6_Month": "sum",
                                "cltv_6_months": "sum"})