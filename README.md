# J-pre
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "% matplotlib inline\n",
    "\n",
    "df=pd.read_csv('train_new.csv')\n",
    "test=pd.read_csv('test_new.csv')\n",
    "df= df.drop(1567, axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#################チームの組み合わせ#######\n",
    "df_home_dm=pd.get_dummies(df['home'])\n",
    "df_away_dm=pd.get_dummies(df['away'])\n",
    "df_oh1=df_home_dm+df_away_dm\n",
    "df=pd.concat([df,df_oh1], axis=1)\n",
    "df=df.drop('home', axis=1)\n",
    "df=df.drop('away', axis=1)\n",
    "\n",
    "test_home_dm=pd.get_dummies(test['home'])\n",
    "test_away_dm=pd.get_dummies(test['away'])\n",
    "test_oh1=test_home_dm+test_away_dm\n",
    "test=pd.concat([test,test_oh1], axis=1)\n",
    "test=test.drop('home', axis=1)\n",
    "test=test.drop('away', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "################################ stage\n",
    "def stage (x):\n",
    "    x=x[1:]\n",
    "    return x\n",
    "\n",
    "df[\"stage\"]=df[\"stage\"].apply(stage)\n",
    "test[\"stage\"]=test[\"stage\"].apply(stage)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "################################ match 節\n",
    "#df['match']= df['match'].str.strip('第節')\n",
    "def match (x):\n",
    "    x=x[1:]\n",
    "    x=x[:-4]\n",
    "    return x\n",
    "df[\"match\"]=df[\"match\"].apply(match)\n",
    "test[\"match\"]=test[\"match\"].apply(match)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "############################開幕戦\n",
    "def match2(x):\n",
    "    if x=='１':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "df[\"firstgame\"]=df[\"match\"].apply(match2)\n",
    "test[\"firstgame\"]=test[\"match\"].apply(match2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "############################J2最終戦　\n",
    "def match3(x):\n",
    "    if x=='４２':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "df[\"lastgame2\"]=df[\"match\"].apply(match3)\n",
    "test[\"lastgame2\"]=test[\"match\"].apply(match3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "############################J1のみ最終戦\n",
    "#J１とJ２のデータを抽出\n",
    "df_j1=df[df.stage=='１']\n",
    "df_j2=df[df.stage=='２']\n",
    "def match4(x):\n",
    "    if x=='３４':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "df[\"lastgame1\"]=df_j1[\"match\"].apply(match4)\n",
    "#df[\"lastgame1\"]\n",
    "df[\"lastgame1\"]=df[\"lastgame1\"].fillna(0.0)\n",
    "\n",
    "#J１とJ２のデータを抽出\n",
    "test_j1=test[test.stage=='１']\n",
    "test_j2=test[test.stage=='２']\n",
    "def match4(x):\n",
    "    if x=='３４':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "test[\"lastgame1\"]=test_j1[\"match\"].apply(match4)\n",
    "#df[\"lastgame1\"]\n",
    "test[\"lastgame1\"]=test[\"lastgame1\"].fillna(0.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "#########################J1とJ２の最終戦の結合　変数名だけで、結果に影響なし\n",
    "df[\"lastgame\"]=df[\"lastgame2\"]+df[\"lastgame1\"]\n",
    "#vc = df[\"lastgame\"].value_counts()\n",
    "\n",
    "test[\"lastgame\"]=test[\"lastgame2\"]+test[\"lastgame1\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "################################################################################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "#############################month\n",
    "def month (x):\n",
    "    x=x[:2]\n",
    "    return x\n",
    "df[\"game_month\"]=df[\"gameday\"].apply(month)  \n",
    "test[\"game_month\"]=test[\"gameday\"].apply(month)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "#############################12月\n",
    "def decenber (x):\n",
    "    if x=='12':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "df['decenber']=df[\"game_month\"].apply(decenber)\n",
    "test['decenber']=test[\"game_month\"].apply(decenber)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "##############################　weather\n",
    "dfweather= df['weather'].str.strip('一時曇晴れのち々')\n",
    "def weather1(x):\n",
    "    if x=='雨':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "df[\"rain\"]=df[\"weather\"].apply(weather1)\n",
    "def weather2(x):\n",
    "    if x=='屋内':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "df[\"dome\"]=df[\"weather\"].apply(weather2)\n",
    "\n",
    "testweather= test['weather'].str.strip('一時曇晴れのち々')\n",
    "def weather1(x):\n",
    "    if x=='雨':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "test[\"rain\"]=test[\"weather\"].apply(weather1)\n",
    "def weather2(x):\n",
    "    if x=='屋内':\n",
    "        x=1\n",
    "    else: \n",
    "        x=0\n",
    "    return x\n",
    "test[\"dome\"]=test[\"weather\"].apply(weather2)\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "###############################　得点王\n",
    "df_14=df[df.year==2014]\n",
    "test_14=test[test.year==2014]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "def king14(x):\n",
    "    y=0\n",
    "    if x==\"大久保　嘉人\" or x==\"大黒　将志\" :\n",
    "        y=y+1\n",
    "    elif x==\"豊田　陽平\":\n",
    "        y=y+1\n",
    "    elif x==\"マルキーニョス\":\n",
    "        y=y+1  \n",
    "    else:\n",
    "        y=0\n",
    "    return y\n",
    "df[\"topscorere\"]=df_14['home_01'].apply(king14)+df_14['home_02'].apply(king14)+df_14['home_03'].apply(king14)+df_14['home_04'].apply(king14)+df_14['home_05'].apply(king14)+df_14['home_06'].apply(king14)+df_14['home_07'].apply(king14)+df_14['home_08'].apply(king14)+df_14['home_09'].apply(king14)+df_14['home_10'].apply(king14)+df_14['home_11'].apply(king14)+df_14['away_01'].apply(king14)+df_14['away_02'].apply(king14)+df_14['away_03'].apply(king14)+df_14['away_04'].apply(king14)+df_14['away_05'].apply(king14)+df_14['away_06'].apply(king14)+df_14['away_07'].apply(king14)+df_14['away_08'].apply(king14)+df_14['away_09'].apply(king14)+df_14['away_10'].apply(king14)+df_14['away_11'].apply(king14)\n",
    "df[\"topscorere\"]=df[\"topscorere\"].fillna(0.0)\n",
    "test[\"topscorere\"]=test_14['home_01'].apply(king14)+test_14['home_02'].apply(king14)+test_14['home_03'].apply(king14)+test_14['home_04'].apply(king14)+test_14['home_05'].apply(king14)+test_14['home_06'].apply(king14)+test_14['home_07'].apply(king14)+test_14['home_08'].apply(king14)+test_14['home_09'].apply(king14)+test_14['home_10'].apply(king14)+test_14['home_11'].apply(king14)+test_14['away_01'].apply(king14)+test_14['away_02'].apply(king14)+test_14['away_03'].apply(king14)+test_14['away_04'].apply(king14)+test_14['away_05'].apply(king14)+test_14['away_06'].apply(king14)+test_14['away_07'].apply(king14)+test_14['away_08'].apply(king14)+test_14['away_09'].apply(king14)+test_14['away_10'].apply(king14)+test_14['away_11'].apply(king14)\n",
    "test[\"topscorere\"]=test[\"topscorere\"].fillna(0.0)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "#######################  日本代表\n",
    "def national(x):\n",
    "    y=0\n",
    "    if x==\"酒井　高徳\":\n",
    "        y=y+1\n",
    "    elif x==\"森重　真人\":\n",
    "        y=y+1\n",
    "    elif x==\"遠藤　保仁\":\n",
    "        y=y+1  \n",
    "    elif x==\"清武　弘嗣\":\n",
    "        y=y+1\n",
    "    elif x==\"遠藤　保仁\":\n",
    "        y=y+1  \n",
    "    elif x==\"柿谷　曜一朗\":\n",
    "        y=y+1\n",
    "    elif x==\"西川　周作\":\n",
    "        y=y+1  \n",
    "    elif x==\"大久保　嘉人\":\n",
    "        y=y+1\n",
    "    elif x==\"青山　敏弘\":\n",
    "        y=y+1 \n",
    "    elif x==\"今野　泰幸\":\n",
    "        y=y+1  \n",
    "    elif x==\"山口　蛍\":\n",
    "        y=y+1\n",
    "    elif x==\"大迫　勇也\":\n",
    "        y=y+1  \n",
    "    elif x==\"伊野波　雅彦\":\n",
    "        y=y+1\n",
    "    elif x==\"齋藤　学\":\n",
    "        y=y+1  \n",
    "    elif x==\"酒井　宏樹\":\n",
    "        y=y+1\n",
    "    elif x==\"権田　修一\":\n",
    "        y=y+1 \n",
    "    else:\n",
    "        y=0\n",
    "    return y\n",
    "df[\"national\"]=df['home_01'].apply(national)+df['home_02'].apply(national)+df['home_03'].apply(national)+df['home_04'].apply(national)+df['home_05'].apply(national)+df['home_06'].apply(national)+df['home_07'].apply(national)+df['home_08'].apply(national)+df['home_09'].apply(national)+df['home_10'].apply(national)+df['home_11'].apply(national)+df['away_01'].apply(national)+df['away_02'].apply(national)+df['away_03'].apply(national)+df['away_04'].apply(national)+df['away_05'].apply(national)+df['away_06'].apply(national)+df['away_07'].apply(national)+df['away_08'].apply(national)+df['away_09'].apply(national)+df['away_10'].apply(national)+df['away_11'].apply(national)\n",
    "test[\"national\"]=test['home_01'].apply(national)+test['home_02'].apply(national)+test['home_03'].apply(national)+test['home_04'].apply(national)+test['home_05'].apply(national)+test['home_06'].apply(national)+test['home_07'].apply(national)+test['home_08'].apply(national)+test['home_09'].apply(national)+test['home_10'].apply(national)+test['home_11'].apply(national)+test['away_01'].apply(national)+test['away_02'].apply(national)+test['away_03'].apply(national)+test['away_04'].apply(national)+test['away_05'].apply(national)+test['away_06'].apply(national)+test['away_07'].apply(national)+test['away_08'].apply(national)+test['away_09'].apply(national)+test['away_10'].apply(national)+test['away_11'].apply(national)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "#######################  日本代表cap\n",
    "def cap(x):\n",
    "    y=0\n",
    "    if x==\"川口　能活\":\n",
    "        y=y+1\n",
    "    elif x==\"中澤　佑二\":\n",
    "        y=y+1\n",
    "    elif x==\"中村　俊輔\":\n",
    "        y=y+1  \n",
    "    elif x==\"三浦　知良\":\n",
    "        y=y+1\n",
    "    elif x==\"三都主　アレサンドロ\":\n",
    "        y=y+1  \n",
    "    elif x==\"稲本　潤一\":\n",
    "        y=y+1\n",
    "    elif x==\"駒野　友一\":\n",
    "        y=y+1  \n",
    "    elif x==\"楢崎　正剛\":\n",
    "        y=y+1\n",
    "    elif x==\"玉田　圭司\":\n",
    "        y=y+1 \n",
    "    elif x==\"中村　憲剛\":\n",
    "        y=y+1  \n",
    "    elif x==\"加地　亮\":\n",
    "        y=y+1\n",
    "    elif x==\"高原　直泰\":\n",
    "        y=y+1  \n",
    "    elif x==\"中田　浩二\":\n",
    "        y=y+1\n",
    "    elif x==\"小野　伸二\":\n",
    "        y=y+1  \n",
    "    elif x==\"阿部　勇樹\":\n",
    "        y=y+1 \n",
    "    else:\n",
    "        y=0\n",
    "    return y\n",
    "df[\"cap\"]=df['home_01'].apply(cap)+df['home_02'].apply(cap)+df['home_03'].apply(cap)+df['home_04'].apply(cap)+df['home_05'].apply(cap)+df['home_06'].apply(cap)+df['home_07'].apply(cap)+df['home_08'].apply(cap)+df['home_09'].apply(cap)+df['home_10'].apply(cap)+df['home_11'].apply(cap)+df['away_01'].apply(cap)+df['away_02'].apply(cap)+df['away_03'].apply(cap)+df['away_04'].apply(cap)+df['away_05'].apply(cap)+df['away_06'].apply(cap)+df['away_07'].apply(cap)+df['away_08'].apply(cap)+df['away_09'].apply(cap)+df['away_10'].apply(cap)+df['away_11'].apply(cap)\n",
    "test[\"cap\"]=test['home_01'].apply(cap)+test['home_02'].apply(cap)+test['home_03'].apply(cap)+test['home_04'].apply(cap)+test['home_05'].apply(cap)+test['home_06'].apply(cap)+test['home_07'].apply(cap)+test['home_08'].apply(cap)+test['home_09'].apply(cap)+test['home_10'].apply(cap)+test['home_11'].apply(cap)+test['away_01'].apply(cap)+test['away_02'].apply(cap)+test['away_03'].apply(cap)+test['away_04'].apply(cap)+test['away_05'].apply(cap)+test['away_06'].apply(cap)+test['away_07'].apply(cap)+test['away_08'].apply(cap)+test['away_09'].apply(cap)+test['away_10'].apply(cap)+test['away_11'].apply(cap)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ホールドアウト法\n",
    "X = df.loc[:, ['stage','temperature','capa','firstgame','lastgame','rain','dome','decenber',\"national\",\"cap\",\"topscorere\",\n",
    "       'アビスパ福岡','アルビレックス新潟','カターレ富山','カマタマーレ讃岐','ガイナーレ鳥取','ガンバ大阪','ギラヴァンツ北九州','コンサドーレ札幌','サガン鳥栖','サンフレッチェ広島','ザスパクサツ群馬','ザスパ草津','ジェフユナイテッド千葉','ジュビロ磐田','セレッソ大阪','ファジアーノ岡山','ベガルタ仙台','モンテディオ山形','ロアッソ熊本','ヴァンフォーレ甲府','ヴィッセル神戸','京都サンガF.C.','名古屋グランパス','大分トリニータ','大宮アルディージャ','川崎フロンターレ','徳島ヴォルティス','愛媛ＦＣ','東京ヴェルディ','松本山雅ＦＣ','柏レイソル','栃木ＳＣ','横浜Ｆ・マリノス','横浜ＦＣ','水戸ホーリーホック','浦和レッズ','清水エスパルス','湘南ベルマーレ','鹿島アントラーズ','ＦＣ岐阜','ＦＣ東京','ＦＣ町田ゼルビア','Ｖ・ファーレン長崎'\n",
    "              ]].values\n",
    "y = df.loc[:, ['y']].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ホールド・アウト法によるデータの分割\n",
    "# random_stateを0に固定してあるので､毎回同じサンプルに分割\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)\n",
    "# 線形回帰のモデルを学習させる\n",
    "from sklearn.linear_model import LinearRegression\n",
    "lr = LinearRegression()\n",
    "lr.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 7.78122251e+02,  3.91657100e+01,  2.31463883e-01,\n",
       "         2.01780192e+03,  3.63821039e+03, -1.29481754e+03,\n",
       "        -6.24355853e+02,  2.31401692e+03,  9.33340929e+02,\n",
       "         6.85590970e+02,  7.61469794e+02,  3.80973833e+11,\n",
       "         3.80973841e+11,  3.80973832e+11,  3.80973832e+11,\n",
       "         3.80973832e+11,  3.80973838e+11,  3.80973833e+11,\n",
       "         3.80973835e+11,  3.80973836e+11,  3.80973835e+11,\n",
       "         3.80973833e+11,  3.80973832e+11,  3.80973835e+11,\n",
       "         3.80973835e+11,  3.80973840e+11,  3.80973835e+11,\n",
       "         3.80973838e+11,  3.80973834e+11,  3.80973833e+11,\n",
       "         3.80973836e+11,  3.80973835e+11,  3.80973834e+11,\n",
       "         3.80973838e+11,  3.80973833e+11,  3.80973836e+11,\n",
       "         3.80973837e+11,  3.80973833e+11,  3.80973833e+11,\n",
       "         3.80973830e+11,  3.80973836e+11,  3.80973837e+11,\n",
       "         3.80973833e+11,  3.80973837e+11,  3.80973835e+11,\n",
       "         3.80973833e+11,  3.80973846e+11,  3.80973837e+11,\n",
       "         3.80973835e+11,  3.80973837e+11,  3.80973833e+11,\n",
       "         3.80973838e+11,  3.80973833e+11,  3.80973833e+11]])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 偏回帰係数a1,a2を出力\n",
    "# [LSTATの係数, RMの係数]\n",
    "lr.coef_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "adjusted R^2\n",
      "train: 0.784903\n",
      "test : 0.795862\n"
     ]
    }
   ],
   "source": [
    "# 自由度調整済み決定係数\n",
    "# (決定係数, trainまたはtestのサンプル数, 利用した特徴量の数)\n",
    "def adjusted(score, n_sample, n_features):\n",
    "    adjusted_score = 1 - (1 - score) * ((n_sample - 1) / (n_sample - n_features - 1))\n",
    "    return adjusted_score\n",
    "# 自由度調整済み決定係数を出力\n",
    "print('adjusted R^2')\n",
    "print('train: %3f' % adjusted(lr.score(X_train, y_train), len(y_train), 2))\n",
    "print('test : %3f' % adjusted(lr.score(X_test, y_test), len(y_test), 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE\n",
      "train: 3761.853\n",
      "test : 3630.523\n"
     ]
    }
   ],
   "source": [
    "# MSEを出力する関数を読み込む\n",
    "from sklearn.metrics import mean_squared_error as mse\n",
    "# RMSEをtrainとtestに分けて出力\n",
    "# 過学習をしているかどうかを確認\n",
    "print('RMSE')\n",
    "print('train: %.3f' % (mse(y_train, lr.predict(X_train)) ** (1/2)))\n",
    "print('test : %.3f' % (mse(y_test, lr.predict(X_test)) ** (1/2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/koudai/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/site-packages/pandas/core/indexing.py:1472: FutureWarning: \n",
      "Passing list-likes to .loc or [] with any missing label will raise\n",
      "KeyError in the future, you can use .reindex() as an alternative.\n",
      "\n",
      "See the documentation here:\n",
      "https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike\n",
      "  return self._getitem_tuple(key)\n"
     ]
    }
   ],
   "source": [
    "Xt= test.loc[:, ['stage','temperature','capa','firstgame','lastgame','rain','dome','decenber',\"national\",\"cap\",\"topscorere\",\n",
    "       'アビスパ福岡','アルビレックス新潟','カターレ富山','カマタマーレ讃岐','ガイナーレ鳥取','ガンバ大阪','ギラヴァンツ北九州','コンサドーレ札幌','サガン鳥栖','サンフレッチェ広島','ザスパクサツ群馬','ザスパ草津','ジェフユナイテッド千葉','ジュビロ磐田','セレッソ大阪','ファジアーノ岡山','ベガルタ仙台','モンテディオ山形','ロアッソ熊本','ヴァンフォーレ甲府','ヴィッセル神戸','京都サンガF.C.','名古屋グランパス','大分トリニータ','大宮アルディージャ','川崎フロンターレ','徳島ヴォルティス','愛媛ＦＣ','東京ヴェルディ','松本山雅ＦＣ','柏レイソル','栃木ＳＣ','横浜Ｆ・マリノス','横浜ＦＣ','水戸ホーリーホック','浦和レッズ','清水エスパルス','湘南ベルマーレ','鹿島アントラーズ','ＦＣ岐阜','ＦＣ東京','ＦＣ町田ゼルビア','Ｖ・ファーレン長崎'\n",
    "              ]]\n",
    "Xt=Xt.fillna(0.0)\n",
    "Xt=Xt.values\n",
    "test_predict=lr.predict(Xt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15822</td>\n",
       "      <td>12325.871582</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15823</td>\n",
       "      <td>14768.933350</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15824</td>\n",
       "      <td>31871.965088</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id             y\n",
       "0  15822  12325.871582\n",
       "1  15823  14768.933350\n",
       "2  15824  31871.965088"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_predict=test_predict.reshape(-1)\n",
    "test[\"y\"]=pd.DataFrame(test_predict)\n",
    "test_id=pd.DataFrame(test[\"id\"])\n",
    "predict=pd.concat([test_id,test[\"y\"]], axis=1)\n",
    "predict.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "predict.to_csv('predict3.csv')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
