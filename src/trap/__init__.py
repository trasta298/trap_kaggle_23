
correct_features2 = ['user', 'anime_id']
x_train2 = train_data_merged[correct_features2]
x_test2 = test_data_merged[correct_features2]


# user ごとの score の平均値を計算
user_score_mean = train_data_merged.groupby('user')['score'].mean()

# anime ごとの score の平均値を計算
anime_score_mean = train_data_merged.groupby('anime_id')['score'].mean()

def predict_score(user_id, anime_id):
    # user と anime の score の相乗平均を返す。片方が存在しない場合は、存在する方の score を返す。両方存在しない場合は、Noneを返す
    user_score = user_score_mean.get(user_id)
    anime_score = anime_score_mean.get(anime_id)
    if user_score is None and anime_score is None:
        return None
    elif user_score is None:
        return anime_score
    elif anime_score is None:
        return user_score
    else:
        return math.sqrt(user_score * anime_score)  # type: ignore
    

# x_train2 の userscore と animescore の相乗平均を予測値とする
y_pred_test2 = np.array([predict_score(row['user'], row['anime_id']) for _, row in x_test2.iterrows()])