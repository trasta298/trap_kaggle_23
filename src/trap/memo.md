ageの欠損値をそのままにするようにした
Mean Squared Error for test data: 3.639573539201261
-> 3.24270

userscore_meanの追加
Mean Squared Error for test data: 3.566464017811187
-> 3.26696

userscore_0~10の正規化
Mean Squared Error for test data: 3.5767545883274803
-> 3.28104

一旦戻す

Mean Squared Error for test data: 3.6397764731207927

sqrt(target)を試してみる

Mean Squared Error for test data: 3.7112010628605807

なんか思ったより悪くないので、アンサンブルには使えそう

age_categoryの追加

Mean Squared Error for test data: 3.645220001084081

userに変えてみた

Mean Squared Error for test data: 3.6370072811235055

-> 3.26536

---

'episodes', 'members', 'ranked', 'start_year', 'gen_', 'userscore_', 'agg_'
-> 3.23605

age追加
Mean Squared Error for test data: 3.6383144561680143

text_len追加
Mean Squared Error for test data: 3.6229735525904156
-> 3.21609

umap追加
Mean Squared Error for test data: 3.642247105183543
削除

season追加
Mean Squared Error for test data: 3.6158496338371973
-> 3.20979

anime_id でgroupbyしたものを追加
Mean Squared Error for test data: 3.6086669429237914
-> 3.23026


num_leaves 127
-> 3.19716

num_leaves 127 + num_leaves 63 + nn
-> 3.19529


Monitored metric val_loss did not improve in the last 5 records. Best score: 3.945. Signaling Trainer to stop.

Monitored metric val_loss did not improve in the last 5 records. Best score: 3.925. Signaling Trainer to stop.