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
