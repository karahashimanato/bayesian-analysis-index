# モデリング前のEDA(探索的データ分析)

ベイズモデルを組み始める前に、データの分布の形・説明変数間の相関・周期性をどう確認し、それをモデル設計(尤度分布・説明変数の取捨選択・構造)にどう反映するか。事前分布のスケール決定([techniques/prior-predictive-check.md](prior-predictive-check.md#定常状態の解析式に実データを代入して事前分布の中心値を決める))や外れ値の扱い([techniques/data-pitfalls.md](data-pitfalls.md#外れ値は除外する前に中身を見る))は既存カテゴリを参照。ここでは「モデルの構造そのものを決める前段」としてのEDAに絞る。

## EDAの実施手順

モデルを組む前に、以下の順で確認すると手戻りが少ない(各プロジェクトで実際に踏んだ手順から抽出)。

1. **基礎集計**: サンプルサイズ・グループ別件数・欠損の有無をまず確認する。[bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#分析の流れnotebooks)の`eda.ipynb`は群間サンプルサイズの確認から始まり、[Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#notebook構成)の`eda_basic_aggregation.ipynb`も腕の数・全体CTR・腕別成績の基礎集計を最初のノートブックに置いている。
2. **単位・母数の確認**: 指標が何あたりの値か(パーセントか件数か、何人あたりか)をデータソースの定義に立ち返って確認する。→ [techniques/data-pitfalls.md](data-pitfalls.md#指標の母数単位を見落とすとパラメータ推定が丸ごと狂う)
3. **分布の形を見る**: 分散/平均比(過分散)や裾の重さを確認し、尤度分布の候補を絞る。→ [下記](#分散平均比過分散や分布の裾の重さは尤度分布を決める前にedaで確認する)
4. **外れ値の中身を見る**: 統計的な外れ値がノイズか、意味のある高価値セグメントかを判断する。→ [techniques/data-pitfalls.md](data-pitfalls.md#外れ値は除外する前に中身を見る)
5. **説明変数間の相関を確認する**: 多重共線性のリスクがある組み合わせを洗い出す。→ [下記](#候補となる説明変数間の相関はモデルに組み込む前にedaで確認し多重共線性リスクを除外基準にする)
6. **周期性・季節性を確認する**: 時系列であれば、モデルの成分構成(トレンドのみか周期成分を含むか)に反映する。→ [下記](#季節性周期性の有無をedaで確認しモデルの成分構成に反映する)
7. **ランダム化の有無を確認する**: 各説明変数が実験的に割り当てられたか、単なる観測変数かを区別する(因果と相関のどちらとして解釈できるかが変わる)。→ [techniques/data-pitfalls.md](data-pitfalls.md#ランダム化されていない観測変数から因果を主張しない)
8. **数値をモデル設計に持ち込む**: ここで得た増減速度・平衡水準・季節振幅などの数値を、事前分布の中心値の逆算に使う。→ [techniques/prior-predictive-check.md](prior-predictive-check.md#定常状態の解析式に実データを代入して事前分布の中心値を決める)

---

### 分散/平均比(過分散)や分布の裾の重さは、尤度分布を決める前にEDAで確認する

- **症状**: 尤度分布(Poissonなど)を先に決めてからモデルを組むと、実データの分散が平均を大きく上回る過分散や、強い右裾を持つ分布形状と前提が矛盾し、モデルの土台から崩れる。
- **対処**: モデリングに入る前のEDAで分散/平均比や分布の形を定量的に確認する。bayesian-causal-inferenceでは日次アウトカム(Apparelカテゴリの商品詳細ページビュー数)の分散/平均比が約68と判明し、Poissonを不採用としてlog1pスケールのガウス状態空間モデルを採用した。bayesian-A-B-testingでは`eda.ipynb`で`total ads`の分布が強い右裾であることを最初に確認している。
- **なぜ効くか**: 尤度分布の選択はモデル構造の根幹であり、サンプリング後の診断で矛盾に気づくと設計の大幅な作り直しになる。EDAで分散・裾の形を先に定量化しておけば、尤度分布の候補をモデルを組む前に絞り込める。
- **登場プロジェクト**: [bayesian-causal-inference](https://github.com/karahashimanato/bayesian-causal-inference/blob/main/README.md#結果) / [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#分析の流れnotebooks)

---

### 候補となる説明変数間の相関は、モデルに組み込む前にEDAで確認し多重共線性リスクを除外基準にする

- **症状**: 対照変数・説明変数の候補が複数ある場合、相関を確認せずにまとめてモデルへ投入すると、多重共線性により個々の係数の事後分布が不必要に広がる。
- **対処**: モデルに投入する前に候補変数間の相関をEDAで計算し、相関が高い組み合わせがあれば一方を除外する。bayesian-causal-inferenceでは対照系列の候補(`office`/`electronics`/`google_brand`)の相関を確認し、`electronics`との相関が0.95と高かった`office`を候補から除外した。
- **なぜ効くか**: 相関の高い説明変数を同時に投入すると、それぞれの効果を分離する情報がデータに乏しくなり、事後分布の分散が膨らむ(非識別性に近い状態になる)。モデル構築後の診断で原因を切り分けるより、EDA段階で相関構造を把握して候補を絞る方がコストが低い。
- **登場プロジェクト**: [bayesian-causal-inference](https://github.com/karahashimanato/bayesian-causal-inference/blob/main/README.md#分析デザイン)

---

### 季節性・周期性の有無をEDAで確認し、モデルの成分構成に反映する

- **症状**: 時系列モデルの構造(ローカルレベルのみか、周期成分を含むか)を勘で決め打ちすると、実在する周期パターンを取りこぼしたり、逆に存在しない周期性のために不要な成分でモデルを複雑にしたりする。
- **対処**: モデル構造を決める前にEDAで周期パターンの有無と大きさを確認する。bayesian-causal-inferenceでは週次季節性(平日340〜390件 vs 週末225〜247件)をEDAで確認した上で、`ZeroSumNormal`による曜日固定効果をモデルに組み込んだ。年間トレンドより週次季節性の寄与が支配的だと分かっていたため、ローカルレベルもlocal linear trendではなくlocal levelのみを採用している。
- **なぜ効くか**: データに実在する周期構造を先に定量化しておくことで、モデルのどの成分がその変動を説明すべきかを設計段階で決められる。事後の診断で「モデルが季節性を捉えられていない」ことに気づいてから成分を追加する手戻りを避けられる。
- **登場プロジェクト**: [bayesian-causal-inference](https://github.com/karahashimanato/bayesian-causal-inference/blob/main/README.md#結果)
