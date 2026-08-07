# 統計的バイアス・概念

Jensen不等式、Informative Censoring、傾向スコアなど、`techniques/`の複数ファイルに症状としてバラバラに登場する統計的な概念そのものの用語辞典。定義・仕組み・使い分けを引くためのリファレンス。

---

### Jensen不等式(Jensen's Inequality)

![Jensen不等式によるロジスティック回帰の予測確率のズレ: 素朴な点推定sigmoid(β̄0+β̄1x)と事後平均E[sigmoid(β0+β1x)]は、係数の不確実性が乗る領域(x=3.7付近)で最大-0.086のズレを生む](../assets/statistical-biases/jensen_inequality_gap.png)

*PyMCで実際にベイズロジスティック回帰をサンプリングした結果(n=15の小サンプルであえて係数の事後分布を広くとり、ズレを見やすくしている。生成スクリプト: [scripts/generate_statistical_biases_plots.py](../scripts/generate_statistical_biases_plots.py))。*

- **定義**: 凸関数`f`について`f(E[X]) ≤ E[f(X)]`が成り立つという不等式(凹関数なら不等号は逆向き)。ベイズモデルでは、非線形なリンク関数(ロジスティック回帰のシグモイド関数など)を通した予測の期待値が、パラメータの点推定だけから素朴に計算した値とズレる原因になる。
- **数式・仕組み**: ロジスティック回帰`p = sigmoid(β0 + β1*x)`で、`β0`の点推定(事後平均)が正しくても、`β1`の事後分布に分散があると`E[sigmoid(β0 + β1*x)] ≠ sigmoid(β0 + E[β1]*x)`となる。sigmoidは区間によって凸/凹が入れ替わる非線形関数のため、他パラメータの不確実性(分散)が加わるだけで予測平均がズレる。
- **使い分け**: 非線形なリンク関数を持つモデルで、事前予測チェックの予測平均が「係数の点推定から素朴に計算した値」とズレていても、それ自体はバグではなくJensen不等式による正常な挙動である可能性を疑う。サンプリング後の診断だけでは気づきにくいため、事前予測チェックの段階で確認する([techniques/prior-predictive-check.md](../techniques/prior-predictive-check.md#事前予測チェックはサンプリング前に必ず実施する)参照)。
- **登場プロジェクト**: [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#分析の流れnotebooks)(ベイズロジスティック回帰)

---

### Informative Censoring(情報を持つ打ち切り)

- **定義**: 生存時間分析において、打ち切り(観測終了時点でイベント未発生)が起こるかどうかが、分析対象の真の性質(イベントの起こりやすさ)と統計的に無関係ではない(=打ち切り自体が情報を持つ)状態。多くの生存時間分析手法は打ち切りが「非情報的(non-informative)」であることを前提にしている。
- **数式・仕組み**: 打ち切りメカニズムが分析対象のハザードと相関していると、観測データ(打ち切られずに残ったデータ)の分布が母集団の真の分布から系統的にズレる(選択バイアスの一種)。恣意的なヒューリスティックでデータを除外すると、除外基準自体が対象の性質と相関し、除外されなかったデータに対して事実上のinformative censoringを持ち込んでしまう。
- **使い分け**: データの除外基準は「客観的でプロトコル・仕様由来のフラグ」に限定し、推測に基づく除外(「これはノイズだろう」)は避ける。除外基準が対象の値そのものと無関係に決まることを確認できれば、informative censoringのリスクを排除できる([techniques/data-pitfalls.md](../techniques/data-pitfalls.md#除外基準は客観的なフラグに限定し恣意的な推測での除外を避ける)参照)。
- **登場プロジェクト**: [bitcoin-utxo-survival](https://github.com/karahashimanato/bitcoin-utxo-survival/blob/main/README.md#データの定義除外方針)

---

### Propensity Score(傾向スコア)

- **定義**: 各サンプルが、実際に採用された行動・処置(広告の表示位置、割り当てられた腕など)を、ログ収集時の方策のもとでどれくらいの確率で選ばれていたかを表す値。オフ方策評価(OPE)における重み付け補正の基礎になる。
- **数式・仕組み**: `π_b(a|x)`(ログ収集方策がコンテキスト`x`で行動`a`を選ぶ確率)として定義される。[IPS/DR/SNIPS](evaluation-metrics.md#ips-inverse-propensity-scoring)などの推定量は、この値の逆数で観測報酬を重み付けすることで、ログ収集方策と評価したい方策のズレを補正する。
- **使い分け**: 傾向スコアは条件(表示位置など)によって分布が異なりうるため、一様に扱わず、どの条件で分布が変わるかを事前に確認する必要がある。傾向スコアの推定精度・分布の妥当性にOPE推定量の結果がそのまま依存するため、この値自体の性質を疑わずに使うとバイアスや高分散の原因になる([techniques/data-pitfalls.md](../techniques/data-pitfalls.md#傾向スコアpropensity-scoreなど補正用の値は条件によって分布が異なる点に注意する)参照)。
- **登場プロジェクト**: [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#実装上の注意点)

---

### Ecological Bias(生態学的錯誤)

- **定義**: 集計データ(グループ単位の平均・合計など)から導いた関係性を、そのまま個体レベルの関係性として解釈してしまうことで生じるバイアス。集計の粒度を上げるほど、個体間の異質性が打ち消し合い、真の個体レベルの関係が歪んで見えることがある。
- **数式・仕組み**: 集計テーブル(BigQuery側で作った区間×金額ビンごとのexposure数・イベント数など)に対してモデルをフィットすると計算コストは大きく下がるが、集計の粒度自体が個体レベルの分布(ビン内でのばらつき)を消してしまっているため、集計モデルの推定が個体レベルの真の関係を正しく代表しているとは限らない。
- **使い分け**: 大規模データを直接ローカルで扱えない場合、(1)集計してローカルで軽量にモデルをフィットするステージと、(2)層化サンプリングした生の個体レベルデータに[SVI](inference-methods.md#advi--mean-field変分推論svi)などでモデルをフィットし、(1)の集計が真の関係を歪めていないか検証するステージの両方を用意する([techniques/model-evaluation.md](../techniques/model-evaluation.md#独立した外部データ公開指標との突き合わせで妥当性を検証する)参照)。
- **登場プロジェクト**: [bitcoin-utxo-survival](https://github.com/karahashimanato/bitcoin-utxo-survival/blob/main/README.md#計算戦略二段構え)

---

### IPCW(逆確率重み付け、Inverse Probability of Censoring Weighting)

- **定義**: 生存時間分析の評価指標(Brier Score、時間依存性AUCなど)を計算する際、打ち切りによって「本来観測されるはずだった結果」が欠測していることの影響を補正する重み付け手法。打ち切りが起きる確率の逆数で重み付けすることで、打ち切りがなかった場合の指標を近似的に復元する。
- **数式・仕組み**: 打ち切り生存確率`G(t)`(=ある時点`t`まで打ち切られずに残っている確率)を別途推定し、その逆数`1/G(t)`を評価指標の計算に重みとして使う。最大生存時間に生存者が集中している(全員が打ち切られている)ケースでは`G(t_max) = 0`となり、逆数計算がゼロ除算でクラッシュする。
- **使い分け**: [Brier Score](evaluation-metrics.md#brier-score)・[Time-Dependent AUC](evaluation-metrics.md#c-index-time-dependent-auc)など、打ち切りを含む生存時間データで確率較正・順位付けの精度を評価する指標を計算する際の標準的な補正手法として使う。ゼロ除算の実務的な回避策としては、評価対象の最大時間をごくわずかにクリップする方法がある([techniques/implementation-hacks.md](../techniques/implementation-hacks.md#打ち切り時刻ちょうどのゼロ除算をわずかなクリップで回避する)参照)。
- **登場プロジェクト**: [bayesian-hazard-models](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/README.md#診断実装における重要ハック)
