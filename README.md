# bayesian-analysis-index

これまで取り組んできたベイズ分析プロジェクト群を横断する、自分用の技術リファレンス。

各プロジェクトのREADMEには「事前分布のスケールをどう決めたか」「なぜ暴走したか、どう直したか」といった手法・失敗パターンが個別に埋もれている。ここでは、それらを**手法・テクニック別**に串刺しにして、「この症状、前にも見た」を即座に引けるようにする。

## 使い方

各カテゴリファイルには、以下の固定フォーマットでエントリが並んでいる。

```
### タイトル
- 症状: 何が起きたか
- 対処: どうしたか
- なぜ効くか: 背景にある原理
- 登場プロジェクト: 元READMEの該当セクションへのリンク
```

内容は元READMEの要約であり、詳細は登場プロジェクトのリンク先を参照する(ハイブリッド方式: 一次情報は各プロジェクトのリポジトリ、ここは検索用の索引)。

## 分析の種類

「手法・テクニック別」「道具別」に串刺しにする前に、そもそも各プロジェクトがどんな**種類**のベイズ分析なのか(機構論的モデル、階層ベイズ、状態空間モデル、生存時間分析、点過程、バンディット/OPE、ベイズ回帰・A/Bテスト)を俯瞰できるカタログ。

→ [analysis-types.md](analysis-types.md)

## カテゴリ一覧

| カテゴリ | 内容 |
|---|---|
| [モデリング前のEDA(探索的データ分析)](techniques/eda.md) | 尤度分布選択・説明変数間の相関・季節性確認など、モデル構造を決める前のデータ理解 |
| [事前分布設計・prior predictive check](techniques/prior-predictive-check.md) | 事前分布のスケール決定、サンプリング前の検証 |
| [尤度・観測モデル選択](techniques/observation-model.md) | 観測データをモデルのどの量に対応付けるか、分布族の選択 |
| [パラメータ化・非識別性対策](techniques/reparameterization.md) | 比が意味を持つ量の再パラメータ化、値の矛盾の構造的排除 |
| [診断・収束判定](techniques/diagnostics.md) | r_hat/ESS/divergence、原因不明の異常を切り分ける診断プロセス |
| [モデル評価・比較](techniques/model-evaluation.md) | LOO/AUC/Brier scoreなど集計指標の使い方と限界 |
| [データ・単位・前処理の落とし穴](techniques/data-pitfalls.md) | 単位の取り違え、正規化の見落とし、外れ値・因果解釈の注意点 |
| [実装上のハック](techniques/implementation-hacks.md) | PyMC/ArviZ/JAX/pytensor固有のバグ回避・実装テクニック |

## 道具一覧

「手法・テクニック別(症状/対処)」の`techniques/`とは別に、分析で使う手段そのもの(評価指標・推定量など)の定義・仕組みを引くための用語辞典。

| カテゴリ | 内容 |
|---|---|
| [評価指標・推定量](tools/evaluation-metrics.md) | LOO/AUC-ROC/Brier Score/C-indexなどのモデル比較指標、IPS/DM/DR/SNIPS/SNDRなどのOPE推定量の定義と使い分け |
| [観測モデル・尤度分布](tools/observation-models.md) | Poisson/Beta-Binomial/Gamma-Poisson/Dirichlet-Multinomialなどの分布族、Weibull/Piecewise Exponentialなどのハザードモデル、Hawkes過程・forward algorithmの定義と使い分け |
| [ギリシャ文字の用途一覧](tools/greek-letters.md) | α/β/κ/λ/μ/σ/φなど、各プロジェクトでよく使う記号が何を表すかの早見表 |
| [MCMC診断指標](tools/mcmc-diagnostics.md) | r_hat/ESS/divergence/target_acceptの定義と使い分け |
| [推論エンジン・サンプリング手法](tools/inference-methods.md) | NUTS/ADVI/Compound Step/Replay法/Thompson Samplingの定義と使い分け |
| [事後分布の幾何学的病理](tools/posterior-pathologies.md) | Funnel/Ridge型非識別性/ラベルスイッチング/マルチモダリティの定義と使い分け |
| [統計的バイアス・概念](tools/statistical-biases.md) | Jensen不等式/Informative Censoring/傾向スコア/Ecological Bias/IPCWの定義と使い分け |
| [状態空間モデルの型](tools/state-space-models.md) | 変化点モデル/GaussianRandomWalk/非線形状態空間/Markov-Switching/Stochastic Volatilityの定義と使い分け |
| [事前分布の選び方](tools/prior-distributions.md) | Beta/HalfNormal・HalfCauchy・Gamma/DiscreteUniform/Normal/LogNormalなど、パラメータの性質ごとの事前分布の選び方 |
| [空間モデルの型](tools/spatial-models.md) | ICAR/BYM/BYM2/空間時系列BYM(Knorr-Held Type I〜IV)/LGCPの定義と使い分け |
| [欠測データ処理の型](tools/missing-data.md) | MCAR/MAR/MNAR(欠測メカニズムの分類)、フルベイズ同時モデル/MICE/Selection Model/Pattern-Mixture Model/多変量同時欠測モデルの定義と使い分け |
| [獲得関数の型](tools/acquisition-functions.md) | PI/EI/UCB/GP版Thompson Samplingの定義と使い分け |

## 演習問題(コーディングテスト)

`techniques/`・`tools/`の各エントリを実際にコードを書いて理解度確認できる自己学習用演習。push/PR時にGitHub Actionsが自動採点する。

→ [exercises/](exercises/)

## 対象プロジェクト

| プロジェクト | 概要 | index取り込み状況 |
|---|---|:---:|
| [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md) | PyMCによるベイズ機構論的感染症モデリング(SIR/SIS/SIRS/SEIR) | ✅ |
| [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md) | ベイズ統計モデリングによる広告A/BテストのCVR分析 | ✅ |
| [bayesian-hazard-models](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/README.md) | ベイズ生存時間分析(Telco Customer Churn) | ✅ |
| [bitcoin-utxo-survival](https://github.com/karahashimanato/bitcoin-utxo-survival/blob/main/README.md) | UTXO滞留時間のベイズ生存時間分析(bayesian-hazard-modelsの延長) | ✅ |
| [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md) | 多腕バンディット・オフ方策評価、階層ベイズモデル比較 | ✅ |
| [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md) | ベイズ時系列・階層モデル・点過程の学習ジャーナル(旧markov-regime-switching) | ✅ |
| [bayesian-causal-inference](https://github.com/karahashimanato/bayesian-causal-inference/blob/main/README.md) | BigQuery公開データセットを用いたBSTS因果推論、半合成デザインによる検出力キャリブレーション | ✅ |
| [bayesian-gaussian-process](https://github.com/karahashimanato/bayesian-gaussian-process/blob/main/README.md) | PyMCによるベイズ・ガウス過程(GP)モデリング。標準RBF/複合カーネル/非ガウス尤度(Poisson)/スパースGPの4ケーススタディ | ✅ |
| [bayesian-spatial-models](https://github.com/karahashimanato/bayesian-spatial-models/blob/main/README.md) | PyMCによるベイズ空間統計。ICAR→BYM→BYM2(スコットランド口唇癌)、空間時系列BYM(オハイオ州COVID-19)、空間点過程LGCP(能登半島地震) | ✅ |
| [bayesian-missing-data](https://github.com/karahashimanato/bayesian-missing-data/blob/main/README.md) | 欠測データ処理(Missing Data)のベイズ分析。World Bank開発指標パネルデータでMCAR/MAR/MNARを横断検証し、フルベイズ同時モデル・MICE・Selection Model・Pattern-Mixture Model・多変量同時欠測モデルを実装 | ✅ |
| [bayesian-optimization](https://github.com/karahashimanato/bayesian-optimization/blob/main/README.md) | PyMCによるベイズ最適化。GP代理モデル+獲得関数(PI/EI/UCB/GP版Thompson Sampling)を1次元/2次元Branin/6次元Hartmannベンチマークで検証し、XGBoostハイパーパラメータ探索に実応用 | ✅ |
| [bayesian-deep-learning](https://github.com/karahashimanato/bayesian-deep-learning/blob/main/README.md) | ベイズ深層学習における不確実性定量化の手法比較(PyTorch)。MC Dropout/Deep Ensembles/Laplace近似/Bayes by Backpropを1次元合成回帰・California Housing・MNIST vs Fashion-MNIST・時系列で比較 | ✅ |

## 運用ルール

- 新しいベイズ分析プロジェクトが一段落したら、そのREADMEの「学び」「ハック」に相当する箇所を該当カテゴリに追記する。
- 既存カテゴリに当てはまらない手法が3件以上溜まったら、新しいカテゴリファイルへの分割を検討する(`techniques/`・`tools/`共通のルール)。
- 新しく登場した評価指標・推定量は`tools/`に追記し、関連する`techniques/`の教訓エントリからも相互リンクを張る。
