# ギリシャ文字の用途一覧

各プロジェクトでよく使うギリシャ文字が、実際に何を表すために使われているかの早見表。同じ記号でもプロジェクト・モデルによって意味が異なることが多いため、**1記号×1意味=1行**で並べている。「この記号、前にどう使ったっけ」を引くための一覧であり、`tools/evaluation-metrics.md`・`tools/observation-models.md`のような定義型カードとは違う形式。

| 記号 | 読み | よく表す意味 | 登場プロジェクト |
|:---:|---|---|---|
| α | アルファ | Beta分布の形状パラメータ(集中度κと組み合わせて `α = μκ` の形で使われることが多い) | [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#分析の流れnotebooks) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#mlb打率-階層ベイズbeta-binomial) |
| α(`α_conc`) | アルファ・コンク | Gamma-Poisson/Dirichlet-Multinomialの集中度(concentration)パラメータ | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#サメ襲撃件数-階層ベイズgamma-poisson)(サメ) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#ポケモンカード封入率-階層ベイズdirichlet-multinomial)(ポケモン) |
| β | ベータ | SIR系モデルの感染率(transmission rate) | [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#sir-eyamペスト流行1666) |
| β | ベータ | 生存時間分析の回帰係数(log hazard ratio、`exp(β)`がハザード比) | [bayesian-hazard-models](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/README.md#モデル一覧) |
| β | ベータ | Beta分布の第2パラメータ | [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#分析の流れnotebooks) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#mlb打率-階層ベイズbeta-binomial) |
| β | ベータ | Hawkes過程の減衰速度(decay rate) | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#能登半島地震-自己励起点過程hawkesetas) |
| γ | ガンマ | SIR系モデルの回復率(recovery rate) | [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#sir-eyamペスト流行1666) |
| δ | デルタ | 季節変動モデルの振幅(seasonal amplitude) | [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#sirs-米国季節性インフルエンザ)(SIRS) |
| ε | イプシロン | 比が意味を持つ量の再パラメータ化における小さな揺らぎ(`R0 = 1 + ε` など) | [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#sis-ナイジェリア-マラリア罹患率) |
| ε | イプシロン | ε-greedyの強制ランダム選択確率(探索の下限保証) | [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#主な発見) |
| η | イータ | SVモデルのprocess noise標準偏差(`σ_η`の形で登場) | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-stochastic-volatility) |
| η | イータ | GPカーネルの振幅(amplitude、`eta_trend`/`eta_season`等) | [bayesian-gaussian-process](https://github.com/karahashimanato/bayesian-gaussian-process/blob/main/README.md#複合カーネル-mauna-loa-co2濃度) |
| η | イータ | LGCPのGPカーネルの分散(variance) | [bayesian-spatial-models](https://github.com/karahashimanato/bayesian-spatial-models/blob/main/README.md#part-3-空間点過程lgcp能登半島地震) |
| θ | シータ | BYMモデルの地区固有の非構造項(unstructured heterogeneity) | [bayesian-spatial-models](https://github.com/karahashimanato/bayesian-spatial-models/blob/main/README.md#1-bymの非識別性θφの分離が決まりにくい) |
| κ | カッパ | Beta-Binomialの集中度(concentration、大きいほど個体差が小さい) | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#mlb打率-階層ベイズbeta-binomial) |
| κ | カッパ | Hawkes過程の励起強度(excitation strength) | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#能登半島地震-自己励起点過程hawkesetas) |
| λ | ラムダ | Exponential/Weibullハザード関数のレート・スケールパラメータ | [bayesian-hazard-models](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/README.md#モデル一覧) |
| λ | ラムダ | Poisson分布・Hawkes過程の発生率/強度関数 | [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#sir-eyamペスト流行1666) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#能登半島地震-自己励起点過程hawkesetas) |
| μ | ミュー | Beta-Binomial/Gamma-Poissonの全体平均 | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#mlb打率-階層ベイズbeta-binomial)(MLB) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#サメ襲撃件数-階層ベイズgamma-poisson)(サメ) |
| μ | ミュー | GPの平均関数(mean function)の固定値、潜在関数`f`の中心 | [bayesian-gaussian-process](https://github.com/karahashimanato/bayesian-gaussian-process/blob/main/README.md#非ガウス尤度ポアソン-山火事件発生件数) |
| ξ | クサイ | 免疫喪失率(waning immunity rate) | [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#sirs-米国季節性インフルエンザ)(SIRS) |
| π | パイ | オフ方策評価(OPE)における方策(評価方策`π_e`、ログ収集方策`π_b`) | [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#notebook構成) |
| ρ | ロー | leverage effect(観測ノイズ間の非対称な相関) | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-stochastic-volatility) |
| ρ | ロー | BYM2の空間分散の割合(全体スケールσのうち空間構造項が占める比率) | [bayesian-spatial-models](https://github.com/karahashimanato/bayesian-spatial-models/blob/main/README.md#2-bym2による解消) |
| σ | シグマ | 標準偏差・ボラティリティ全般(MSMのレジーム別`σ_0`/`σ_1`など) | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-markov-switching-model) |
| σ | シグマ | GP尤度の観測ノイズ標準偏差 | [bayesian-gaussian-process](https://github.com/karahashimanato/bayesian-gaussian-process/blob/main/README.md#標準rbfカーネル-世界平均気温偏差) |
| σ | シグマ | SEIRモデルの潜伏期からの遷移率(1/潜伏期間、E→Iの速度) | [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#seir-湖北省covid-19初期流行) |
| φ | ファイ | AR(1)の持続性パラメータ(persistence、絶対値が1未満で定常) | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-stochastic-volatility) |
| φ | ファイ | 季節変動モデルの位相(phase) | [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#sirs-米国季節性インフルエンザ)(SIRS) |
| φ | ファイ | ICAR/BYMの空間構造項(隣接地区間で相関する空間効果) | [bayesian-spatial-models](https://github.com/karahashimanato/bayesian-spatial-models/blob/main/README.md#part-1-bym2スコットランド口唇癌データ) |
| ψ | プサイ | 空間時系列BYMの空間×時間交互作用項 | [bayesian-spatial-models](https://github.com/karahashimanato/bayesian-spatial-models/blob/main/README.md#part-2-空間時系列bymオハイオ州covid-19) |
| ω | オメガ | 周期モデルの角周波数(angular frequency) | [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#sunspot-周期性を持つ非線形状態空間モデル)(Sunspot) |

## 関連

- 記号そのものの数式的な扱い(比への再パラメータ化など)は[techniques/reparameterization.md](../techniques/reparameterization.md)を参照。
- 分布族としての定義(Beta-Binomialのα/β、Gamma-Poissonのα_conc/μなど)は[tools/observation-models.md](observation-models.md)を参照。
- OPE推定量の`π_e`/`π_b`の数式的な使われ方は[tools/evaluation-metrics.md](evaluation-metrics.md#ips-inverse-propensity-scoring)を参照。
