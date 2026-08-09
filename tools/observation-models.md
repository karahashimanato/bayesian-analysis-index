# 観測モデル・尤度分布

観測データをモデルのどの分布・尤度で結びつけるかの用語辞典。`techniques/observation-model.md`が「症状/対処」型の教訓集であるのに対し、こちらは各分布・尤度パターンそのものの定義・仕組み・使い分けを引くためのリファレンス。分布を選ぶ前提となるEDA(過分散・裾の重さの確認など)は[techniques/eda.md](../techniques/eda.md)を参照。

## 尤度分布を見分けるフローチャート

観測データの性質から、どの尤度分布に当たるかを見分けるための特徴質問チャート。各entryの「使い分け」で説明している判断基準(過分散の有無、個体間のばらつきの有無など)を経路にしたもので、上から順に絞り込む決定木として使える。

```mermaid
flowchart LR
    Start(["観測データの性質は?"])

    Start -->|"連続時間上のイベント発生<br/>時刻そのもの(地震の余震など)"| Hawkes["Hawkes過程"]
    Start -->|"『イベントまでの時間』が目的変数<br/>(打ち切りデータあり)"| Hazard["生存時間分析<br/>(ハザード関数)"]
    Start -->|"時間とともに切り替わる<br/>離散レジームがある"| Forward["forward algorithm<br/>(離散潜在状態の周辺化)"]
    Start -->|"離散イベントの発生回数<br/>(カウントデータ)"| Count["Poisson系"]
    Start -->|"二値の成功/失敗、<br/>または成功回数"| Binary["Bernoulli/Binomial系"]
    Start -->|"複数カテゴリへの配分比率"| Multi["Dirichlet-Multinomial"]
    Start -->|"連続値(残差・リターンなど)"| Cont["Normal/Student-t系"]

    Count -->|"分散が平均とほぼ等しい"| Poisson["Poisson"]
    Count -->|"分散が平均を大きく上回る<br/>(overdispersion)"| GP["Gamma-Poisson"]

    Binary -->|"成功確率pを固定値として扱う"| BB["Bernoulli / Binomial"]
    Binary -->|"個体・グループ間で<br/>成功確率がばらつく"| BetaBinom["Beta-Binomial"]

    Cont -->|"外れ値に頑健にしたい<br/>(裾が重いと疑われる)"| StudentT["Student-t"]
    Cont -->|"特に問題なし"| Normal["Normal"]

    Hazard -->|"ハザード率は時間で一定でよい"| Exp["Exponential"]
    Hazard -->|"時間とともに単調に増減<br/>(KM曲線と系統的にズレる)"| Weib["Weibull"]
    Hazard -->|"U字型など、単純な関数形<br/>では捉えきれない形状"| PE["Piecewise Exponential"]
    PE -->|"共変量だけで説明できない<br/>グループ間の系統差がある"| Frail["Frailty"]
```

- [Poisson](#poisson) / [Bernoulli / Binomial](#bernoulli--binomial) / [Beta-Binomial](#beta-binomial) / [Gamma-Poisson](#gamma-poisson負の二項分布相当) / [Dirichlet-Multinomial](#dirichlet-multinomial) / [Normal / Student-t](#normal--student-t観測分布) / [Exponential](#exponentialハザード関数) / [Weibull](#weibullハザード関数) / [Piecewise Exponential](#piecewise-exponentialcox比例ハザード) / [Frailty](#frailty変量効果) / [Hawkes過程](#hawkes過程点過程の尤度) / [forward algorithm](#forward-algorithm離散潜在状態の周辺化尤度)

---

### Poisson

![Poisson: 分散/平均比=0.91(等分散)のデータに対しPyMCでフィットしたPoissonモデルの事後予測分布(青線)は観測データ(紫)の形をよく捉え、95%予測区間のカバレッジは99%](../assets/observation-models/poisson_equidispersion_fit.png)

*PyMCで実際にサンプリングし事後予測チェック(PPC)を行った結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。等分散データではPoissonの制約が問題にならないことを示す、隣接する[Gamma-Poisson](#gamma-poisson負の二項分布相当)エントリ(過分散データでのPPC過小評価)との対比。*

- **定義**: 単位時間・単位区間あたりに起こる離散イベントの回数をモデル化する分布。平均と分散が等しい(equidispersion)という制約を持つ。
- **数式・仕組み**: `P(k) = λ^k * e^{-λ} / k!`。連続時間モデル(SIR/SIS/SIRSなど)では、状態変数の予測値(感染者数I(t)など)を`λ`とするPoisson尤度で観測データと結びつけることが多い。
- **使い分け**: 観測プロセスが同一(同じ記録方法)である複数の観測変数には分布族を統一する。分散が平均を大きく上回る(overdispersion)場合は[Gamma-Poisson](#gamma-poisson負の二項分布相当)への切り替えを検討する。実際、SIRモデルでは「分散=平均」という制約が、急峻なピーク高さの過小評価に寄与した可能性が確認されている。ガウス過程回帰の潜在関数`f(x)`に対して`y ~ Poisson(exp(f(x)))`のように結びつける非ガウス尤度としても使われ、この場合は`pm.gp.Marginal`の解析的周辺化が使えず、潜在関数を明示的にサンプリングする必要がある([tools/inference-methods.md](inference-methods.md#pmgplatent--hsgp基底関数近似)参照)。
- **登場プロジェクト**: [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#sir-eyamペスト流行1666) / [bayesian-gaussian-process](https://github.com/karahashimanato/bayesian-gaussian-process/blob/main/README.md#非ガウス尤度ポアソン-山火事件発生件数)

---

### Bernoulli / Binomial

![Bernoulli(行単位)とBinomial(集計値)の事後分布は完全に重なる: 事後平均はBernoulli尤度0.361、Binomial尤度0.362(真値p=0.35)](../assets/observation-models/bernoulli_binomial_equivalence.png)

*同一データを行単位のBernoulli尤度と集計済みのBinomial尤度でそれぞれ実際にPyMCでフィットし比較した結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: 二値の成功/失敗(クリック有無、勝敗など)を1回ずつモデル化するのがBernoulli、n回の試行のうちの成功回数をまとめてモデル化するのがBinomial。
- **数式・仕組み**: `Bernoulli(p): P(x=1)=p`、`Binomial(n,p): P(k)=C(n,k) p^k (1-p)^{n-k}`。数学的には同一の尤度に帰着する。
- **使い分け**: 個々の試行を行単位で扱うか、集計済みの成功回数として扱うかで選ぶ。共役事前分布としてBetaを使うと解析的な事後計算ができる([Beta-Binomial](#beta-binomial)参照)。
- **登場プロジェクト**: [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#notebook構成)(Beta-Bernoulli Thompson Sampling) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#mlb打率-階層ベイズbeta-binomial)(打率のBinomial尤度)

---

### Beta-Binomial

![階層Beta-Binomialモデルによる部分プーリング: 観測回数が少ない個体(紫、n≤15)ほど事後平均が全体平均へ強く縮小(平均|縮小幅|=0.104)し、観測回数が多い個体(黄緑〜黄、n≥300)はほぼ観測値のまま(平均|縮小幅|=0.007)](../assets/observation-models/beta_binomial_shrinkage.png)

*PyMCで実際にサンプリングした結果(観測打席数10〜500の18個体を模したシミュレーション。生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: 成功確率`p`を固定値ではなくBeta分布に従う確率変数として扱う階層モデル。個体ごとに異なる成功確率のばらつきを表現できる。
- **数式・仕組み**: `p_i ~ Beta(α,β)`、`x_i ~ Binomial(n_i, p_i)`。`α, β`は「全体平均μ」「集中度κ」に再パラメータ化されることが多い(`α=μκ`, `β=(1-μ)κ`)。
- **使い分け**: 個体(打者、広告群など)ごとの観測回数が異なり、個体間のばらつきそのものを階層的にモデル化・部分プーリング(shrinkage)したい場合に使う。単純なBinomial(固定p)では個体差を無視してしまう。
- **登場プロジェクト**: [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#分析の流れnotebooks)(広告群 vs psa群のCVR比較) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#mlb打率-階層ベイズbeta-binomial)(MLB打率の階層モデル)

---

### Gamma-Poisson(負の二項分布相当)

![Gamma-Poissonによるoverdispersion補正: 分散/平均比2.20のカウントデータに対し、Poisson固定分散モデルの95%予測区間[2,12]は実測データの88%しかカバーしないが、Gamma-Poisson階層モデルの区間[1,16]は98%をカバーする](../assets/observation-models/gamma_poisson_overdispersion.png)

*PyMCで実際にサンプリングし事後予測チェック(PPC)を行った結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: Poissonの発生率`λ`を固定値ではなくGamma分布に従う確率変数として扱う階層モデル。overdispersion(分散>平均)を表現できる。
- **数式・仕組み**: `λ_i ~ Gamma(α_conc, β)`、`x_i ~ Poisson(λ_i)`。`λ`を周辺化するとNegative Binomial分布と数学的に同値になる。
- **使い分け**: カウントデータの分散が平均より大きく、かつ個体間のばらつきを階層的に表現したい場合に使う。集中度パラメータ`α_conc`が0に近づくと分散が発散する構造的な落とし穴があるので注意([techniques/prior-predictive-check.md](../techniques/prior-predictive-check.md#分母のパラメータが0に近づくと分散が発散する病理は分布族を問わず繰り返す)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#サメ襲撃件数-階層ベイズgamma-poisson)

---

### Dirichlet-Multinomial

![集中度パラメータ→0でDirichlet-Multinomialの分散(N=100固定)は頭打ち(N²π(1-π)=2100)になるが、Gamma-Poisson(総量Nの制約なし)の分散は発散する(集中度=0.01でDM分散2079 vs GP分散90030)](../assets/observation-models/dirichlet_multinomial_bounded_variance.png)

*Dirichlet-Multinomial分布とNegative Binomial(Gamma-Poisson)分布の解析的な分散公式を比較した結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: 複数カテゴリへの配分比率をDirichlet分布に従う確率変数として扱い、観測されたカテゴリ別カウントをMultinomial尤度で結びつける階層モデル。
- **数式・仕組み**: `p ~ Dirichlet(concentration * base_measure)`、`x ~ Multinomial(N, p)`。総量`N`がカテゴリに配分される構造のため、集中度パラメータが0に近づいても個々のカテゴリ値は`N`で頭打ちになり、[Gamma-Poisson](#gamma-poisson負の二項分布相当)のような分散発散が起こらない。
- **使い分け**: 複数カテゴリへの配分比率(封入率など)を、カテゴリ間の相関も考慮しつつモデル化したい場合に使う。「総量が固定か青天井か」でGamma-Poissonとの発散リスクの有無が変わる点を覚えておくと事前予測チェックの労力を節約できる。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#ポケモンカード封入率-階層ベイズdirichlet-multinomial)

---

### Normal / Student-t(観測分布)

![外れ値(5/60件、高レバレッジ側に一貫して下方へ外れる)に対しNormal観測分布は回帰直線が大きく引っ張られ推定傾きが真値1.2から0.537まで崩れるが、Student-t観測分布は真の傾きに近い1.220を保つ](../assets/observation-models/normal_vs_studentt_robustness.png)

*PyMCで実際にNormal回帰とStudent-t回帰(自由度νも推定)をフィットし比較した結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: 連続値の観測データをモデル化する基本分布。Normalは正規分布、Student-tは自由度パラメータ`ν`を持ち、Normalより裾が重い(外れ値に頑健な)分布。
- **数式・仕組み**: Student-tは`ν→∞`でNormalに収束する。金融時系列のようにファットテール(急激な変動)を持つデータでは、Normal観測分布だと外れ値の影響を過大評価してしまう。
- **使い分け**: 残差やリターンの分布が正規分布より裾が重いと疑われる場合はStudent-tを試す。ただし、裾の重さを変えるだけでは解決しない構造的な問題(モデルの表現力不足など)もあり、Student-tへの変更だけで狙った現象(ACFのギャップなど)が解消するとは限らない([techniques/model-evaluation.md](../techniques/model-evaluation.md#有意なパラメータと狙っていた問題の解決は別軸で検証する)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-stochastic-volatility)

---

### Exponential(ハザード関数)

![真のハザードが時間とともに増加する合成生存時間データに対し、Exponential(ハザード一定、λ=0.105)フィットはKaplan-Meier曲線と系統的にズレるが、Weibull(k=1.70、真値1.8)フィットは追従する](../assets/observation-models/hazard_exponential_vs_weibull.png)

*ExponentialとWeibullを両方実際にPyMCでフィットしKaplan-Meier曲線と比較した結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。この画像は次の[Weibull](#weibullハザード関数)エントリと共通。*

- **定義**: 生存時間分析における最も単純なハザード関数。時間によらずハザード率(瞬間イベント発生率)が一定であると仮定する。
- **数式・仕組み**: `h(t) = λ`(定数)。生存関数は`S(t) = exp(-λt)`。
- **使い分け**: モデルの基準点・最も制約の強いベースラインとして使う。Kaplan-Meier曲線との比較で系統的なズレ(序盤の急減衰・終盤の緩やかな減衰など)が見える場合、ハザード率一定という仮定がデータと整合しないサインであり、[Weibull](#weibullハザード関数)など時間依存のハザードへ拡張する。
- **登場プロジェクト**: [bayesian-hazard-models](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/README.md#-exponentialモデル)

---

### Weibull(ハザード関数)

![同じ比較画像: WeibullフィットはKaplan-Meier曲線に追従するが、Exponential(ハザード一定)は系統的にズレる](../assets/observation-models/hazard_exponential_vs_weibull.png)

*[Exponential](#exponentialハザード関数)エントリと同じ画像・同じ実験(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: ハザード率が時間とともに単調に増加/減少することを表現できる、[Exponential](#exponentialハザード関数)の一般化。
- **数式・仕組み**: `h(t) = (k/λ)(t/λ)^{k-1}`。形状パラメータ`k`が1より小さければハザードは時間とともに減少、1より大きければ増加(`k=1`でExponentialに一致)。
- **使い分け**: Exponential(ハザード一定)がKaplan-Meier曲線と系統的にズレる場合の次の選択肢。`k`の事後分布が1を明確に下回れば「時間とともにイベントが起きにくくなる」ことを定量的に示せる。
- **登場プロジェクト**: [bayesian-hazard-models](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/README.md#-weibullモデル)

---

### Piecewise Exponential(Cox比例ハザード)

![U字型の真のベースラインハザード(6区間)を持つ合成生存時間データに対し、Piecewise ExponentialモデルをPyMCで実際にフィットし、区間ごとの推定値(90%区間)がU字型の形状を正しく復元する](../assets/observation-models/piecewise_exponential_recovery.png)

*exposure matrix(区間ごとの滞在時間)によるPoisson-trick尤度をPyMCで実際にサンプリングした結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: 時間を複数の区間に区切り、区間ごとに異なる定数ハザード(ベースラインハザード)を推定するノンパラメトリックに近いアプローチ。Cox比例ハザードモデルのベイズ的表現として広く使われる。
- **数式・仕組み**: `h(t|x) = h_{0,k(t)} × exp(x^T β)`(`k(t)`は`t`が属する区間、`h_{0,k(t)}`は区間ごとのベースラインハザード、`exp(x^T β)`が共変量による比例的な補正)。各対象が実際に通過した区間ごとの滞在時間を表す行列(exposure matrix)で累積ハザードを積算して尤度を計算する。
- **使い分け**: Exponential/Weibullのような単純な関数形では捉えきれない、U字型などの複雑なベースラインハザードの形状を、強いパラメトリックな仮定を置かずに推定したい場合に使う。
- **登場プロジェクト**: [bayesian-hazard-models](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/README.md#-cox比例ハザード-piecewise-exponential) / [bitcoin-utxo-survival](https://github.com/karahashimanato/bitcoin-utxo-survival/blob/main/README.md#計算戦略二段構え)(UTXO滞留時間への同型モデルの転用)

---

### Frailty(変量効果)

![Frailtyモデルは5グループの真のfrailty z(0.5〜2.0)を90%区間でほぼ正しく復元する(divergence=0)。frailtyを無視した単一推定(baseline=0.1)では、グループ別の経験的ハザード率(0.05〜0.24)のばらつきを捉え損ねる](../assets/observation-models/frailty_group_recovery.png)

*グループごとに異なる乗算的frailtyを持つ合成生存時間データをPyMCで実際にフィットした結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: 個体・グループごとに観測されない異質性(frailty)があると仮定し、ベースラインハザードに乗算的な変量効果を導入する、[Piecewise Exponential](#piecewise-exponentialcox比例ハザード)の拡張。
- **数式・仕組み**: `h(t|x) = h_{0,k(t)} × exp(x^T β) × z_group`(`z`はグループごとの変量効果、平均1に制約されることが多い)。
- **使い分け**: 共変量(契約タイプなど)だけでは説明しきれないグループ間の系統差(支払方法ごとの解約しやすさなど)がある場合に導入する。LOOでの改善幅が有効パラメータ数の増加(過学習ペナルティ)に見合うか確認する([tools/evaluation-metrics.md](evaluation-metrics.md#loo-leave-one-out-cross-validation-psis-loo)参照)。
- **登場プロジェクト**: [bayesian-hazard-models](https://github.com/karahashimanato/bayesian-hazard-models/blob/main/README.md#4-支払方法paymentmethodごとの解約しやすさfrailty-z)

---

### Hawkes過程(点過程の尤度)

![Hawkes過程: Ogataのthinning法で生成した合成イベント系列(全60イベント)に対し、log L = Σlog λ(t_i) - ∫λ(t)dtをpm.Potentialで直接実装しPyMCでフィット。推定した強度関数(青)は真の強度関数(黒破線)のイベント直後のスパイクと減衰をほぼ正確に再現し、パラメータもμ=0.42(真値0.5)・κ=0.85(真値0.8)・β=1.45(真値1.2)と良好に復元する(divergence=0)](../assets/observation-models/hawkes_process_recovery.png)

*既製の確率分布に対応しない対数尤度をpm.Potentialで直接実装し、PyMCで実際にサンプリングした結果(生成スクリプト: [scripts/generate_observation_models_plots.py](../scripts/generate_observation_models_plots.py))。*

- **定義**: 過去のイベントが将来のイベント発生強度を一時的に押し上げる「自己励起性」を持つ連続時間の点過程。地震の余震などに使われる。
- **数式・仕組み**: 強度関数`λ(t) = μ + Σ_{t_i<t} κ・exp(-β(t-t_i))`(`μ`が背景強度、`κ`・`β`が励起の強さと減衰速度)。対数尤度は`log L = Σ_i log λ(t_i) - ∫_0^T λ(t)dt`で、既製の確率分布に対応しないため`pm.Potential`で直接実装する。
- **使い分け**: 離散時系列・横断データとは異なり、連続時間のイベント発生時刻そのもの(地震の発生時刻など)をモデル化したい場合に使う。`κ`/`β`のように2パラメータが似た形で強度を押し上げる構造だと、ridge型の非識別性が起きやすい点に注意([techniques/reparameterization.md](../techniques/reparameterization.md#比が意味を持つ量は比自体への再パラメータ化でridge型の非識別性を緩和する)参照)。同じ地震カタログの発生「時刻」ではなく発生「位置」の集中度を扱いたい場合は、空間点過程LGCP([tools/spatial-models.md](spatial-models.md#lgcplog-gaussian-cox-process空間点過程)参照)を使う。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#能登半島地震-自己励起点過程hawkesetas)

---

### forward algorithm(離散潜在状態の周辺化尤度)

![Markov-Switching Model: forward algorithmで周辺化した尤度をPyMCで実際にサンプリングし、遷移確率(持続性p_stay0=0.959/p_stay1=0.855、真値0.95/0.90)とレジーム平均(mu0=-0.01/mu1=2.962、真値0/3.0)をdivergence=0で復元する](../assets/state-space-models/markov_switching_transition_recovery.png)

*[tools/state-space-models.md](state-space-models.md#markov-switching-model)のMarkov-Switching Modelエントリと同じ画像・同じ実験(forward algorithmによる周辺化そのものがこのエントリの主題のため)。生成スクリプト: [scripts/generate_state_space_models_plots.py](../scripts/generate_state_space_models_plots.py)。*

- **定義**: レジーム(離散潜在状態)を持つモデル(Markov-Switching Modelなど)で、離散状態をMCMCで直接サンプリングせず、解析的に周辺化(積分消去)して連続パラメータだけをサンプリング対象にする尤度計算手法。
- **数式・仕組み**: 各時点の状態確率分布を`pytensor.scan`で逐次更新し(前の時点の状態確率×遷移確率×観測尤度)、最終的な対数周辺尤度を`pm.Potential`としてモデルに加える。離散潜在状態自体は事後的に別途復元できる。
- **使い分け**: 離散潜在状態をNUTSで直接サンプリングしようとすると、Compound Step(離散部分はMetropolis)によりESSが著しく低下する。forward algorithmで周辺化すれば、連続パラメータ(遷移確率・平均・分散)だけをNUTSでサンプリングできる、離散HMM系のベイズ実装の定石。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-markov-switching-model)
