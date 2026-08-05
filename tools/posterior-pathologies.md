# 事後分布の幾何学的病理

funnel・ridge型非識別性・ラベルスイッチング・マルチモダリティなど、事後分布の"形"そのものに起因してサンプリングを不安定にする、名前のついた病理の用語辞典。`techniques/reparameterization.md`・`techniques/diagnostics.md`が「症状/対処」型の教訓集であるのに対し、こちらは各病理そのものの定義・仕組みを引くためのリファレンス。

---

### Funnel(漏斗状の病理、Neal's funnel)

- **定義**: 階層モデルで、グループ間のばらつきを表すスケールパラメータ(σ)が0に近づくにつれて、個々のグループ効果の事後分布が急激に細くなる(漏斗状になる)幾何学的病理。中心化パラメータ化(グループ効果を「全体平均+誤差」として直接表現する)の階層モデルで典型的に発生する。
- **数式・仕組み**: 階層モデル`θ_i ~ Normal(μ, σ)`を中心化パラメータ化のまま素朴にサンプリングすると、σが小さい領域では`θ_i`の事後分布の幅も比例して狭くなる。この「入口が広く出口が狭い漏斗」のような曲率の急激な変化に、固定的なステップサイズを持つHMC/NUTSは対応できず、[divergence](mcmc-diagnostics.md#divergence発散)を起こす。
- **使い分け**: 非中心化パラメータ化(`θ_i = μ + σ * offset_i`、`offset_i ~ Normal(0,1)`)に書き換えると、σと`θ_i`の依存関係がモデルの外(deterministic変換)に押し出され、サンプラーは常に滑らかな標準正規空間だけを探索すればよくなる。`target_accept`引き上げや事前分布への下限追加は対症療法であり、根本対処は非中心化への書き換え([techniques/reparameterization.md](../techniques/reparameterization.md#非中心化パラメータ化non-centered-parameterizationでfunnel問題を回避する)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#nile川-ベイズ変化点分析)(Neal's funnel発見→`target_accept`引き上げ/事前分布下限で対処) / [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#実装上の注意点)(階層ベイズ版TSの`offset_raw`非中心化)

---

### Ridge型非識別性

- **定義**: 2つ以上のパラメータが、互いに打ち消し合う/補い合う形で尤度にほぼ同じ影響を与えるため、事後分布のペアプロットに(円形の等高線ではなく)斜めに伸びた尾根(ridge)状の強い相関構造が現れる非識別性。
- **数式・仕組み**: 例えばHawkes過程の`κ`(興奮強度)と`β`(減衰速度)は、指数減衰カーネルを`dt→0`近傍でテイラー展開するとどちらも同じ1次の項として現れ、数式レベルで似た役割を持つ。このため「`κ`を上げて`β`も上げる」方向には尤度がほとんど変化せず、その方向に沿って事後分布が細長く伸びる。
- **使い分け**: 意味のある比(`M=κ/β`)そのものを独立パラメータとして再パラメータ化すると、ridge構造の"沿う方向"を1つのパラメータに集約でき、冗長な方向をサンプラーが探索する必要がなくなる([techniques/reparameterization.md](../techniques/reparameterization.md#比が意味を持つ量は比自体への再パラメータ化でridge型の非識別性を緩和する)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#能登半島地震-自己励起点過程hawkesetas)

---

### ラベルスイッチング(Label Switching)

- **定義**: モデルの構造が対称(複数の潜在成分を入れ替えても尤度が変わらない)である場合に、MCMCの探索中にチェーンが成分の"名前"を入れ替えてしまい、事後平均を取ると成分同士が混ざり合って潰れてしまう現象。
- **数式・仕組み**: 例えば2レジームのMarkov-Switching Modelでは、「レジーム0=平時、レジーム1=危機」という割り当てと「レジーム0=危機、レジーム1=平時」という割り当ては尤度上区別できない(対称)。どちらの割り当てを取るかがチェーンごと・時点ごとに揺れると、素朴に事後平均を取ったときに2つのレジームのボラティリティが同じ値に収束し区別不能になる。divergence=0・r_hat≈1.00という「健全」な診断結果でも発生しうるため、診断指標だけでは検出できない。
- **使い分け**: 事後的に成分を区別する場合は`pt.sort()`などで順序制約(例: `σ_0 < σ_1`)を課す。あらかじめ非対称性を仮定できる場合は、成分ごとに事前分布のレンジを分離する(例: `φ^fast ~ Beta(2,5)`、`φ^slow ~ Beta(20,1.5)`)([techniques/reparameterization.md](../techniques/reparameterization.md#ラベルスイッチングは順序制約または事前分布のレンジ分離で解消する)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-markov-switching-model) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-stochastic-volatility)

---

### マルチモダリティ(多峰性)

- **定義**: 事後分布が複数の分離した"山"(峰)を持ち、MCMCのチェーンがそれぞれ別の峰に落ちたまま行き来できなくなる病理。周期性・位相を持つパラメータ(三角関数の極形式など)で特に起こりやすい。
- **数式・仕組み**: 周期パラメータの候補値がわずかに異なると、長い時系列全体にわたって位相が大きくズレていくため、尤度面に複数の「谷」(局所解)ができやすい。複数チェーンがそれぞれ異なる谷に収束すると、chain間で[r_hat](mcmc-diagnostics.md#r_hatgelman-rubin統計量)が大きく悪化する(例: r_hat=2.10)一方、各チェーン内の局所的な探索自体は健全なため[divergence](mcmc-diagnostics.md#divergence発散)=0のままになる。
- **使い分け**: Divergence=0という結果だけで健全性を判断せず、必ずr_hatも確認する。r_hatが高い場合、chainごとの推定値の平均を比較し、近い値に集まっていれば「チェーン長不足」、明確に異なる値に分かれていれば「真の多峰性」を疑う([techniques/diagnostics.md](../techniques/diagnostics.md#chain別の平均値を比較して真の多峰性かチェーン長不足かを切り分ける)参照)。根本対処としては、三角関数の極形式を線形係数(直交形式)へ再パラメータ化するなど、尤度面を滑らかにする書き換えが有効([techniques/reparameterization.md](../techniques/reparameterization.md#三角関数の中にあるパラメータ周波数位相は直交形式へ再パラメータ化する)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#sunspot-周期性を持つ非線形状態空間モデル)
