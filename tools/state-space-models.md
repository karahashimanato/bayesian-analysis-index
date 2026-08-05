# 状態空間モデルの型

観測にどの分布を割り当てるか(`tools/observation-models.md`)とは別に、時間とともに変化する潜在状態そのものをどう構造化するかの用語辞典。定義・仕組み・使い分けを引くためのリファレンス。

---

### 変化点モデル(Changepoint Model)

- **定義**: 時系列のある未知の時点(変化点)の前後で、モデルのパラメータ(平均水準など)が切り替わると仮定する状態空間モデル。
- **数式・仕組み**: 変化点位置`τ`を離散一様分布(`DiscreteUniform`)からサンプリングし、`switch`関数で`t<τ`と`t≥τ`の2つのパラメータ値を切り替える。離散変数`τ`はPyMCのCompound Stepの対象になりESSが低下しやすいため、シグモイド関数で滑らかに遷移させる連続緩和がよく使われる。
- **使い分け**: 「ある時点を境に構造が変わった」という明確な仮説がある時系列(ダム建設前後の河川流量など)に使う。連続緩和した場合、遷移の急峻さを表す新パラメータが加わり、それが[funnel](posterior-pathologies.md#funnel漏斗状の病理neals-funnel)等の新しい病理を生まないか別途確認が必要になる。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#nile川-ベイズ変化点分析)

---

### GaussianRandomWalk(時変パラメータの状態空間表現)

- **定義**: パラメータが時間とともに滑らかに、しかし予測不能に変化していくことを表現する、正規分布のステップを積み重ねた離散時間の状態空間モデル。
- **数式・仕組み**: `x_t = x_{t-1} + ε_t`(`ε_t ~ Normal(0, σ)`)。各時点の値が直前の時点からの正規分布のランダムな増分で決まる。PyMCの`GaussianRandomWalk`はこの構造を1つの分布として提供する。
- **使い分け**: 周期モデルの振幅など、固定値では捉えきれない緩やかな時間変化を表現したい場合に使う。`shape`引数を使うとステップ数がint8にキャストされ`n>127`でオーバーフローする実装上の癖があるため、`steps`/`init_dist`を明示的に指定する([techniques/implementation-hacks.md](../techniques/implementation-hacks.md#gaussianrandomwalkのshape引数によるint8オーバーフローをstepsinit_distで回避する)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#sunspot-周期性を持つ非線形状態空間モデル)

---

### 非線形状態空間モデル(process noise付き)

- **定義**: 状態遷移が線形でない(生物学的成長モデルなど)関数で記述され、各時点の遷移に観測されないランダムな変動(process noise)が加わる状態空間モデル。
- **数式・仕組み**: 例えばロジスティック成長: `x_t = x_{t-1} + r(1 - exp(x_{t-1})/K) + process_noise`。process noiseの大きさ`sigma_process`と観測誤差の大きさ`sigma_obs`は、どちらも「モデルの予測と実データのズレ」を説明する点で役割が似ており、非識別性を起こしやすい。
- **使い分け**: 決定論的な構造パラメータ(成長率`r`や環境収容力`K`など)だけでは実データの変動を説明しきれない場合に、process noiseを追加して柔軟性を持たせる。ただしPPC(事後予測チェック)が良好でも、process noiseを取り除いたforward simulationで実データの特徴(周期振動など)を再現できるか別途確認しないと、「モデルの機構による説明」なのか「process noiseによる帳尻合わせ」なのかを見誤る([techniques/model-evaluation.md](../techniques/model-evaluation.md#ppcが良好でもモデルの機構が現象を説明しているとは限らない)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#lynx-非線形状態空間モデル)

---

### Markov-Switching Model(レジームスイッチング状態空間)

- **定義**: 観測データの生成過程が離散的な「レジーム」(平時/危機など)によって切り替わり、レジーム自体も時間とともに(マルコフ連鎖に従って)遷移すると仮定する状態空間モデル。
- **数式・仕組み**: 各時点のレジーム`S_t`はマルコフ連鎖(`P(S_t | S_{t-1})`の遷移確率行列)に従って遷移し、観測はレジームごとに異なるパラメータ(平均・分散など)を持つ分布から生成される。離散潜在状態`S_t`自体はforward algorithmで周辺化して推定することが多い([tools/observation-models.md](observation-models.md#forward-algorithm離散潜在状態の周辺化尤度)参照)。
- **使い分け**: 「平時/危機」のように、データの生成過程そのものが質的に異なる複数の状態を行き来すると考えられる時系列に使う。対称な構造を持つレジーム同士はラベルスイッチングを起こしやすいため、順序制約や事前分布のレンジ分離が必要になることが多い([tools/posterior-pathologies.md](posterior-pathologies.md#ラベルスイッチングlabel-switching)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-markov-switching-model)

---

### Stochastic Volatility(SV)モデル

- **定義**: 観測値(金融資産のリターンなど)の分散(ボラティリティ)自体を、直接観測されない連続潜在状態として時系列的にモデル化する状態空間モデル。
- **数式・仕組み**: 対数ボラティリティ`h_t`をAR(1)過程`h_t = φ・h_{t-1} + η_t`(`η_t ~ Normal(0, σ_η)`)としてモデル化し、観測値`r_t ~ Normal(0, exp(h_t/2))`(または[Student-t](observation-models.md#normal--student-t観測分布))とする。`φ`はボラティリティの持続性(persistence)を表す。
- **使い分け**: リターンの分散が時間とともにクラスタリングする(ボラティリティの高い時期・低い時期がまとまって現れる)金融時系列に使う。1つのAR(1)過程では捉えきれない挙動(ACFのギャップなど)が残る場合、fast/slowの2成分に分ける拡張(2-factor SV)を検討する。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-stochastic-volatility)
