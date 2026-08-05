# 診断・収束判定

r_hat/ESS/divergenceをどう読むか、原因不明のサンプリング異常をどう切り分けるか。

---

### r_hat → ESS → divergencesの3段階診断ワークフロー

- **症状**: 診断指標を1つだけ見て「収束した」と判断すると、他の指標が悪化していても見逃す。またmean-field ADVIのようにr_hatが構造的に定義できない手法もある。
- **対処**: r_hat → ESS → divergencesの順に3段階で確認する。手法の特性(ADVI系はr_hatが使えない等)を理解した上で診断軸を選ぶ。
- **なぜ効くか**: 各指標は異なる種類の問題(全体の収束/実効サンプルサイズ/局所的な探索失敗)を検出するため、1つだけでは不十分。各指標そのものの定義は[tools/mcmc-diagnostics.md](../tools/mcmc-diagnostics.md)を参照。
- **登場プロジェクト**: [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#得られた方法論的な学び)

---

### 表面的改善と根本問題の解決を区別する

- **症状**: `target_accept` を上げてdivergencesが減ると「解決した」と判断しがちだが、ESSやr_hatが悪化しているケースがある。
- **対処**: 1つの指標の改善だけで満足せず、他の診断指標もあわせて確認し、根本原因(モデル構造・パラメータ化)が解消されたかを判断する。
- **なぜ効くか**: `target_accept` はサンプラーの挙動を変えるだけで、非識別性やモデル誤設定そのものは解消しない。指標間のトレードオフを見ないと誤診断する。`target_accept`そのものの仕組みは[tools/mcmc-diagnostics.md](../tools/mcmc-diagnostics.md#target_accept)を参照。
- **登場プロジェクト**: [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#得られた方法論的な学び) / [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#lynx-非線形状態空間モデル)(`target_accept=0.99`でdivergenceが155→34に激減した一方、Kのess_bulkが310→27・r_hatが1.00→1.12に悪化した事例)

---

### 複数の候補仮説を1つずつ実験で反証する診断プロセス

- **症状**: サンプリングや事前予測に異常(短周期スパイク、立ち上がりの再現不能など)が出たとき、原因の仮説が複数(単位誤り、数値積分の不安定性、パラメータ設定、観測開始時点の遅れ)あり、直感だけでは特定できない。
- **対処**: 候補仮説を列挙し、コード確認や実験(例: `n_steps_per_week` を10→50に増やして数値不安定性説を検証、R0やσ・γを極端な値まで動かして再現性を検証)で1つずつ反証していく消去法を取る。
- **なぜ効くか**: 複数の要因が絡む異常は、いきなり「これが原因」と決め打ちすると誤った修正に時間を溶かす。反証可能な形で仮説を1つずつ潰すことで、真因への到達が保証される。
- **登場プロジェクト**: [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#横断的な学び)

---

### 非識別性とモデルの誤設定は別種の壁である

- **症状**: サンプリングが健全(divergence=0, r_hat=1.00)なのに、モデルがデータの挙動(急峻な立ち上がりなど)を再現できないケースがある。「収束していないから悪い」という単純な判断では原因を見誤る。
- **対処**: 非識別性(再パラメータ化で緩和可能)と、モデルの生成過程とデータの実際の生成過程の食い違い(パラメータ探索では原理的に解決不能)を区別する。後者はパラメータを主要な自由度すべてで動かし尽くしても解消しないことで判別できる。
- **なぜ効くか**: 「モデルが健全に収束すること」と「モデルが正しい問いに答えていること」は独立の軸であるため、収束の健全性だけでモデルの正しさを保証できない。
- **登場プロジェクト**: [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#seir-湖北省covid-19初期流行)

---

### 変分推論(ADVI)とMCMC(NUTS)の不確実性を比較する

- **症状**: mean-field ADVIは高速だが、パラメータ間の相関を無視するため不確実性を過小評価しうる。
- **対処**: ADVIとNUTSの事後分布を並べて比較し、乖離の大きさ(SDで約15倍の差が出たケースあり)を確認してから採用手法を決める。
- **なぜ効くか**: mean-field近似は変数間の共分散を0とみなすため、相関の強いパラメータ空間では不確実性の過小評価が体系的に起こる。ADVI/NUTSそのものの定義は[tools/inference-methods.md](../tools/inference-methods.md)を参照。
- **登場プロジェクト**: [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#分析の流れnotebooks)

---

### Divergence=0でもr_hatでしか検出できないマルチモダリティがある

- **症状**: Divergences=0(局所的な探索は健全)にもかかわらず、r_hat=2.10のような深刻な収束の失敗が発生する。trace plotを見ると、複数のchainがそれぞれ全く異なる値(周期パラメータの候補値など)に固定されたまま、一切混ざっていない。
- **対処**: Divergences=0という結果だけで健全性を判断せず、必ずr_hatも確認する。マルチモダリティが疑われる場合は、chainごとの推定値やtrace plotを見て、chainが別々の"谷"(局所解)に落ちて出られなくなっていないか確認する。
- **なぜ効くか**: Divergenceは「サンプラーが局所的に破綻したか」を検出する指標であり、「複数のchainが尤度面の異なる谷に別々に収束してしまう」というグローバルな病理(マルチモダリティ)は原理的に検出できない。周期・位相を持つパラメータは特にこの問題を起こしやすい。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#sunspot-周期性を持つ非線形状態空間モデル)

---

### chain別の平均値を比較して、真の多峰性かチェーン長不足かを切り分ける

- **症状**: Divergences=0だがr_hatが高い(例: 1.06)という結果が出たとき、マルチモダリティ(真の多峰性)なのか、単にチェーン長(tune/draws)が足りず各chainがまだ収束しきっていないだけなのか、区別がつかない。
- **対処**: chainごとの推定値の平均を比較する。近い値に集まっていれば「チェーン長不足」の可能性が高く、tune/drawsを増やして再実行し改善するか確認する。明確に異なる値に分かれていれば真の多峰性を疑う。
- **なぜ効くか**: マルチモダリティ(chainが別々の谷に落ちる)とチェーン長不足(chainがまだ目標分布に到達していない)は、どちらもr_hatを悪化させるが、原因も対処法も全く異なる。chain別の平均値という一手間の確認だけで、この2つを高い精度で切り分けられる。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#日経225-stochastic-volatility)

---

### Divergent pointsの分布パターン(局所集中 vs 分散)で病理の種類を切り分ける

- **症状**: Divergenceが発生しているが、それが構造的な非識別性(funnel等)によるものか、単にステップサイズがギリギリ足りていないだけなのか、対処の前に判断がつかない。
- **対処**: divergent pointsがパラメータ空間のどこに現れているかを確認する。特定の隅・境界に局所集中していれば構造的な非識別性・funnelを示唆し、事後分布の主要な塊全体に薄く分散していれば単なるステップサイズ不足の可能性が高く、tune増加などで解消しやすい。
- **なぜ効くか**: 両者は同じ「divergence数」という指標に現れるが、原因も対処法(モデルの再パラメータ化 vs サンプラー設定の調整)も異なる。分布パターンという追加情報を見ることで、無駄な対処(構造的でない問題にモデル変更で挑む、あるいはその逆)を避けられる。divergenceそのものの定義は[tools/mcmc-diagnostics.md](../tools/mcmc-diagnostics.md#divergence発散)を参照。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#lynx-非線形状態空間モデル)

---

### 離散変数はESSが低くなりやすい

- **症状**: 離散変数(変化点の位置`tau`など)のESSだけが、他の連続変数より1桁近く低くなる。
- **対処**: 可能であれば、離散変数を連続変数に緩和する(例: `switch`関数による離散的な切り替えを、シグモイド関数による滑らかな遷移に置き換え、`tau`自体を連続変数として扱う)ことでNUTSのみでサンプリング可能にする。
- **なぜ効くか**: PyMCは離散変数に対して自動的にMetropolis法を、連続変数にはNUTSを割り当てるCompound Stepを使う。Metropolisはランダムウォーク的な提案のため自己相関が強く、同じサンプル数でも実効サンプルサイズ(ESS)が少なくなる。ただし連続緩和は新たなパラメータ(遷移の急さ等)を導入することが多く、それがfunnel等の新しい病理を生まないか別途確認が必要になる([reparameterization.md](reparameterization.md)参照)。ESSそのものの定義は[tools/mcmc-diagnostics.md](../tools/mcmc-diagnostics.md#ess-effective-sample-size)を参照。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#nile川-ベイズ変化点分析)

---

### オンライン方策のロックインは「探索の下限保証」の欠如を疑う

- **症状**: 階層ベイズ版トンプソン抽出(TS)を大規模実データで検証したところ、特定の凡庸な腕(真のCTR順位57/80)に選択が偏って自己修正しない「ロックイン」が発生し、独立版TSにも劣る結果になった。
- **対処**: 原因を「探索の下限を保証しない実装」と特定し、確率10%で強制的にランダム選択するε-greedyミックスを導入したところロックインが解消した。
- **なぜ効くか**: トンプソン抽出は理論上は事後分布のサンプリングで自然に探索と活用のバランスを取るが、収縮バイアスなど他の要因で特定の腕への確信が過度に強まると、事後サンプリングだけでは抜け出せない局所解に陥ることがある。強制探索の下限を別途設けることで、この種の失敗モードに対する保険になる。Thompson Samplingそのものの定義は[tools/inference-methods.md](../tools/inference-methods.md#thompson-sampling)を参照。
- **登場プロジェクト**: [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#主な発見)
