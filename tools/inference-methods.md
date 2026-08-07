# 推論エンジン・サンプリング手法

事後分布をどう計算するか(MCMC/変分推論)、方策をどう評価・選択するか(バンディットアルゴリズム)の用語辞典。定義・仕組み・使い分けを引くためのリファレンス。

---

### NUTS (No-U-Turn Sampler)

- **定義**: HMC(ハミルトニアンモンテカルロ)を発展させた、軌道の長さを自動的に決定するMCMCサンプラー。PyMCで連続パラメータをサンプリングする際のデフォルト手法。
- **数式・仕組み**: パラメータ空間に仮想的な「運動量」を導入し、ハミルトン力学の軌道をシミュレートしてサンプルを生成する(勾配を使うため、ランダムウォークするMetropolis法より効率的に空間を探索できる)。軌道が"Uターン"し始めるタイミングを自動検出して停止することで、手動でのステップ数調整を不要にする。
- **使い分け**: PyMCで連続パラメータをサンプリングする際の基本選択。離散パラメータには使えず、混在するモデルでは[Compound Step](#compound-step)による組み合わせになる。
- **登場プロジェクト**: [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#実装上の注意点)(`pymc`+`numpyro`のNUTS) / [bitcoin-utxo-survival](https://github.com/karahashimanato/bitcoin-utxo-survival/blob/main/README.md#計算戦略二段構え)(ローカルPyMC/NUTS)

---

### ADVI / mean-field変分推論(SVI)

- **定義**: 事後分布を、扱いやすい形の近似分布(mean-fieldの場合、パラメータ間の相関を無視した独立正規分布の積)で近似し、両者の差(KLダイバージェンス)を最小化する最適化問題として解く、MCMCより高速な近似推論手法。
- **数式・仕組み**: MCMCのようにサンプルを逐次生成せず、勾配降下法で近似分布のパラメータ(平均・分散)を直接最適化する。mean-field近似は変数間の共分散を0とみなすため、パラメータ間の相関が強い場合に不確実性を系統的に過小評価する。
- **使い分け**: NUTSより高速だが不確実性を過小評価しうるため、採用前にNUTSと事後分布を比較し乖離の大きさを確認する(SDで約15倍の差が出たケースあり)。大規模データで個体レベルモデルをローカルでNUTSサンプリングするのが非現実的な場合の代替手段としても使う([techniques/diagnostics.md](../techniques/diagnostics.md#変分推論adviとmcmcnutsの不確実性を比較する)参照)。
- **登場プロジェクト**: [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#分析の流れnotebooks) / [bitcoin-utxo-survival](https://github.com/karahashimanato/bitcoin-utxo-survival/blob/main/README.md#計算戦略二段構え)(numpyro SVIで個体レベルモデル)

---

### Compound Step

![変化点モデル: 離散変数tau(Compound StepでMetropolis法)のESSは連続変数lambda1/lambda2(NUTS)の約6分の1(2290 vs 14865/13439、総draws数16000)](../assets/mcmc-diagnostics/discrete_ess_gap.png)

*PyMCで実際にサンプリングした結果(Poisson変化点モデル、離散パラメータtauと連続パラメータlambda1,lambda2を比較。生成スクリプト: [scripts/generate_mcmc_diagnostics_plots.py](../scripts/generate_mcmc_diagnostics_plots.py))。詳細は[tools/mcmc-diagnostics.md](mcmc-diagnostics.md#ess-effective-sample-size)を参照。*

- **定義**: PyMCが、1つのモデルの中に離散変数と連続変数が混在する場合に自動的に採用する、変数の種類ごとに異なるサンプラーを組み合わせる仕組み。
- **数式・仕組み**: 連続変数には[NUTS](#nuts-no-u-turn-sampler)、離散変数にはMetropolis法を個別に割り当て、1イテレーション内で両方のステップを順に実行する。
- **使い分け**: 離散パラメータ(変化点の位置など)を含むモデルで自動的に発動する。Metropolisはランダムウォーク的な提案のため自己相関が強く、同じサンプル数でも離散変数のESSが連続変数より1桁近く低くなりやすい。離散変数を連続変数に緩和できれば(シグモイド近似など)NUTSのみでサンプリングでき、この問題を回避できる([tools/mcmc-diagnostics.md](mcmc-diagnostics.md#ess-effective-sample-size)参照)。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#nile川-ベイズ変化点分析)(変化点`tau`)

---

### Replay法(オフライン方策評価のシミュレーション)

- **定義**: 過去にランダム方策で収集したログデータだけを使って、別の(評価したい)方策をオンラインで運用した場合の挙動を再現するオフラインシミュレーション手法。
- **数式・仕組み**: ログの各行を時系列順に1件ずつ処理し、評価方策が実際にログと同じ行動(腕)を選んだ場合のみ、そのログの報酬を採用してモデルを更新する。行動が一致しない行は「その方策が選ばなかった」ものとして破棄する。ログ収集方策が一様ランダムであれば、この手続きは評価方策を実際にオンライン運用した場合と統計的に同じ分布を持つ推定になる。
- **使い分け**: 新しい方策(階層ベイズ版Thompson Samplingなど)を実際に本番投入せずに、過去のランダム方策ログだけで性能を検証したい場合に使う。行動が一致しない行を大量に破棄するため、有効に使えるラウンド数はログ全体よりかなり少なくなる(本プロジェクトでは約1,235万行のログに対し、実際に使えたのは約1.7万ラウンド)。
- **登場プロジェクト**: [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#notebook構成)

---

### Thompson Sampling

![階層Thompson Samplingのロックイン: 真に最良の腕4(CTR18%だが試行数n=20)に対し、独立モデルは正しく次に選ばれる確率P(argmax p)=0.54で最有力と推すが、階層モデルは他の腕に事後分布を引っ張られてP=0.30まで下がり、試行数の多い平凡な腕3(CTR10%、n=500)をP=0.49で誤って最有力と推す](../assets/inference-methods/thompson_sampling_lockin.png)

*PyMCで実際に階層Beta-Binomialモデルをサンプリングし、5本腕のうち1本だけ試行数が極端に少ないスナップショットに対する「次にその腕が選ばれる確率」を、独立モデル(閉形式のBeta事後分布)と比較した結果(生成スクリプト: [scripts/generate_inference_methods_plots.py](../scripts/generate_inference_methods_plots.py))。*

- **定義**: 各腕(選択肢)の報酬確率の事後分布から1つサンプルを引き、そのサンプル値が最大の腕を選択する、ベイズ的な多腕バンディットアルゴリズム。事後分布のサンプリングそのものが、探索(不確実な腕を試す)と活用(良さそうな腕を選ぶ)のバランスを自然に取る。
- **数式・仕組み**: 独立版は各腕ごとに独立な[Beta-Binomial](observation-models.md#beta-binomial)(Beta-Bernoulli)事後分布を持つ。階層版は腕間で情報を共有する階層ベイズモデルの事後分布からサンプルする。
- **使い分け**: 階層版は腕間の情報共有によりMAEを改善できる一方、収縮バイアスや、事後分布の確信が強まりすぎて探索が止まる「ロックイン」を起こしうる。ロックインは「探索の下限を保証する」仕組み(ε-greedyミックスなど)を別途組み合わせて対策する([techniques/diagnostics.md](../techniques/diagnostics.md#オンライン方策のロックインは探索の下限保証の欠如を疑う)参照)。
- **登場プロジェクト**: [Multi-Armed-Bandit](https://github.com/karahashimanato/Multi-Armed-Bandit/blob/main/README.md#notebook構成)
