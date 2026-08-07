# 事前分布設計・prior predictive check

事前分布のスケールをどう決めるか、サンプリング前にどう検証するか。

---

### 定常状態の解析式に実データを代入して事前分布の中心値を決める

- **症状**: 事前分布のスケールを勘や一般的な相場観だけで決めると、prior predictiveが非現実的な範囲に暴走する。
- **対処**: モデルの解析的な定常状態(平衡点・ピーク高さなど)の式に、EDAで観測した増減速度・平衡水準・季節振幅を代入して、事前分布の中心値を逆算する。SIR/SIS/SIRS/SEIRの4モデル共通で採用した手順。
- **なぜ効くか**: パラメータ空間で「データと矛盾しない範囲」を先に見積もってから事前分布を置くため、prior predictive checkで何度も往復する回数が減る。
- **登場プロジェクト**: [bayesian-epidemiological-models](https://github.com/karahashimanato/bayesian-epidemiological-models/blob/main/README.md#横断的な学び)

---

### 事前予測チェックはサンプリング前に必ず実施する

- **症状**: パラメータのスケール設定ミス(予測確率が0/1に張り付く)や、Jensen不等式に起因する非直感的な挙動(他パラメータの分散が加わるだけで予測平均がズレる)は、サンプリング後の診断だけでは気づきにくい。
- **対処**: モデル構築の各段階でサンプリング前にprior predictive checkを行い、予測分布の形・範囲を目視確認する。
- **なぜ効くか**: 事後分布の診断(r_hat/ESS/divergence)が「健全」でも、それはモデルが収束しただけであってモデル設計自体の誤りは検出できない。prior predictiveは設計段階のミスをサンプリング前に切り分けられる数少ない手段。Jensen不等式そのものの定義は[tools/statistical-biases.md](../tools/statistical-biases.md#jensen不等式jensens-inequality)を参照。
- **登場プロジェクト**: [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#得られた方法論的な学び)

---

### 「分母のパラメータが0に近づくと分散が発散する」病理は分布族を問わず繰り返す

![分母のパラメータが0に近づくと分散が発散する病理: Exponential(1)事前分布(0で密度が最大)由来のprior predictiveは最大115万まで暴走するが、Gamma(shape=2)事前分布(0で密度がゼロ)なら最大1.7万に収まる](../assets/prior-predictive/denominator_variance_explosion.png)

*prior predictiveを実際にサンプリングした結果(μ²/αの形の分散を例に、αの事前分布による違いを比較。生成スクリプト: [scripts/generate_prior_predictive_plots.py](../scripts/generate_prior_predictive_plots.py))。*

- **症状**: prior predictiveの最大値が現実離れした値(数百〜数千倍)に暴走する。SVモデルの $\sigma_\eta^2/(1-\phi^2)$、Gamma-Poisson階層モデルの $\mu^2/\alpha_{conc}$、Hawkes過程の減衰項 $e^{-\beta\cdot dt}$( $\beta\to0$で減衰が機能しなくなる)など、形は違うが「あるパラメータが0に近づくと分散・強度が発散する」という同型の構造が、独立した3つのモデルクラスで繰り返し発生した。
- **対処**: 分散が発散しうるパラメータの事前分布を、0付近の密度がゼロになる分布族(`Exponential`ではなく`Gamma(alpha>1, ...)`など)に変更する構造的対処を優先する。平均を動かす対症療法(スケールを緩める)より、分布の形自体を変える方が極端な裾を強く抑制できる。
- **なぜ効くか**: `Exponential`は0付近に確率密度の山を持つため、分母に来るパラメータとして使うと一定確率で分散爆発を引き起こす。`Gamma(shape>1)`は0での密度がゼロになるため、この経路を構造的に塞げる。一方Dirichlet-Multinomialのように「配分先の合計が固定されている」構造では、集中度パラメータが0に近づいても総量自体は上限を超えられないため、同型の発散がそもそも起こらない。「総量が固定か青天井か」を先に見抜けると、prior predictive checkの労力を節約できる。パラメータの性質ごとの事前分布の選び方一般は[tools/prior-distributions.md](../tools/prior-distributions.md)を参照。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#横断的な学び)

---

### 対処後も、暴走の主犯が別パラメータへ移っていないか都度ペアプロットで確認する

- **症状**: 1つのパラメータの事前分布を構造的に対処(0付近の密度をゼロにする分布へ変更)しても、prior predictiveの最大値がまだ大きい(例: 5880→3639)。
- **対処**: 残った暴走の原因を、単一パラメータの事前分布をさらに締めることで済ませず、複数パラメータ間のペアプロット(散布図)で診断し直し、暴走の主犯が別のパラメータに移っていないか確認する。移っていれば、そのパラメータにも同じ構造的対処を適用する。
- **なぜ効くか**: 複数パラメータが絡んで分散・強度を決める構造(Hawkes過程の $\kappa$と $\beta$など)では、1つを対処しても残りのパラメータが同じ役割を引き継いで暴走を再現することがある。単発の修正で終わらせず、都度診断し直すことで真に構造的な解決に到達できる。
- **登場プロジェクト**: [bayesian-modeling-lab](https://github.com/karahashimanato/bayesian-modeling-lab/blob/main/README.md#能登半島地震-自己励起点過程hawkesetas)

---

### 判断基準は極値だけでなく割合・信用区間幅で見る

- **症状**: prior predictiveの最大値・最小値だけを見て「範囲内だから大丈夫」と判断すると、分布の大部分が非現実的な領域に偏っていても見逃す。
- **対処**: 極値(min/max)に加えて、現実的な範囲に入る割合や信用区間の幅で事前分布の妥当性を判断する。
- **なぜ効くか**: 極値は外れ値1つでも動くため、分布全体の健全性の指標としては弱い。
- **登場プロジェクト**: [bayesian-A-B-testing](https://github.com/karahashimanato/bayesian-A-B-testing/blob/main/README.md#得られた方法論的な学び)
