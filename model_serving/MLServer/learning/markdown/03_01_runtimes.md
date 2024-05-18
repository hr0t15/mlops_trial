<!--
# Inference Runtimes

Inference runtimes allow you to define how your model should be used within
MLServer.
You can think of them as the **backend glue** between MLServer and your machine
learning framework of choice.

![](../assets/architecture.svg)

Out of the box, MLServer comes with a set of pre-packaged runtimes which let
you interact with a subset of common ML frameworks.
This allows you to start serving models saved in these frameworks straight
away.
To avoid bringing in dependencies for frameworks that you don't need to use,
these runtimes are implemented as independent (and optional) Python packages.
This mechanism also allows you to **rollout your [own custom runtimes](./custom.md)
very easily**.

To pick which runtime you want to use for your model, you just need to make
sure that the right package is installed, and then point to the correct runtime
class in your `model-settings.json` file.
-->

# 推論ランタイム

推論ランタイムを使用すると、MLServer内でモデルをどのように使用するかを定義できます。
これらは、MLServerと選択した機械学習フレームワークの間の**バックエンドの接着剤**と考えることができます。

![](../assets/architecture.svg)

MLServerには、一連の事前パッケージ化されたランタイムが付属しており、一部の一般的なMLフレームワークで保存されたモデルを直ちに使用できます。
これにより、これらのフレームワークで保存されたモデルのサービスをすぐに開始できます。
必要のないフレームワークの依存関係を持ち込まないために、これらのランタイムは独立した（かつオプションの）Pythonパッケージとして実装されています。
このメカニズムにより、**非常に簡単に[独自のカスタムランタイムを展開](./custom.md)することも可能**です。

モデルに使用するランタイムを選択するには、適切なパッケージがインストールされていることを確認し、その後、`model-settings.json`ファイルで正しいランタイムクラスを指定するだけです。

<!--
## Included Inference Runtimes

| Framework    | Package Name            | Implementation Class                       | Example                                                    | Documentation                                                    |
| ------------ | ----------------------- | ------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| Scikit-Learn | `mlserver-sklearn`      | `mlserver_sklearn.SKLearnModel`            | [Scikit-Learn example](../examples/sklearn/README.md)      | [MLServer SKLearn](./sklearn)                                    |
| XGBoost      | `mlserver-xgboost`      | `mlserver_xgboost.XGBoostModel`            | [XGBoost example](../examples/xgboost/README.md)           | [MLServer XGBoost](./xgboost)                                    |
| Spark MLlib  | `mlserver-mllib`        | `mlserver_mllib.MLlibModel`                | Coming Soon                                                | [MLServer MLlib](./mllib)                                        |
| LightGBM     | `mlserver-lightgbm`     | `mlserver_lightgbm.LightGBMModel`          | [LightGBM example](../examples/lightgbm/README.md)         | [MLServer LightGBM](./lightgbm)                                  |
| CatBoost     | `mlserver-catboost`     | `mlserver_catboost.CatboostModel`          | [CatBoost example](../examples/catboost/README.md)         | [MLServer CatBoost](./catboost)                                  |
| Tempo        | `tempo`                 | `tempo.mlserver.InferenceRuntime`          | [Tempo example](../examples/tempo/README.md)               | [`github.com/SeldonIO/tempo`](https://github.com/SeldonIO/tempo) |
| MLflow       | `mlserver-mlflow`       | `mlserver_mlflow.MLflowRuntime`            | [MLflow example](../examples/mlflow/README.md)             | [MLServer MLflow](./mlflow)                                      |
| Alibi-Detect | `mlserver-alibi-detect` | `mlserver_alibi_detect.AlibiDetectRuntime` | [Alibi-detect example](../examples/alibi-detect/README.md) | [MLServer Alibi-Detect](./alibi-detect)                          |
-->


## 含まれる推論ランタイム

| フレームワーク  | パッケージ名                 | 実装クラス                             | 例                                                    | ドキュメント                                               |
| -------------- | ------------------------- | ------------------------------------ | ---------------------------------------------------- | -------------------------------------------------------- |
| Scikit-Learn   | `mlserver-sklearn`        | `mlserver_sklearn.SKLearnModel`      | [Scikit-Learnの例](../examples/sklearn/README.md)      | [MLServer SKLearn](./sklearn)                             |
| XGBoost        | `mlserver-xgboost`        | `mlserver_xgboost.XGBoostModel`      | [XGBoostの例](../examples/xgboost/README.md)           | [MLServer XGBoost](./xgboost)                             |
| Spark MLlib    | `mlserver-mllib`          | `mlserver_mllib.MLlibModel`          | 近日公開予定                                           | [MLServer MLlib](./mllib)                                  |
| LightGBM       | `mlserver-lightgbm`       | `mlserver_lightgbm.LightGBMModel`    | [LightGBMの例](../examples/lightgbm/README.md)        | [MLServer LightGBM](./lightgbm)                            |
| CatBoost       | `mlserver-catboost`       | `mlserver_catboost.CatboostModel`    | [CatBoostの例](../examples/catboost/README.md)        | [MLServer CatBoost](./catboost)                            |
| Tempo          | `tempo`                   | `tempo.mlserver.InferenceRuntime`    | [Tempoの例](../examples/tempo/README.md)              | [`github.com/SeldonIO/tempo`](https://github.com/SeldonIO/tempo) |
| MLflow         | `mlserver-mlflow`         | `mlserver_mlflow.MLflowRuntime`      | [MLflowの例](../examples/mlflow/README.md)            | [MLServer MLflow](./mlflow)                                 |
| Alibi-Detect   | `mlserver-alibi-detect`   | `mlserver_alibi_detect.AlibiDetectRuntime` | [Alibi-detectの例](../examples/alibi-detect/README.md) | [MLServer Alibi-Detect](./alibi-detect)                     |

```{toctree}
:hidden:
:titlesonly:

SKLearn <./sklearn>
XGBoost <./xgboost>
MLflow <./mlflow>
Tempo <https://tempo.readthedocs.io>
Spark MLlib <./mllib>
LightGBM <./lightgbm>
Catboost <./catboost>
Alibi-Detect <./alibi-detect>
Alibi-Explain <./alibi-explain>
HuggingFace <./huggingface>
Custom <./custom>
```
