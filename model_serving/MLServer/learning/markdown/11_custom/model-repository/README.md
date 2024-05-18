<!--
# Model Repository API

MLServer supports loading and unloading models dynamically from a models repository.
This allows you to enable and disable the models accessible by MLServer on demand.
This extension builds on top of the support for [Multi-Model Serving](../mms/README.md), letting you change at runtime which models is MLServer currently serving.

The API to manage the model repository is modelled after [Triton's Model Repository extension](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md) to the V2 Dataplane and is thus fully compatible with it.

This notebook will walk you through an example using the Model Repository API.
-->


# モデルリポジトリ API

MLServer は、モデルリポジトリから動的にモデルをロードおよびアンロードすることをサポートしています。
これにより、要求に応じて MLServer がアクセス可能なモデルを有効または無効にすることができます。
この拡張機能は [マルチモデルサービング](../mms/README.md) のサポートを基に構築され、実行時に MLServer が現在提供しているモデルを変更することができます。

モデルリポジトリを管理するための API は、V2 Dataplane に対する [Triton のモデルリポジトリ拡張](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md) をモデル化し、それと完全に互換性があります。

このノートブックでは、モデルリポジトリ API を使用した例を紹介します。

<!--
## Training

First of all, we will need to train some models.
For that, we will re-use the models we trained previously in the [Multi-Model Serving example](../mms/README.md).
You can check the details on how they are trained following that notebook.
-->

## トレーニング

まず、いくつかのモデルをトレーニングする必要があります。
そのために、[マルチモデルサービングの例](../mms/README.md) で以前にトレーニングしたモデルを再利用します。
それらがどのようにトレーニングされるかの詳細は、そのノートブックを参照してください。


```python
!cp -r ../mms/models/* ./models
```

<!--
## Serving

Next up, we will start our `mlserver` inference server.
Note that, by default, this will **load all our models**.
-->

## サービング

次に、`mlserver` 推論サーバーを起動します。
デフォルトでは、これにより**すべてのモデルがロードされる**ことに注意してください。

```shell
mlserver start .
```

<!--
## List available models

Now that we've got our inference server up and running, and serving 2 different models, we can start using the Model Repository API.
To get us started, we will first list all available models in the repository.
-->

## 利用可能なモデルのリスト

推論サーバーが起動し、2つの異なるモデルを提供しているので、モデルリポジトリ API の使用を開始できます。
まず、リポジトリ内のすべての利用可能なモデルをリストアップします。

```python
import requests

response = requests.post("http://localhost:8080/v2/repository/index", json={})
response.json()
```

<!--
As we can, the repository lists 2 models (i.e. `mushroom-xgboost` and `mnist-svm`).
Note that the state for both is set to `READY`.
This means that both models are loaded, and thus ready for inference.

## Unloading our `mushroom-xgboost` model

We will now try to unload one of the 2 models, `mushroom-xgboost`.
This will unload the model from the inference server but will keep it available on our model repository.
-->



```python
requests.post("http://localhost:8080/v2/repository/models/mushroom-xgboost/unload")
```

<!--
If we now try to list the models available in our repository, we will see that the `mushroom-xgboost` model is flagged as `UNAVAILABLE`.
This means that it's present in the repository but it's not loaded for inference.
-->

リポジトリ内で利用可能なモデルをリストアップしようとすると、`mushroom-xgboost` モデルが `UNAVAILABLE` としてマークされていることがわかります。
これは、リポジトリ内に存在するものの、推論のためにロードされていないことを意味します。


```python
response = requests.post("http://localhost:8080/v2/repository/index", json={})
response.json()
```

<!--
## Loading our `mushroom-xgboost` model back

We will now load our model back into our inference server.
-->

```python
requests.post("http://localhost:8080/v2/repository/models/mushroom-xgboost/load")
```

<!--
If we now try to list the models again, we will see that our `mushroom-xgboost` is back again, ready for inference.
-->

```python
response = requests.post("http://localhost:8080/v2/repository/index", json={})
response.json()
```


```python

```
