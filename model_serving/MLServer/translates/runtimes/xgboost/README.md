<!--
# XGBoost runtime for MLServer

This package provides a MLServer runtime compatible with XGBoost.
-->

# XGBoostランタイム for MLServer

このパッケージは、XGBoostと互換性のあるMLServerランタイムを提供します。

<!--
## Usage

You can install the runtime, alongside `mlserver`, as:
-->

## 使用法

ランタイムを`mlserver`と一緒にインストールするには、以下のようにします：

```bash
pip install mlserver mlserver-xgboost
```

<!--
For further information on how to use MLServer with XGBoost, you can check out
this [worked out example](../../docs/examples/xgboost/README.md).

## XGBoost Artifact Type

The XGBoost inference runtime will expect that your model is serialised via one
of the following methods:

| Extension | Docs                                                                                                                 | Example                            |
| --------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `*.json`  | [JSON Format](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html#introduction-to-model-io)         | `booster.save_model("model.json")` |
| `*.ubj`   | [Binary JSON Format](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html#introduction-to-model-io)  | `booster.save_model("model.ubj")`  |
| `*.bst`   | [(Old) Binary Format](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html#introduction-to-model-io) | `booster.save_model("model.bst")`  |
-->

XGBoostを使用したMLServerの使い方についての詳細は、[この具体的な例](../../docs/examples/xgboost/README.md)を参照してください。

## XGBoost アーティファクトタイプ

XGBoost 推論ランタイムは、モデルが以下の方法のいずれかを使用してシリアライズされていることを期待します：

| 拡張子    | ドキュメント                                                                                                          | 例                                  |
| --------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `*.json`  | [JSONフォーマット](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html#introduction-to-model-io)     | `booster.save_model("model.json")` |
| `*.ubj`   | [バイナリJSONフォーマット](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html#introduction-to-model-io) | `booster.save_model("model.ubj")`  |
| `*.bst`   | [(旧) バイナリフォーマット](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html#introduction-to-model-io) | `booster.save_model("model.bst")`  |



````{note}
<!--
By default, the runtime will look for a file called `model.[json | ubj | bst]`.
However, this can be modified through the `parameters.uri` field of your
{class}`ModelSettings <mlserver.settings.ModelSettings>` config (see the
section on [Model Settings](../../docs/reference/model-settings.md) for more
details).
-->
デフォルトでは、ランタイムは `model.[json | ubj | bst]` という名前のファイルを探します。
ただし、これは `{class}`ModelSettings <mlserver.settings.ModelSettings>` の設定の `parameters.uri` フィールドを通じて変更することができます（詳細は [モデル設定](../../docs/reference/model-settings.md) のセクションを参照してください）。



```{code-block} json
---
emphasize-lines: 3-5
---
{
  "name": "foo",
  "parameters": {
    "uri": "./my-own-model-filename.json"
  }
}
```
````

<!--
## Content Types

If no [content type](../../docs/user-guide/content-type) is present on the
request or metadata, the XGBoost runtime will try to decode the payload as a
[NumPy Array](../../docs/user-guide/content-type).
To avoid this, either send a different content type explicitly, or define the
correct one as part of your [model's
metadata](../../docs/reference/model-settings).
-->
## コンテントタイプ

リクエストやメタデータに[コンテントタイプ](../../docs/user-guide/content-type)が指定されていない場合、XGBoostランタイムはペイロードを[NumPy Array](../../docs/user-guide/content-type)としてデコードしようとします。
これを避けるためには、別のコンテントタイプを明示的に送信するか、または[モデルのメタデータ](../../docs/reference/model-settings)の一部として正しいものを定義してください。


<!--
## Model Outputs

The XGBoost inference runtime exposes a number of outputs depending on the
model type.
These outputs match to the `predict` and `predict_proba` methods of the XGBoost
model.

| Output          | Returned By Default | Availability                                                          |
| --------------- | ------------------- | --------------------------------------------------------------------- |
| `predict`       | ✅                  | Available on all XGBoost models.                                      |
| `predict_proba` | ❌                  | Only available on non-regressor models (i.e. `XGBClassifier` models). |

By default, the runtime will only return the output of `predict`.
However, you are able to control which outputs you want back through the
`outputs` field of your {class}`InferenceRequest
<mlserver.types.InferenceRequest>` payload.

For example, to only return the model's `predict_proba` output, you could
define a payload such as:
-->

## モデル出力

XGBoost 推論ランタイムは、モデルのタイプに応じていくつかの出力を提供します。
これらの出力は、XGBoost モデルの `predict` および `predict_proba` メソッドに対応しています。

| 出力             | デフォルトで返される | 利用可能性                                                             |
| --------------- | ------------------- | -------------------------------------------------------------------- |
| `predict`       | ✅                  | すべての XGBoost モデルで利用可能。                                    |
| `predict_proba` | ❌                  | 回帰以外のモデル（例: `XGBClassifier` モデル）でのみ利用可能。 |

デフォルトでは、ランタイムは `predict` の出力のみを返します。
しかし、`outputs` フィールドを通じて、どの出力を返すかを制御することができます。
このフィールドは、あなたの {class}`InferenceRequest <mlserver.types.InferenceRequest>` ペイロード内にあります。

例えば、モデルの `predict_proba` 出力のみを返すように設定する場合、以下のようなペイロードを定義できます：

```{code-block} json
---
emphasize-lines: 10-12
---
{
  "inputs": [
    {
      "name": "my-input",
      "datatype": "INT32",
      "shape": [2, 2],
      "data": [1, 2, 3, 4]
    }
  ],
  "outputs": [
    { "name": "predict_proba" }
  ]
}
```
