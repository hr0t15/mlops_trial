<!--
# Scikit-Learn runtime for MLServer

This package provides a MLServer runtime compatible with Scikit-Learn.

## Usage

You can install the runtime, alongside `mlserver`, as:
-->

# Scikit-Learnランタイム for MLServer

このパッケージは、Scikit-Learnと互換性のあるMLServerランタイムを提供します。

## 使用法

ランタイムを`mlserver`と一緒にインストールするには、以下のようにします：

```bash
pip install mlserver mlserver-sklearn
```

<!--
For further information on how to use MLServer with Scikit-Learn, you can check
out this [worked out example](../../docs/examples/sklearn/README.md).
-->

Scikit-Learnを使用したMLServerの使い方についての詳細は、[この具体的な例](../../docs/examples/sklearn/README.md)を参照してください。

<!--
## Content Types

If no [content type](../../docs/user-guide/content-type) is present on the
request or metadata, the Scikit-Learn runtime will try to decode the payload as
a [NumPy Array](../../docs/user-guide/content-type).
To avoid this, either send a different content type explicitly, or define the
correct one as part of your [model's
metadata](../../docs/reference/model-settings).
-->

## コンテントタイプ

リクエストやメタデータに[コンテントタイプ](../../docs/user-guide/content-type)が指定されていない場合、Scikit-Learnランタイムはペイロードを[NumPy Array](../../docs/user-guide/content-type)としてデコードしようとします。
これを避けるためには、別のコンテントタイプを明示的に送信するか、または[モデルのメタデータ](../../docs/reference/model-settings)の一部として正しいものを定義してください。

<!--
## Model Outputs

The Scikit-Learn inference runtime exposes a number of outputs depending on the
model type.
These outputs match to the `predict`, `predict_proba` and `transform` methods
of the Scikit-Learn model.

| Output          | Returned By Default | Availability                                                                                                         |
| --------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `predict`       | ✅                  | Available on most models, but not in [Scikit-Learn pipelines](https://scikit-learn.org/stable/modules/compose.html). |
| `predict_proba` | ❌                  | Only available on non-regressor models.                                                                              |
| `transform`     | ❌                  | Only available on [Scikit-Learn pipelines](https://scikit-learn.org/stable/modules/compose.html).                     |

By default, the runtime will only return the output of `predict`.
However, you are able to control which outputs you want back through the
`outputs` field of your {class}`InferenceRequest
<mlserver.types.InferenceRequest>` payload.

For example, to only return the model's `predict_proba` output, you could
define a payload such as:
-->

## モデルのアウトプット

Scikit-Learnの推論ランタイムは、モデルタイプに応じていくつかのアウトプットを公開します。
これらのアウトプットは、Scikit-Learnモデルの`predict`、`predict_proba`、`transform`メソッドに対応しています。

| アウトプット          | デフォルトで返される | 利用可能性                                                                                                    |
| --------------- | ------------------- | --------------------------------------------------------------------------------------------------------- |
| `predict`       | ✅                  | ほとんどのモデルで利用可能ですが、[Scikit-Learnパイプライン](https://scikit-learn.org/stable/modules/compose.html)では利用できません。 |
| `predict_proba` | ❌                  | 非リグレッサーモデルでのみ利用可能です。                                                                              |
| `transform`     | ❌                  | [Scikit-Learnパイプライン](https://scikit-learn.org/stable/modules/compose.html)でのみ利用可能です。                      |

デフォルトでは、ランタイムは`predict`のアウトプットのみを返します。
ただし、`outputs`フィールドを通じて、どのアウトプットを返すかを制御することができます。
このフィールドは、あなたの{class}`InferenceRequest <mlserver.types.InferenceRequest>`ペイロード内にあります。

たとえば、モデルの`predict_proba`アウトプットのみを返すように設定する場合、以下のようなペイロードを定義できます：

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
