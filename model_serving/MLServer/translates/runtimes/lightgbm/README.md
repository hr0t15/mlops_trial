<!--
# LightGBM runtime for MLServer

This package provides a MLServer runtime compatible with LightGBM.

## Usage

You can install the runtime, alongside `mlserver`, as:
-->
# LightGBMランタイム for MLServer

このパッケージは、LightGBMと互換性のあるMLServerランタイムを提供します。

## 使用法

ランタイムを`mlserver`と一緒にインストールするには、以下のようにします：


```bash
pip install mlserver mlserver-lightgbm
```

<!--
For further information on how to use MLServer with LightGBM, you can check out
this [worked out example](../../docs/examples/lightgbm/README.md).

## Content Types

If no [content type](../../docs/user-guide/content-type) is present on the
request or metadata, the LightGBM runtime will try to decode the payload as
a [NumPy Array](../../docs/user-guide/content-type).
To avoid this, either send a different content type explicitly, or define the
correct one as part of your [model's
metadata](../../docs/reference/model-settings).
-->
LightGBMを使用したMLServerの使い方についての詳細は、[この具体的な例](../../docs/examples/lightgbm/README.md)を参照してください。

## コンテントタイプ

リクエストやメタデータに[コンテントタイプ](../../docs/user-guide/content-type)が指定されていない場合、LightGBMランタイムはペイロードを[NumPy Array](../../docs/user-guide/content-type)としてデコードしようとします。
これを避けるためには、別のコンテントタイプを明示的に送信するか、または[モデルのメタデータ](../../docs/reference/model-settings)の一部として正しいものを定義してください。
