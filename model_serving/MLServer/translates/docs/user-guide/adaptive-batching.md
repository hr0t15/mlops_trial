<!--
# Adaptive Batching

MLServer includes support to batch requests together transparently on-the-fly.
We refer to this as "adaptive batching", although it can also be known as
"predictive batching".

![](../assets/adaptive-batching.svg)
-->

# アダプティブバッチ処理

MLServerには、透明にリクエストをオンザフライでバッチ処理する機能が含まれています。
これを「アダプティブバッチ処理」と呼びますが、「予測バッチ処理」としても知られています。

![](../assets/adaptive-batching.svg)


<!--
## Benefits

There are usually two main reasons to adopt adaptive batching:

- **Maximise resource usage**.
  Usually, inference operations are “vectorised” (i.e. are designed to operate
  across batches).
  For example, a GPU is designed to operate on multiple data points at the same
  time.
  Therefore, to make sure that it’s used at maximum capacity, we need to run
  inference across batches.

- **Minimise any inference overhead**.
  Usually, all models will have to _“pay”_ a constant overhead when running any
  type of inference.
  This can be something like IO to communicate with the GPU or some kind of
  processing in the incoming data.
  Up to a certain size, this overhead tends to not scale linearly with the
  number of data points.
  Therefore, it’s in our interest to send as large batches as we can without
  deteriorating performance.

However, these benefits will usually scale only up to a certain point, which is
usually determined by either the infrastructure, the machine learning
framework used to train your model, or a combination of both.
Therefore, to maximise the performance improvements brought in by adaptive
batching it will be important to [configure it with the appropriate values for
your model](#usage).
Since these values are usually found through experimentation, **MLServer won't
enable by default adaptive batching on newly loaded models**.
-->
## メリット

アダプティブバッチ処理を採用する主な理由は通常2つあります：

- **リソース使用の最大化**。
  通常、推論操作は「ベクトル化」されています（つまり、バッチで操作するように設計されています）。
  例えば、GPUは同時に複数のデータポイントで操作するように設計されています。
  そのため、最大限の能力を発揮するためには、バッチで推論を実行する必要があります。

- **推論オーバーヘッドの最小化**。
  通常、すべてのモデルは任意の推論を実行する際に一定のオーバーヘッドを「支払う」必要があります。
  これは、GPUと通信するためのIOや、受信データの処理などが含まれることがあります。
  あるサイズまでは、このオーバーヘッドはデータポイントの数と線形にスケールしない傾向にあります。
  そのため、パフォーマンスを低下させることなく、できるだけ大きなバッチを送信することが私たちの利益になります。

しかし、これらの利点は通常、インフラストラクチャやモデルを訓練するために使用された機械学習フレームワーク、またはその両方によって決定される特定のポイントまでのみスケールします。
したがって、アダプティブバッチングによるパフォーマンス向上を最大化するためには、[適切な値で設定すること](#usage)が重要になります。
これらの値は通常、実験を通じて見つかるため、**MLServerは新しくロードされたモデルでデフォルトでアダプティブバッチ処理を有効にしません**。

<!--
## Usage

MLServer lets you configure adaptive batching independently for each model
through two main parameters:

- **Maximum batch size**, that is how many requests you want to group together.
- **Maximum batch time**, that is how much time we should wait for new
  requests until we reach our maximum batch size.
-->

## 使用法

MLServerでは、以下の2つの主要なパラメータを通じて、各モデルごとにアダプティブバッチ処理を個別に設定できます：

- **最大バッチサイズ**：これは、いくつのリクエストを一緒にグループ化したいかを示します。
- **最大バッチ時間**：これは、最大バッチサイズに達するまで新しいリクエストを待つべき時間を示します。


<!--
### `max_batch_size`

The `max_batch_size` field of the `model-settings.json` file (or
alternatively, the `MLSERVER_MODEL_MAX_BATCH_SIZE` global environment
variable) controls the maximum number of requests that should be grouped
together on each batch.
The expected values are:

- `N`, where `N > 1`, will create batches of up to `N` elements.
- `0` or `1`, will disable adaptive batching.
-->

### `max_batch_size`

`model-settings.json`ファイルの`max_batch_size`フィールド（または、代わりに`MLSERVER_MODEL_MAX_BATCH_SIZE`グローバル環境変数）は、各バッチにグループ化するべきリクエストの最大数を制御します。
期待される値は：

- `N`（`N > 1`の場合）、最大`N`要素のバッチを作成します。
- `0`または`1`は、アダプティブバッチングを無効にします。

<!--
### `max_batch_time`

The `max_batch_time` field of the `model-settings.json` file (or
alternatively, the `MLSERVER_MODEL_MAX_BATCH_TIME` global environment
variable) controls the time that MLServer should wait for new requests to come
in until we reach our maximum batch size.

The expected format is in seconds, but it will take fractional values.
That is, 500ms could be expressed as `0.5`.

The expected values are:

- `T`, where `T > 0`, will wait `T` seconds at most.
- `0`, will disable adaptive batching.
-->

### `max_batch_time`

`model-settings.json`ファイルの`max_batch_time`フィールド（または、代わりに`MLSERVER_MODEL_MAX_BATCH_TIME`グローバル環境変数）は、MLServerが最大バッチサイズに達するまで新しいリクエストを待つべき時間を制御します。

期待される形式は秒ですが、小数値を取ることができます。
つまり、500ミリ秒は`0.5`と表現できます。

期待される値は：

- `T`（`T > 0`の場合）、最大で`T`秒待ちます。
- `0`は、アダプティブバッチングを無効にします。


<!--
### Merge and split of custom paramters

MLserver allows adding custom parameters to the `parameters` field of the requests.
These parameters are recived as a merged list of parameters inside the server, e.g.
-->

### カスタムパラメーターのマージと分割

MLServerはリクエストの`parameters`フィールドにカスタムパラメーターを追加することを許可しています。
これらのパラメーターは、サーバー内でマージされたパラメーターリストとして受け取られます。例えば、以下のようになります。

```python
# request 1
types.RequestInput(
    name="parameters-np",
    shape=[1],
    datatype="BYTES",
    data=[],
    parameters=types.Parameters(
        custom-param='value-1',
    )
)

# request 2
types.RequestInput(
    name="parameters-np",
    shape=[1],
    datatype="BYTES",
    data=[],
    parameters=types.Parameters(
        custom-param='value-2',
    )
)
```

<!--
is recived as follows in the batched request in the server:
-->
は、バッチ処理されたリクエストとしてサーバーで以下のように受け取られます：


```python
types.RequestInput(
    name="parameters-np",
    shape=[2],
    datatype="BYTES",
    data=[],
    parameters=types.Parameters(
        custom-param=['value-1', 'value-2'],
    )
)
```

<!--
The same way if the request is sent back from the server as a batched request
-->
同様に、リクエストがバッチ処理されたリクエストとしてサーバーから送り返される場合も以下のようになります：

```python
types.ResponseOutput(
    name="foo",
    datatype="INT32",
    shape=[3, 3],
    data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    parameters=types.Parameters(
        content_type="np",
        foo=["foo_1", "foo_2"],
        bar=["bar_1", "bar_2", "bar_3"],
    ),
)
```

<!--
it will be returned unbatched from the server as follows:
-->
それは以下のようにサーバーからアンバッチ（個別のリクエストとして）で返されます：

```python
# Request 1
types.ResponseOutput(
    name="foo",
    datatype="INT32",
    shape=[1, 3],
    data=[1, 2, 3],
    parameters=types.Parameters(
        content_type="np", foo="foo_1", bar="'bar_1"
    ),
)

# Request 2
types.ResponseOutput(
    name="foo",
    datatype="INT32",
    shape=[1, 3],
    data=[4, 5, 6],
    parameters=types.Parameters(
        content_type="np", foo="foo_2", bar="bar_2"
    ),
)

# Request 3
types.ResponseOutput(
    name="foo",
    datatype="INT32",
    shape=[1, 3],
    data=[7, 8, 9],
    parameters=types.Parameters(content_type="np", bar="bar_3"),
)
```
