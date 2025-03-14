<!--
# Metrics

Out-of-the-box, MLServer exposes a set of metrics that help you monitor your
machine learning workloads in production.
These include standard metrics like number of requests and latency.

On top of these, you can also register and track your own [custom
metrics](#custom-metrics) as part of your [custom inference
runtimes](./custom).
-->

# メトリクス

MLServer は、本番環境での機械学習ワークロードの監視に役立つ一連のメトリクスを提供しています。これには、リクエスト数やレイテンシなどの標準的なメトリクスが含まれます。



<!--
## Default Metrics

By default, MLServer will expose metrics around inference requests (count and
error rate) and the status of its internal requests queues.
These internal queues are used for [adaptive batching](./adaptive-batching) and
[communication with the inference workers](./parallel-inference).

| Metric Name                   | Description                                                         |
| ----------------------------- | ------------------------------------------------------------------- |
| `model_infer_request_success` | Number of successful inference requests.                            |
| `model_infer_request_failure` | Number of failed inference requests.                                |
| `batch_request_queue`         | Queue size for the [adaptive batching](./adaptive-batching) queue.  |
| `parallel_request_queue`      | Queue size for the [inference workers](./parallel-inference) queue. |
-->

## デフォルトのメトリクス

MLServer は、推論リクエスト（成功数とエラー率）や適応的バッチングや推論ワーカーとの通信のための内部リクエストキューの状態など、推論リクエストに関連するメトリクスを公開しています。

| メトリック名                   | 説明                                                               |
| ----------------------------- | -------------------------------------------------------------------- |
| `model_infer_request_success` | 成功した推論リクエストの数                                          |
| `model_infer_request_failure` | 失敗した推論リクエストの数                                          |
| `batch_request_queue`         | 適応的バッチング用のキューのサイズ                                 |
| `parallel_request_queue`      | 推論ワーカー用のキューのサイズ                                     |

<!--
### REST Server Metrics

On top of the default set of metrics, MLServer's REST server will also expose a
set of metrics specific to REST.

```{note}
The prefix for the REST-specific metrics will be dependent on the
`metrics_rest_server_prefix` flag from the [MLServer settings](#settings).
```

| Metric Name                               | Description                                                    |
| ----------------------------------------- | -------------------------------------------------------------- |
| `[rest_server]_requests`                  | Number of REST requests, labelled by endpoint and status code. |
| `[rest_server]_requests_duration_seconds` | Latency of REST requests.                                      |
| `[rest_server]_requests_in_progress`      | Number of in-flight REST requests.                             |
-->


### REST サーバーのメトリクス

MLServer の REST サーバーは、REST リクエストに特有の追加のメトリクスを公開します。

```note
REST に特有のメトリクスのプレフィックスは、[MLServer 設定](#settings)の `metrics_rest_server_prefix` フラグに依存します。
```

| メトリック名                               | 説明                                                    |
| ----------------------------------------- | ------------------------------------------------------ |
| `[rest_server]_requests`                  | エンドポイントとステータスコードでラベル付けされた REST リクエストの数 |
| `[rest_server]_requests_duration_seconds` | REST リクエストのレイテンシ                             |
| `[rest_server]_requests_in_progress`      | 実行中の REST リクエストの数                             |


<!--
### gRPC Server Metrics

On top of the default set of metrics, MLServer's gRPC server will also expose a
set of metrics specific to gRPC.

| Metric Name           | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `grpc_server_handled` | Number of gRPC requests, labelled by gRPC code and method. |
| `grpc_server_started` | Number of in-flight gRPC requests.                         |
-->

### gRPC サーバーメトリクス

デフォルトのメトリクスに加えて、MLServer の gRPC サーバーは gRPC に固有のメトリクスも公開します。

| メトリック名           | 説明                                                |
| --------------------- | -------------------------------------------------- |
| `grpc_server_handled` | gRPC リクエストの数。gRPC コードとメソッドでラベル付けされています。 |
| `grpc_server_started` | 実行中の gRPC リクエストの数。                         |


<!--
## Custom Metrics

MLServer allows you to register custom metrics within your custom inference
runtimes.
This can be done through the `mlserver.register()` and `mlserver.log()`
methods.

- {func}`mlserver.register`: Register a new metric.
- {func}`mlserver.log`:
  Log a new set of metric / value pairs.
  If there's any unregistered metric, it will get registered on-the-fly.

```{note}
Under the hood, metrics logged through the {func}`mlserver.log` method will get
exposed to Prometheus as a Histogram.
```

Custom metrics will generally be registered in the {func}`load()
<mlserver.MLModel.load>` method and then used in the {func}`predict()
<mlserver.MLModel.predict>` method of your [custom runtime](./custom).
-->

## カスタムメトリクス

MLServer では、カスタム推論ランタイム内でカスタムメトリクスを登録することができます。
これは、`mlserver.register()` と `mlserver.log()` メソッドを使用して行います。

- {func}`mlserver.register`: 新しいメトリックを登録します。
- {func}`mlserver.log`: メトリック / 値のペアを新しくログします。
  未登録のメトリックがある場合、動的に登録されます。

```{note}
内部的には、`mlserver.log` メソッドを介してログされたメトリックは、Prometheus にヒストグラムとして公開されます。
```

カスタムメトリックは、一般的には {func}`load() <mlserver.MLModel.load>` メソッドで登録され、その後、[カスタムランタイム](./custom)の {func}`predict() <mlserver.MLModel.predict>` メソッドで使用されます。


```{code-block} python
---
emphasize-lines: 1, 8, 12
---
import mlserver

from mlserver.types import InferenceRequest, InferenceResponse

class MyCustomRuntime(mlserver.MLModel):
  async def load(self) -> bool:
    self._model = load_my_custom_model()
    mlserver.register("my_custom_metric", "This is a custom metric example")
    return True

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    mlserver.log(my_custom_metric=34)
    # TODO: Replace for custom logic to run inference
    return self._model.predict(payload)
```

<!--
## Metrics Labelling

For metrics specific to a model (e.g. [custom metrics](#custom-metrics),
request counts, etc), MLServer will always label these with the **model name**
and **model version**.
Downstream, this will allow to aggregate and query metrics per model.

```{note}
If these labels are not present on a specific metric, this means that those
metrics can't be sliced at the model level.
```

Below, you can find the list of standardised labels that you will be able to
find on model-specific metrics:

| Label Name      | Description                         |
| --------------- | ----------------------------------- |
| `model_name`    | Model Name (e.g. `my-custom-model`) |
| `model_version` | Model Version (e.g. `v1.2.3`)       |
-->



## メトリクスのラベリング

モデル固有のメトリック（例：[カスタムメトリック](#custom-metrics)、リクエスト数など）に関するメトリックは、常に**モデル名**と**モデルバージョン**でラベル付けされます。
これにより、下流でモデルごとにメトリックを集計およびクエリできるようになります。

```{note}
特定のメトリックにこれらのラベルが存在しない場合、それらのメトリックをモデルレベルでスライスすることはできません。
```

以下、モデル固有のメトリクスで見つけることができる標準化されたラベルのリストをご覧いただけます:

| ラベル名        | 説明                                 |
| --------------- | ------------------------------------ |
| `model_name`    | モデル名（例: `my-custom-model`）    |
| `model_version` | モデルバージョン（例: `v1.2.3`）    |


<!--
## Settings

MLServer will expose metric values through a metrics endpoint exposed on its
own metric server.
This endpoint can be polled by [Prometheus](https://prometheus.io/) or other
[OpenMetrics](https://openmetrics.io/)-compatible backends.

Below you can find the [settings](../reference/settings) available to control
the behaviour of the metrics server:

| Label Name                   | Description                                                                                                                                                                                                                                                                                       | Default                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `metrics_endpoint`           | Path under which the metrics endpoint will be exposed.                                                                                                                                                                                                                                            | `/metrics`                                         |
| `metrics_port`               | Port used to serve the metrics server.                                                                                                                                                                                                                                                            | `8082`                                             |
| `metrics_rest_server_prefix` | Prefix used for metric names specific to MLServer's REST inference interface.                                                                                                                                                                                                                     | `rest_server`                                      |
| `metrics_dir`                | Directory used to store internal metric files (used to support metrics sharing across [inference workers](./parallel-inference)). This is equivalent to Prometheus' [`$PROMETHEUS_MULTIPROC_DIR`](https://github.com/prometheus/client_python/tree/master#multiprocess-mode-eg-gunicorn) env var. | MLServer's current working directory (i.e. `$PWD`) |
-->
## 設定

MLServerは、その独自のメトリックサーバーで公開されたメトリック値をメトリックエンドポイントを介して公開します。
このエンドポイントは、[Prometheus](https://prometheus.io/)や他の[OpenMetrics](https://openmetrics.io/)互換のバックエンドによってポーリングされる可能性があります。

以下では、メトリックサーバーの動作を制御するために利用可能な[設定](../reference/settings)を見ることができます:

| ラベル名                      | 説明                                                                                                                                                                                                                                                                                           | デフォルト                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `metrics_endpoint`           | メトリックエンドポイントが公開されるパス。                                                                                                                                                                                                                                               | `/metrics`                                         |
| `metrics_port`               | メトリックサーバーを提供するために使用されるポート。                                                                                                                                                                                                                                       | `8082`                                             |
| `metrics_rest_server_prefix` | MLServerのREST推論インターフェースに固有のメトリック名に使用される接頭辞。                                                                                                                                                                                                                  | `rest_server`                                      |
| `metrics_dir`                | 内部メトリックファイルを保存するディレクトリ（[推論ワーカー](./parallel-inference)間でメトリックを共有するために使用されます）。これはPrometheusの[`$PROMETHEUS_MULTIPROC_DIR`](https://github.com/prometheus/client_python/tree/master#multiprocess-mode-eg-gunicorn)環境変数に相当します。 | MLServerの現在の作業ディレクトリ（つまり `$PWD`） |
