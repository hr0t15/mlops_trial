<!--
# Parallel Inference

Out of the box, MLServer includes support to offload inference workloads to a
pool of workers running in separate processes.
This allows MLServer to scale out beyond the limitations of the Python
interpreter.
To learn more about why this can be beneficial, you can check the [concurrency
section](#concurrency-in-python) below.

![](../assets/parallel-inference.svg)

By default, MLServer will spin up a pool with only one worker process to run
inference.
All models will be loaded uniformly across the inference pool workers.
To read more about advanced settings, please see the [usage section
below](#usage).
-->

# 並列推論

初期設定で、MLServerは推論ワークロードを別プロセスで実行されるワーカープールにオフロードするサポートを含んでいます。
これにより、MLServerはPythonインタープリタの制限を超えてスケールアウトすることができます。
この利点について詳しく知りたい場合は、以下の[並行性セクション](#concurrency-in-python)をチェックしてください。

![](../assets/parallel-inference.svg)

デフォルトでは、MLServerは推論を実行するために1つのワーカープロセスを持つプールを起動します。
すべてのモデルは推論プールのワーカー間で均等にロードされます。
詳細な設定については、以下の[使用セクション](#usage)を参照してください。

<!--
## Concurrency in Python

The [Global Interpreter Lock
(GIL)](https://wiki.python.org/moin/GlobalInterpreterLock) is a mutex lock that
exists in most Python interpreters (e.g. CPython).
Its main purpose is to lock Python’s execution so that it only runs on a single
processor at the same time.
This simplifies certain things to the interpreter.
However, it also adds the limitation that a **single Python process will never
be able to leverage multiple cores**.

When we think about MLServer's support for [Multi-Model Serving
(MMS)](../examples/mms/README.md), this could lead to scenarios where a
**heavily-used model starves the other models** running within the same
MLServer instance.
Similarly, even if we don’t take MMS into account, the **GIL also makes it harder
to scale inference for a single model**.

To work around this limitation, MLServer offloads the model inference to a pool
of workers, where each worker is a separate Python process (and thus has its
own separate GIL).
This means that we can get full access to the underlying hardware.
-->

## Pythonの並行性

[グローバルインタープリタロック（GIL）](https://wiki.python.org/moin/GlobalInterpreterLock)は、ほとんどのPythonインタープリタ（例えばCPython）に存在するミューテックスロックです。
その主な目的は、Pythonの実行をロックして、一度に1つのプロセッサ上でのみ実行されるようにすることです。
これはインタープリタにとって一部のことを単純化します。
しかし、**単一のPythonプロセスが複数のコアを活用することができない**という制限も生じます。

[マルチモデルサービング（MMS）](../examples/mms/README.md)のサポートについて考えるとき、これは**頻繁に使用されるモデルが他のモデルを飢えさせる**シナリオにつながる可能性があります。これらは同じMLServerインスタンス内で実行されます。
同様に、MMSを考慮しない場合でも、**GILは単一モデルの推論をスケールすることを難しくします**。

この制限を回避するために、MLServerはモデル推論を別のPythonプロセスであるワーカーのプールにオフロードします（そしてそれぞれが独自のGILを持っています）。
これにより、基盤となるハードウェアへのフルアクセスが可能になります。



<!--
### Overhead

Managing the Inter-Process Communication (IPC) between the main MLServer
process and the inference pool workers brings in some overhead.
Under the hood, MLServer uses the `multiprocessing` library to implement the
distributed processing management, which has been shown to offer the smallest
possible overhead when implementating these type of distributed strategies
{cite}`zhiFiberPlatformEfficient2020`.

The extra overhead introduced by other libraries is usually brought in as a
trade off in exchange of other advanced features for complex distributed
processing scenarios.
However, MLServer's use case is simple enough to not require any of these.

Despite the above, even though this overhead is minimised, this **it can still
be particularly noticeable for lightweight inference methods**, where the extra
IPC overhead can take a large percentage of the overall time.
In these cases (which can only be assessed on a model-by-model basis), the user
has the option to [disable the parallel inference feature](#usage).

For regular models where inference can take a bit more time, this overhead is
usually offset by the benefit of having multiple cores to compute inference on.
-->

### オーバーヘッド

MLServerのメインプロセスと推論プールのワーカー間のプロセス間通信（IPC）の管理にはいくらかのオーバーヘッドが発生します。
内部では、MLServerは`multiprocessing`ライブラリを使用して分散処理管理を実装しており、これはこのタイプの分散戦略を実装する際に可能な限り最小のオーバーヘッドを提供することが示されています{cite}`zhiFiberPlatformEfficient2020`。

他のライブラリによって導入される追加のオーバーヘッドは、通常、複雑な分散処理シナリオのための他の高度な機能とのトレードオフとして持ち込まれます。
しかし、MLServerのユースケースはこれらを必要としないほど十分にシンプルです。

それにもかかわらず、このオーバーヘッドは最小限に抑えられているとしても、**軽量な推論方法では特に顕著になることがあります**。ここで、追加のIPCオーバーヘッドが全体の時間の大部分を占める可能性があります。
これらのケース（モデルごとに評価される必要があります）では、ユーザーは[並列推論機能を無効にするオプション](#usage)を持っています。

通常のモデルでは推論にもう少し時間がかかるため、このオーバーヘッドは複数のコアで推論を計算する利点によって通常相殺されます。

<!--
## Usage

By default, MLServer will always create an inference pool with one single
worker.
The number of workers (i.e. the size of the inference pool) can be adjusted
globally through the server-level `parallel_workers` setting.
-->

## 使用法

デフォルトでは、MLServerは常に1つのワーカーを持つ推論プールを作成します。
ワーカーの数（つまり、推論プールのサイズ）は、サーバーレベルの`parallel_workers`設定を通じてグローバルに調整できます。


<!--
### `parallel_workers`

The `parallel_workers` field of the `settings.json` file (or alternatively, the
`MLSERVER_PARALLEL_WORKERS` global environment variable) controls the size of
MLServer's inference pool.
The expected values are:

- `N`, where `N > 0`, will create a pool of `N` workers.
- `0`, will disable the parallel inference feature.
  In other words, inference will happen within the main MLServer process.
-->

### `parallel_workers`

`settings.json`ファイルの`parallel_workers`フィールド（または代替として、`MLSERVER_PARALLEL_WORKERS`グローバル環境変数）はMLServerの推論プールのサイズを制御します。
期待される値は：

- `N`（`N > 0`の場合）、`N`個のワーカーのプールを作成します。
- `0`は並列推論機能を無効にします。
  つまり、推論はMLServerのメインプロセス内で行われます。

<!--
## References
-->

## 参考文献

```{bibliography}
:filter: docname in docnames
```
