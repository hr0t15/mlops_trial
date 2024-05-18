<!--
# Custom Inference Runtimes

There may be cases where the [inference runtimes](./index) offered
out-of-the-box by MLServer may not be enough, or where you may need **extra
custom functionality** which is not included in MLServer (e.g. custom codecs).
To cover these cases, MLServer lets you create custom runtimes very easily.

This page covers some of the bigger points that need to be taken into account
when extending MLServer.
You can also see this [end-to-end example](../examples/custom/README) which
walks through the process of writing a custom runtime.
-->

# カスタム推論ランタイム

MLServerが提供する[推論ランタイム](./index)だけでは十分でない場合や、MLServerに含まれていない**追加のカスタム機能**が必要な場合（例えば、カスタムコーデックなど）があります。これらのケースに対応するために、MLServerでは非常に簡単にカスタムランタイムを作成できます。

このページでは、MLServerを拡張する際に考慮すべきいくつかの重要なポイントを取り上げています。
カスタムランタイムの作成プロセスを説明する[エンドツーエンドの例](../examples/custom/README)も参照できます。

<!--
## Writing a custom inference runtime

MLServer is designed as an easy-to-extend framework, encouraging users to write
their own custom runtimes easily.
The starting point for this is the {class}`MLModel <mlserver.MLModel>`
abstract class, whose main methods are:

- {func}`load() <mlserver.MLModel.load>`:
  Responsible for loading any artifacts related to a model (e.g. model
  weights, pickle files, etc.).
- {func}`unload() <mlserver.MLModel.unload>`:
  Responsible for unloading the model, freeing any resources (e.g. GPU memory,
  etc.).
- {func}`predict() <mlserver.MLModel.predict>`:
  Responsible for using a model to perform inference on an incoming data point.

Therefore, the _"one-line version"_ of how to write a custom runtime is to
write a custom class extending from {class}`MLModel <mlserver.MLModel>`,
and then overriding those methods with your custom logic.
-->

## カスタム推論ランタイムの作成

MLServerは拡張が容易なフレームワークとして設計されており、ユーザーが自分自身のカスタムランタイムを簡単に作成することを奨励しています。
これを開始するための出発点は、{class}`MLModel <mlserver.MLModel>` 抽象クラスであり、その主なメソッドは以下の通りです：

- {func}`load() <mlserver.MLModel.load>`:
  モデルに関連するアーティファクト（例：モデルの重み、ピクルファイルなど）のローディングを担当します。
- {func}`unload() <mlserver.MLModel.unload>`:
  モデルをアンロードし、リソース（例：GPUメモリなど）を解放することを担当します。
- {func}`predict() <mlserver.MLModel.predict>`:
  送信されたデータポイントに対して推論を行うためにモデルを使用することを担当します。

したがって、カスタムランタイムを書く方法の_"ワンラインバージョン"_は、{class}`MLModel <mlserver.MLModel>`から継承するカスタムクラスを書き、それらのメソッドをカスタムロジックでオーバーライドすることです。


```{code-block} python
---
emphasize-lines: 7-8, 12-13
---
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse

class MyCustomRuntime(MLModel):

  async def load(self) -> bool:
    # TODO: Replace for custom logic to load a model artifact
    self._model = load_my_custom_model()
    return True

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    # TODO: Replace for custom logic to run inference
    return self._model.predict(payload)
```

<!--
### Simplified interface

MLServer exposes an alternative _"simplified" interface_ which can be used to
write custom runtimes.
This interface can be enabled by decorating your `predict()` method with the
`mlserver.codecs.decode_args` decorator.
This will let you specify in the method signature both how you want your
request payload to be decoded and how to encode the response back.

Based on the information provided in the method signature, MLServer will
automatically decode the request payload into the different inputs specified as
keyword arguments.
Under the hood, this is implemented through [MLServer's codecs and content types
system](./content-type.md).

```{note}
MLServer's _"simplified" interface_ aims to cover use cases where encoding /
decoding can be done through one of the codecs built-in into the MLServer
package.
However, there are instances where this may not be enough (e.g. variable number
of inputs, variable content types, etc.).
For these types of cases, please use MLServer's [_"advanced"
interface_](#writing-a-custom-inference-runtime), where you will have full
control over the full encoding / decoding process.
```

As an example of the above, let's assume a model which

- Takes two lists of strings as inputs:
  - `questions`, containing multiple questions to ask our model.
  - `context`, containing multiple contexts for each of the
    questions.
- Returns a Numpy array with some predictions as the output.

Leveraging MLServer's simplified notation, we can represent the above as the
following custom runtime:
-->


### 簡略化インターフェース

MLServerは、カスタムランタイムを作成するために使用できる別の_"簡略化された"インターフェース_を公開しています。
このインターフェースは、`predict()`メソッドを`mlserver.codecs.decode_args`デコレータで装飾することにより有効にすることができます。
これにより、リクエストペイロードのデコード方法とレスポンスのエンコード方法をメソッドのシグネチャで指定できます。

メソッドのシグネチャで提供された情報に基づいて、MLServerはリクエストペイロードを自動的に異なる入力にデコードします。これはキーワード引数として指定されます。
このプロセスは、内部的に[MLServerのコーデックとコンテントタイプシステム](./content-type.md)を通じて実装されています。

```{note}
MLServerの_"簡略化された"インターフェース_は、エンコーディング/デコーディングがMLServerパッケージに組み込まれているコーデックのいずれかを通じて行うことができるユースケースをカバーすることを目的としています。
しかし、これだけでは十分でない場合もあります（例：入力数が可変である、コンテントタイプが可変であるなど）。
このようなケースについては、MLServerの[_"高度な"インターフェース_](#writing-a-custom-inference-runtime)を使用してください。ここでは、エンコーディング/デコーディングプロセス全体を完全に制御できます。
```


上記の例として、以下のようなモデルを想定しましょう。

- 入力として2つの文字列リストを取ります：
  - `questions`は、モデルに尋ねる複数の質問を含んでいます。
  - `context`は、それぞれの質問のための複数の文脈を含んでいます。
- 出力としていくつかの予測を含むNumpy配列を返します。

MLServerの簡略化された表記を活用して、上記を以下のようなカスタムランタイムとして表現できます：


```{code-block} python
---
emphasize-lines: 2-3, 12-13
---
from mlserver import MLModel
from mlserver.codecs import decode_args
from typing import List

class MyCustomRuntime(MLModel):

  async def load(self) -> bool:
    # TODO: Replace for custom logic to load a model artifact
    self._model = load_my_custom_model()
    return True

  @decode_args
  async def predict(self, questions: List[str], context: List[str]) -> np.ndarray:
    # TODO: Replace for custom logic to run inference
    return self._model.predict(questions, context)
```

<!--
Note that, the method signature of our `predict` method now specifies:

- The input names that we should be looking for in the request
  payload (i.e. `questions` and `context`).
- The expected content type for each of the request inputs (i.e. `List[str]` on
  both cases).
- The expected content type of the response outputs (i.e. `np.ndarray`).
-->

<!--
### Read and write headers

```{note}
The `headers` field within the `parameters` section of the request / response
is managed by MLServer.
Therefore, incoming payloads where this field has been explicitly modified will
be overriden.
```

There are occasions where custom logic must be made conditional to extra
information sent by the client outside of the payload.
To allow for these use cases, MLServer will map all incoming HTTP headers (in
the case of REST) or metadata (in the case of gRPC) into the `headers` field of
the `parameters` object within the `InferenceRequest` instance.
-->

`predict`メソッドのメソッドシグネチャには、現在以下が指定されています：

- リクエストペイロードで探すべき入力名（つまり、`questions`と`context`）。
- 各リクエスト入力の期待されるコンテントタイプ（両方のケースで`List[str]`）。
- レスポンス出力の期待されるコンテントタイプ（つまり、`np.ndarray`）。

### ヘッダーの読み取りと書き込み

```{note}
リクエスト／レスポンスの`parameters`セクション内の`headers`フィールドはMLServerによって管理されます。
したがって、このフィールドが明示的に変更された受信ペイロードは上書きされます。
```

クライアントからペイロードの外部に送信された追加情報に条件付けられるカスタムロジックが必要な場合があります。
これらのユースケースに対応するために、MLServerはすべての受信HTTPヘッダー（RESTの場合）またはメタデータ（gRPCの場合）を`InferenceRequest`インスタンス内の`parameters`オブジェクトの`headers`フィールドにマッピングします。


```{code-block} python
---
emphasize-lines: 9-11
---
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse

class CustomHeadersRuntime(MLModel):

  ...

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    if payload.parameters and payload.parametes.headers:
      # These are all the incoming HTTP headers / gRPC metadata
      print(payload.parameters.headers)
    ...
```

<!--
Similarly, to return any HTTP headers (in the case of REST) or metadata (in the
case of gRPC), you can append any values to the `headers` field within the
`parameters` object of the returned `InferenceResponse` instance.
-->

同様に、HTTPヘッダー（RESTの場合）またはメタデータ（gRPCの場合）を返すためには、返される`InferenceResponse`インスタンスの`parameters`オブジェクト内の`headers`フィールドに任意の値を追加することができます。


```{code-block} python
---
emphasize-lines: 13
---
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse

class CustomHeadersRuntime(MLModel):

  ...

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    ...
    return InferenceResponse(
      # Include any actual outputs from inference
      outputs=[],
      parameters=Parameters(headers={"foo": "bar"})
    )
```

<!--
## Loading a custom MLServer runtime

MLServer lets you load custom runtimes dynamically into a running instance of
MLServer.
Once you have your custom runtime ready, all you need to is to move it to your
model folder, next to your `model-settings.json` configuration file.

For example, if we assume a flat model repository where each folder represents
a model, you would end up with a folder structure like the one below:
-->
## カスタムMLServerランタイムのロード

MLServerでは、カスタムランタイムを動作中のMLServerインスタンスに動的にロードすることができます。
カスタムランタイムが準備できたら、それをモデルフォルダに移動し、`model-settings.json`設定ファイルの隣に配置するだけです。

例えば、各フォルダがモデルを表すフラットなモデルリポジトリを想定すると、以下のようなフォルダ構造になります：


```bash
.
└── models
    └── sum-model
        ├── model-settings.json
        ├── models.py
```

<!--
Note that, from the example above, we are assuming that:

- Your custom runtime code lives in the `models.py` file.
- The `implementation` field of your `model-settings.json` configuration file
  contains the import path of your custom runtime (e.g.
  `models.MyCustomRuntime`).
-->
上記の例から、以下を想定しています：

- カスタムランタイムのコードは`models.py`ファイルに存在します。
- `model-settings.json`設定ファイルの`implementation`フィールドには、カスタムランタイムのインポートパス（例えば`models.MyCustomRuntime`）が含まれています。


  ```{code-block} json
  ---
  emphasize-lines: 3
  ---
  {
    "model": "sum-model",
    "implementation": "models.MyCustomRuntime"
  }
  ```

<!--
### Loading a custom Python environment

More often that not, your custom runtimes will depend on external 3rd party
dependencies which are not included within the main MLServer package.
In these cases, to load your custom runtime, MLServer will need access to these
dependencies.

It is possible to load this custom set of dependencies by providing them
through an [environment tarball](../examples/conda/README), whose path can be
specified within your `model-settings.json` file.

```{warning}
To load a custom environment, [parallel inference](./parallel-inference)
**must** be enabled.
```

```{warning}
When loading custom environments, MLServer will always use the same Python
interpreter that is used to run the main process.
In other words, all custom environments will use the same version of Python
than the main MLServer process.
```

If we take the [previous example](#loading-a-custom-mlserver-runtime) above as
a reference, we could extend it to include our custom environment as:
-->

### カスタムPython環境のロード

カスタムランタイムは、主なMLServerパッケージに含まれていない外部のサードパーティ依存関係に依存することが多いです。
このような場合、カスタムランタイムをロードするためには、MLServerがこれらの依存関係にアクセスできる必要があります。

[環境ターボール](../examples/conda/README)を提供することで、このカスタムセットの依存関係をロードすることが可能です。そのパスは`model-settings.json`ファイル内で指定できます。

```{warning}
カスタム環境をロードするには、[並列推論](./parallel-inference)を**有効にする必要があります**。
```

```{warning}
カスタム環境をロードする際、MLServerは常にメインプロセスを実行しているのと同じPythonインタープリタを使用します。
言い換えると、すべてのカスタム環境はメインのMLServerプロセスと同じバージョンのPythonを使用します。
```

[前の例](#loading-a-custom-mlserver-runtime)を参照として取ると、私たちのカスタム環境を含めるためにそれを拡張することができます：


```bash
.
└── models
    └── sum-model
        ├── environment.tar.gz
        ├── model-settings.json
        ├── models.py
```

<!--
Note that, in the folder layout above, we are assuming that:

- The `environment.tar.gz` tarball contains a pre-packaged version of your
  custom environment.
- The `environment_tarball` field of your `model-settings.json` configuration file
  points to your pre-packaged custom environment (i.e.
  `./environment.tar.gz`).
-->
上記のフォルダレイアウトでは、次のことを想定しています：

- `environment.tar.gz`ターボールには、カスタム環境の事前パッケージ化されたバージョンが含まれています。
- `model-settings.json`設定ファイルの`environment_tarball`フィールドが、事前パッケージ化されたカスタム環境（つまり`./environment.tar.gz`）を指しています。

  ```{code-block} json
  ---
  emphasize-lines: 5
  ---
  {
    "model": "sum-model",
    "implementation": "models.MyCustomRuntime",
    "parameters": {
      "environment_tarball": "./environment.tar.gz"
    }
  }
  ```

<!--
## Building a custom MLServer image

```{note}
The `mlserver build` command expects that a Docker runtime is available and
running in the background.
```

MLServer offers built-in utilities to help you build a custom MLServer image.
This image can contain any custom code (including custom inference runtimes),
as well as any custom environment, provided either through a [Conda environment
file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
or a `requirements.txt` file.

To leverage these, we can use the `mlserver build` command.
Assuming that we're currently on the folder containing our custom inference
runtime, we should be able to just run:
-->

## カスタムMLServerイメージの構築

`mlserver build`コマンドは、Dockerランタイムが利用可能でバックグラウンドで実行されていることを前提としています。

MLServerはカスタムMLServerイメージを構築するための組み込みユーティリティを提供します。このイメージには、カスタム推論ランタイムを含む任意のカスタムコードや、[Conda環境ファイル](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)または`requirements.txt`ファイルを通じて提供される任意のカスタム環境を含めることができます。

これを活用するために、`mlserver build`コマンドを使用できます。
現在、カスタム推論ランタイムを含むフォルダにいると仮定すると、次のコマンドを実行するだけで良いでしょう：



```bash
mlserver build . -t my-custom-server
```

<!--
The output will be a Docker image named `my-custom-server`, ready to be used.

### Custom Environment

The [`mlserver build`](../reference/cli) subcommand will search for any Conda
environment file (i.e. named either as `environment.yaml` or `conda.yaml`) and
/ or any `requirements.txt` present in your root folder.
These can be used to tell MLServer what Python environment is required in the
final Docker image.

```{note}
The environment built by the `mlserver build` will be global to the whole
MLServer image (i.e. every loaded model will, by default, use that custom
environment).
For Multi-Model Serving scenarios, it may be better to use [per-model custom
environments](#loading-a-custom-python-environment) instead - which will allow
you to run multiple custom environments at the same time.
```
-->

### カスタム環境

[`mlserver build`](../reference/cli)のサブコマンドは、ルートフォルダに存在するConda環境ファイル（`environment.yaml`または`conda.yaml`として命名されているもの）および/または`requirements.txt`を検索します。
これらを使用して、MLServerに最終的なDockerイメージで必要なPython環境を指定できます。

```{note}
`mlserver build`によって構築された環境は、MLServerイメージ全体で共通です（つまり、デフォルトでロードされたすべてのモデルがそのカスタム環境を使用します）。
Multi-Model Servingシナリオでは、[モデルごとのカスタム環境](#loading-a-custom-python-environment)を使用することが良い場合があります。これにより、複数のカスタム環境を同時に実行できます。
```

<!--
### Default Settings

The `mlserver build` subcommand will treat any
[`settings.json`](../reference/settings) or
[`model-settings.json`](../reference/model-settings) files present on your root
folder as the default settings that must be set in your final image.
Therefore, these files can be used to configure things like the default
inference runtime to be used, or to even include **embedded models** that will
always be present within your custom image.

```{note}
Default setting values can still be overriden by external environment variables
or model-specific `model-settings.json`.
```
-->

### デフォルト設定

`mlserver build`サブコマンドは、ルートフォルダに存在する任意の[`settings.json`](../reference/settings)または[`model-settings.json`](../reference/model-settings)ファイルを、最終イメージで設定する必要のあるデフォルト設定として扱います。
したがって、これらのファイルを使用して、使用するデフォルトの推論ランタイムを構成したり、常にカスタムイメージ内に存在する**組み込みモデル**を含めたりすることができます。

デフォルトの設定値は、外部環境変数またはモデル固有の`model-settings.json`によって引き続きオーバーライドできます。


<!--
### Custom Dockerfile

Out-of-the-box, the `mlserver build` subcommand leverages a default
`Dockerfile` which takes into account a number of requirements, like

- Supporting arbitrary user IDs.
- Building your [base custom environment](#custom-environment) on the fly.
- Configure a set of [default setting values](#default-settings).

However, there may be occasions where you need to customise your `Dockerfile`
even further.
This may be the case, for example, when you need to provide extra environment
variables or when you need to customise your Docker build process (e.g. by
using other _"Docker-less"_ tools, like
[Kaniko](https://github.com/GoogleContainerTools/kaniko) or
[Buildah](https://buildah.io/)).

To account for these cases, MLServer also includes a [`mlserver
dockerfile`](../reference/cli) subcommand which will just generate a
`Dockerfile` (and optionally a `.dockerignore` file) exactly like the one used
by the `mlserver build` command.
This `Dockerfile` can then be customised according to your needs.

````{note}
The base `Dockerfile` requires [Docker's
Buildkit](https://docs.docker.com/build/buildkit/) to be enabled.
To ensure BuildKit is used, you can use the `DOCKER_BUILDKIT=1` environment
variable, e.g.
-->

### カスタムDockerfile

`mlserver build`サブコマンドは、一連の要件を考慮したデフォルトの`Dockerfile`を利用します。

- 任意のユーザーIDのサポート。
- [ベースカスタム環境](#custom-environment)の動的ビルド。
- [デフォルトの設定値](#default-settings)の設定。

ただし、さらにDockerfileをカスタマイズする必要がある場合があります。
たとえば、追加の環境変数を提供する必要がある場合や、Dockerビルドプロセスをカスタマイズする必要がある場合（たとえば、他の _"Docker-less"_ ツール（[Kaniko](https://github.com/GoogleContainerTools/kaniko)や[Buildah](https://buildah.io/)など）を使用する場合）があります。

これらのケースを考慮して、MLServerには[`mlserver dockerfile`](../reference/cli)サブコマンドも用意されており、`mlserver build`コマンドで使用されるものとまったく同じ`Dockerfile`（およびオプションで`.dockerignore`ファイル）を生成します。
この`Dockerfile`は、その後、必要に応じてカスタマイズできます。

````{note}
ベースの`Dockerfile`を使用するには、Dockerの[Buildkit](https://docs.docker.com/build/buildkit/)を有効にする必要があります。
BuildKitを使用するようにするには、`DOCKER_BUILDKIT=1`環境変数を使用できます。たとえば、
```bash
DOCKER_BUILDKIT=1 docker build . -t my-custom-runtime:0.1.0
```
````
