<!--
# Serving a custom model with JSON serialization

The `mlserver` package comes with inference runtime implementations for `scikit-learn` and `xgboost` models.
However, some times we may also need to roll out our own inference server, with custom logic to perform inference.
To support this scenario, MLServer makes it really easy to create your own extensions, which can then be containerised and deployed in a production environment.
-->

<!--
## Overview

In this example, we create a simple `Hello World JSON` model that parses and modifies a JSON data chunk. This is often useful as a means to
quickly bootstrap existing models that utilize JSON based model inputs.
-->

<!--
## Serving

The next step will be to serve our model using `mlserver`. 
For that, we will first implement an extension which serve as the _runtime_ to perform inference using our custom `Hello World JSON` model.
-->

<!--
### Custom inference runtime

This is a trivial model to demonstrate how to conceptually work with JSON inputs / outputs. In this example:

- Parse the JSON input from the client
- Create a JSON response echoing back the client request as well as a server generated message
-->

# カスタムモデルの提供とJSONシリアライゼーション

`mlserver` パッケージには、`scikit-learn` と `xgboost` モデルのための推論ランタイム実装が含まれています。
しかし、時には独自の推論サーバーを展開し、推論を行うためのカスタムロジックが必要になることもあります。
このシナリオをサポートするために、MLServer は独自の拡張を非常に簡単に作成できるようにしており、それらをコンテナ化して本番環境にデプロイすることができます。

## 概要

この例では、クライアントからの JSON データチャンクを解析し、変更するシンプルな `Hello World JSON` モデルを作成します。これは、JSON ベースのモデル入力を利用する既存のモデルを迅速にブートストラップする手段としてよく利用されます。

## サービング

次のステップは、`mlserver` を使用してモデルを提供することです。
そのために、まずカスタムの `Hello World JSON` モデルを使用して推論を行う _ランタイム_ として機能する拡張を実装します。

### カスタム推論ランタイム

これは、JSON 入出力での作業方法を概念的に示すための簡単なモデルです。この例では：

- クライアントからの JSON 入力を解析する
- クライアントのリクエストとサーバーが生成したメッセージを反映した JSON レスポンスを作成する


```python
%%writefile jsonmodels.py
import json

from typing import Dict, Any
from mlserver import MLModel, types
from mlserver.codecs import StringCodec


class JsonHelloWorldModel(MLModel):
    async def load(self) -> bool:
        # Perform additional custom initialization here.
        print("Initialize model")

        # Set readiness flag for model
        return await super().load()

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        request = self._extract_json(payload)
        response = {
            "request": request,
            "server_response": "Got your request. Hello from the server.",
        }
        response_bytes = json.dumps(response).encode("UTF-8")

        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="echo_response",
                    shape=[len(response_bytes)],
                    datatype="BYTES",
                    data=[response_bytes],
                    parameters=types.Parameters(content_type="str"),
                )
            ],
        )

    def _extract_json(self, payload: types.InferenceRequest) -> Dict[str, Any]:
        inputs = {}
        for inp in payload.inputs:
            inputs[inp.name] = json.loads(
                "".join(self.decode(inp, default_codec=StringCodec))
            )

        return inputs

```

<!--
### Settings files

The next step will be to create 2 configuration files: 

- `settings.json`: holds the configuration of our server (e.g. ports, log level, etc.).
- `model-settings.json`: holds the configuration of our model (e.g. input type, runtime to use, etc.).
-->

### 設定ファイル

次のステップは、2つの設定ファイルを作成することです：

- `settings.json`：サーバーの設定を保持します（例：ポート、ログレベルなど）。
- `model-settings.json`：モデルの設定を保持します（例：入力タイプ、使用するランタイムなど）。

#### `settings.json`

```python
%%writefile settings.json
{
    "debug": "true"
}
```

#### `model-settings.json`

```python
%%writefile model-settings.json
{
    "name": "json-hello-world",
    "implementation": "jsonmodels.JsonHelloWorldModel"
}
```

<!--
### Start serving our model

Now that we have our config in-place, we can start the server by running `mlserver start .`. This needs to either be ran from the same directory where our config files are or pointing to the folder where they are.
-->

### モデルの提供を開始

設定が整ったので、`mlserver start .` を実行してサーバーを起動できます。これは、設定ファイルがある同じディレクトリから、またはそれらがあるフォルダーを指定して実行する必要があります。


```shell
mlserver start .
```

<!--
Since this command will start the server and block the terminal, waiting for requests, this will need to be ran in the background on a separate terminal.
-->

このコマンドはサーバーを起動し、リクエストを待機するため、ターミナルをブロックします。そのため、これは別のターミナルでバックグラウンドで実行する必要があります。

<!--
### Send test inference request (REST)

We now have our model being served by `mlserver`.
To make sure that everything is working as expected, let's send a request from our test set.

For that, we can use the Python types that `mlserver` provides out of box, or we can build our request manually.
-->

### テスト推論リクエストの送信（REST）

これで、モデルは `mlserver` によって提供されています。
すべてが期待通りに動作していることを確認するために、テストセットからリクエストを送信しましょう。

そのために、`mlserver` が提供する Python の型を使用することも、リクエストを手動で構築することもできます。


```python
import requests
import json
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=1)

inputs = {"name": "Foo Bar", "message": "Hello from Client (REST)!"}

# NOTE: this uses characters rather than encoded bytes. It is recommended that you use the `mlserver` types to assist in the correct encoding.
inputs_string = json.dumps(inputs)

inference_request = {
    "inputs": [
        {
            "name": "echo_request",
            "shape": [len(inputs_string)],
            "datatype": "BYTES",
            "data": [inputs_string],
        }
    ]
}

endpoint = "http://localhost:8080/v2/models/json-hello-world/infer"
response = requests.post(endpoint, json=inference_request)

print(f"full response:\n")
print(response)
# retrive text output as dictionary
inference_response = InferenceResponse.parse_raw(response.text)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
print(f"\ndata part:\n")
pp.pprint(output)
```

<!--
### Send test inference request (gRPC)

Utilizing string data with the gRPC interface can be a bit tricky. To ensure we are correctly handling inputs and outputs we will be handled correctly.

For simplicity in this case, we leverage the Python types that `mlserver` provides out of the box. Alternatively, the gRPC stubs can be generated regenerated from the V2 specification directly for use by non-Python as well as Python clients.
-->

### テスト推論リクエストの送信（gRPC）

文字列データを gRPC インターフェースで使用することは少し複雑になる場合があります。入出力が正しく処理されることを確実にするためです。

この場合、簡単のために `mlserver` が提供する Python の型を利用します。代わりに、V2 仕様から直接 gRPC スタブを生成し、Python 以外のクライアントおよび Python クライアントによる使用が可能です。

```python
import requests
import json
import grpc
from mlserver.codecs.string import StringRequestCodec
import mlserver.grpc.converters as converters
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.types as types
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=1)

model_name = "json-hello-world"
inputs = {"name": "Foo Bar", "message": "Hello from Client (gRPC)!"}
inputs_bytes = json.dumps(inputs).encode("UTF-8")

inference_request = types.InferenceRequest(
    inputs=[
        types.RequestInput(
            name="echo_request",
            shape=[len(inputs_bytes)],
            datatype="BYTES",
            data=[inputs_bytes],
            parameters=types.Parameters(content_type="str"),
        )
    ]
)

inference_request_g = converters.ModelInferRequestConverter.from_types(
    inference_request, model_name=model_name, model_version=None
)

grpc_channel = grpc.insecure_channel("localhost:8081")
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

response = grpc_stub.ModelInfer(inference_request_g)

print(f"full response:\n")
print(response)
# retrive text output as dictionary
inference_response = converters.ModelInferResponseConverter.to_types(response)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
print(f"\ndata part:\n")
pp.pprint(output)
```
