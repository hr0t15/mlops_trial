# Quickstart

[https://docs.seldon.io/projects/seldon-core/en/latest/workflow/quickstart.html](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/quickstart.html)

<!--
In this page we have put together a very containerised example that will get you up and running with your first Seldon Core model.

We will show you how to deploy your model using a pre-packaged model server, as well as a language wrapper for more custom servers.

You can dive into a deeper dive of each of the components and stages of the Seldon Core Workflow.
-->

このページでは、最初のSeldon Coreモデルを使用してすぐに起動できる非常にコンテナ化された例をまとめています。

プリパッケージされたモデルサーバーを使用してモデルをデプロイする方法と、よりカスタムされたサーバーのための言語ラッパーを使用する方法を示します。

Seldon Core ワークフローの各コンポーネントとステージをより深く掘り下げることができます。

<!--
## Seldon Core Workflow

Once you’ve installed Seldon Core, you can productionise your model with the following three steps:

1. Wrap your model using our prepackaged inference servers or language wrappers
2. Define and deploy your Seldon Core inference graph
3. Send predictions and monitor performance
-->

## Seldon Core ワークフロー

Seldon Core をインストールしたら、次の3つのステップでモデルをプロダクション化できます：

1. プリパッケージされた推論サーバーまたは言語ラッパーを使用してモデルをラップします
2. Seldon Core 推論グラフを定義してデプロイします
3. 予測を送信し、パフォーマンスを監視します


<!--
### 1. Wrap Your Model

The components you want to run in production need to be wrapped as Docker containers that respect the Seldon microservice API. You can create models that serve predictions, routers that decide on where requests go, such as A-B Tests, Combiners that combine responses and transformers that provide generic components that can transform requests and/or responses.

To allow users to easily wrap machine learning components built using different languages and toolkits we provide wrappers that allow you easily to build a docker container from your code that can be run inside seldon-core. Our current recommended tool is RedHat’s Source-to-Image. More detail can be found in Wrapping your models docs.
-->

### 1. モデルをラップする

プロダクションで実行したいコンポーネントは、SeldonマイクロサービスAPIを尊重するDockerコンテナとしてラップする必要があります。予測を提供するモデル、A-Bテストなどのリクエスト先を決定するルーター、レスポンスを組み合わせるコンバイナー、リクエストおよび/またはレスポンスを変換する汎用コンポーネントを提供するトランスフォーマーを作成できます。

異なる言語やツールキットを使用して構築された機械学習コンポーネントを簡単にラップできるようにするために、コードからDockerコンテナを簡単に構築できるラッパーを提供しています。これはseldon-core内で実行できます。現在推奨されているツールはRedHatのSource-to-Imageです。詳細はモデルのラッピングに関するドキュメントで見つけることができます。

<!--
### 2. Define Runtime Service Graph

To run your machine learning graph on Kubernetes you need to define how the components you created in the last step fit together to represent a service graph. This is defined inside a SeldonDeployment Kubernetes Custom resource. A guide to constructing this inference graph is provided.

![graph](https://docs.seldon.io/projects/seldon-core/en/latest/_images/graph1.png)
-->

### 2. ランタイムサービスグラフを定義する

Kubernetes上で機械学習グラフを実行するには、前のステップで作成したコンポーネントがどのように組み合わさってサービスグラフを表すかを定義する必要があります。これは SeldonDeployment Kubernetes カスタムリソース内で定義されます。この推論グラフを構築するためのガイドが提供されています。

![graph](https://docs.seldon.io/projects/seldon-core/en/latest/_images/graph1.png)

<!--
### 3. Deploy and Serve Predictions

You can use kubectl to deploy your ML service like any other Kubernetes resource. This is discussed here. Once deployed you can get predictions by calling the exposed API.
-->

### 3. デプロイして予測を提供する

kubectlを使用して、他のKubernetesリソースと同様にMLサービスをデプロイできます。これについては[ここで議論されています](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/deploy.html)。デプロイしたら、公開されたAPIを呼び出して予測を取得できます。


<!--
## Hands on Example of Seldon Core Workflow

### Install Seldon Core in your Cluster

Install using Helm 3 (you can also use Kustomize)
-->

## Seldon Core ワークフローの実践的な例

### クラスターに Seldon Core をインストール

Helm 3 を使用してインストール（Kustomize も使用可能）


```bash
kubectl create namespace seldon-system

helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --namespace seldon-system \
    --set istio.enabled=true
    # You can set ambassador instead with --set ambassador.enabled=true
```

<!--
For a more advanced guide that shows you how to install Seldon Core with many different options and parameters you can dive further in our detailed installation guide.
-->

詳細なインストールガイドで、さまざまなオプションとパラメータを使用して Seldon Core をインストールする方法を示す、より高度なガイドをご覧ください。

<!--
### Productionise your first Model with Seldon Core

There are two main ways you can productionise using Seldon Core:

* Wrap your model with our pre-packaged inference servers
* Wrap your model with our language wrappers
-->

### Seldon Core で最初のモデルをプロダクション化

Seldon Core を使用してプロダクション化する主な方法は2つあります：

* プリパッケージされた推論サーバーでモデルをラップする
* 言語ラッパーでモデルをラップする

<!--
#### Wrap your model with our pre-packaged inference servers

You can use our pre-packaged inference servers which are optimized for popular machine learning frameworks and languages, and allow for simplified workflows that can be scaled across large number of usecases.

A typical workflow would normally be programmatic (triggered through CI/CD), however below we show the commands you would normally carry out.
-->

#### プリパッケージされた推論サーバーでモデルをラップする

人気のある機械学習フレームワークや言語に最適化されたプリパッケージされた推論サーバーを使用できます。これにより、多くのユースケースにスケールできる簡素化されたワークフローが可能になります。

典型的なワークフローは通常プログラム的です（CI/CDを通じてトリガーされますが）、以下では通常実行するコマンドを示します。

<!--
##### 1. Export your model binaries / artifacts

Export your model binaries using the instructions provided in the requirements outlined in the respective pre-packaged model server you are planning to use.
-->

##### 1. モデルのバイナリ/アーティファクトをエクスポート

使用予定のプリパッケージされたモデルサーバーで概説された要件の指示に従ってモデルバイナリをエクスポートします。

```
>>my_sklearn_model.train(...)
>>joblib.dump(my_sklearn_model, "model.joblib")

[Created file at /mypath/model.joblib]
```

<!--
##### 2. Upload your model to an object store

You can upload your models into any of the object stores supported by our pre-package model server file downloader, or alternatively add your custom file downloader.

For simplicity we have already uploaded it to the bucket so you can just proceed to the next step and run your model on Seldon Core.
-->

##### 2. モデルをオブジェクトストアにアップロード

プリパッケージされたモデルサーバーファイルダウンローダーがサポートするオブジェクトストアのいずれかにモデルをアップロードすることができます。または、カスタムファイルダウンローダーを追加することもできます。

簡単にするために、バケットにすでにアップロードしてあるので、次のステップに進んで Seldon Core でモデルを実行するだけです。


```bash
$ gsutil cp model.joblib gs://seldon-models/v1.19.0-dev/sklearn/iris/model.joblib

[ Saved into gs://seldon-models/v1.19.0-dev/sklearn/iris/model.joblib ]
```

<!--
##### 3. Deploy to Seldon Core in Kubernetes

Finally you can just deploy your model by loading the binaries/artifacts using the pre-packaged model server of your choice. You can build complex inference graphs that use multiple components for inference.
-->

##### 3. Kubernetes の Seldon Core にデプロイ

最終的に、選択したプリパッケージされたモデルサーバーを使用してバイナリ/アーティファクトをロードすることで、モデルをデプロイできます。複数のコンポーネントを使用する複雑な推論グラフを構築できます。


```bash
$ kubectl apply -f - << END
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-model
  namespace: model-namespace
spec:
  name: iris
  predictors:
  - graph:
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.19.0-dev/sklearn/iris
      name: classifier
    name: default
    replicas: 1
END
```

<!--
4. Send a request in Kubernetes cluster

Every model deployed exposes a standardised User Interface to send requests using our OpenAPI schema.

This can be accessed through the endpoint http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/ which will allow you to send requests directly through your browser.

![](https://docs.seldon.io/projects/seldon-core/en/latest/images/rest-openapi.jpg)

Or alternatively you can send requests programmatically using our Seldon Python Client or another Linux CLI:
-->

4. Kubernetes クラスターでリクエストを送信

デプロイされた各モデルは、OpenAPI スキーマを使用してリクエストを送信するための標準化されたユーザーインターフェースを公開します。

これはエンドポイント `http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/` を通じてアクセスでき、ブラウザを通じて直接リクエストを送信できます。

![](https://docs.seldon.io/projects/seldon-core/en/latest/images/rest-openapi.jpg)

または、Seldon Python クライアントや他の Linux CLI を使用してプログラムでリクエストを送信することもできます：

```bash
$ curl -X POST http://<ingress>/seldon/model-namespace/iris-model/api/v1.0/predictions \
    -H 'Content-Type: application/json' \
    -d '{ "data": { "ndarray": [1,2,3,4] } }' | json_pp

{
   "meta" : {},
   "data" : {
      "names" : [
         "t:0",
         "t:1",
         "t:2"
      ],
      "ndarray" : [
         [
            0.000698519453116284,
            0.00366803903943576,
            0.995633441507448
         ]
      ]
   }
}
```

<!--
Wrap your model with our language wrappers
Below are the high level steps required to containerise your model using Seldon Core’s Language Wrappers.

Language wrappers are used for more custom use-cases that require dependencies that are not covered by our pre-packaged model servers. Language wrappers can be built using our graduated Python and Java wrappers - for further details check out our Language Wrappers section.
-->

#### 言語ラッパーを使用してモデルをラップ
以下は、Seldon Core の言語ラッパーを使用してモデルをコンテナ化するために必要な高レベルの手順です。

言語ラッパーは、プリパッケージされたモデルサーバーではカバーされていない依存関係が必要な、よりカスタムされたユースケースに使用されます。言語ラッパーは、成熟した Python と Java のラッパーを使用して構築することができます。詳細については、言語ラッパーのセクションをご覧ください。

<!--
1. Export your model binaries and/or artifacts:

In this case we are also exporting the model binaries/artifacts, but we will be in charge of the logic to load the models. This means that we can use third party dependencies and even external system calls. Seldon Core is running production use-cases with very heterogeneous models.
-->

##### 1. モデルのバイナリおよび/またはアーティファクトをエクスポートします：

この場合もモデルのバイナリ/アーティファクトをエクスポートしますが、モデルのロードロジックを担当します。これは、サードパーティの依存関係や外部システムコールを使用できることを意味します。Seldon Core は、非常に異質なモデルを使用したプロダクションユースケースを実行しています。

```
>> my_sklearn_model.train(...)
>> joblib.dump(my_sklearn_model, "model.joblib")

[Created file at /mypath/model.joblib]
```

<!--
##### 2. Create a wrapper class Model.py

In this case we’re using the Python language wrapper, which allows us to create a custom wrapper file which allows us to expose all functionality through the predict method - any HTTP/GRPC requests sent through the API are passed to that function, and the response will contain whatever we return from that function.

The python SDK also allows for other functions such as load for loading logic, metrics for custom Prometheus metrics, tags for metadata, and more.
-->

##### 2. ラッパークラス Model.py を作成

この場合、Python 言語ラッパーを使用しています。これにより、カスタムラッパーファイルを作成でき、predict メソッドを通じてすべての機能を公開できます。APIを通じて送信された HTTP/GRPC リクエストはすべてその関数に渡され、その関数から返されるものは何でもレスポンスに含まれます。

Python SDK は、ロードロジックのための load、カスタム Prometheus メトリクスのための metrics、メタデータのための tags など、他の関数もサポートしています。

```python
class Model:
    def __init__(self):
        self._model = joblib.load("model.joblib")

    def predict(self, X):
        output = self._model(X)
        return output
```

<!--
##### 3. Test model locally

Before we deploy our model to production, we can actually run our model locally using the Python seldon-core Module microservice CLI functionality.
-->

##### 3. モデルをローカルでテスト

本番環境にモデルをデプロイする前に、Python の seldon-core モジュールのマイクロサービス CLI 機能を使用して、ローカルでモデルを実行することができます。

```
$ seldon-core-microservice Model REST --service-type MODEL

2020-03-23 16:59:17,366 - werkzeug:_log:122 - INFO:   * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)

$ curl -X POST localhost:5000/api/v1.0/predictions \
    -H 'Content-Type: application/json' \
    -d '{ "data": { "ndarray": [1,2,3,4] } }' \
    | json_pp

{
   "meta" : {},
   "data" : {
      "names" : [
         "t:0",
         "t:1",
         "t:2"
      ],
      "ndarray" : [
         [
            0.000698519453116284,
            0.00366803903943576,
            0.995633441507448
         ]
      ]
   }
}
```

<!--
##### 4. Use the Seldon tools to containerise your model

Now we can use the Seldon Core utilities to convert our python class into a fully fledged Seldon Core microservice. In this case we are also containerising the model binaries.

The result below is a container with the name sklearn_iris and the tag 0.1 which we will be able to deploy using Seldon Core.
-->

##### 4. Seldon ツールを使用してモデルをコンテナ化

これで、Seldon Core ユーティリティを使用して Python クラスを完全な Seldon Core マイクロサービスに変換できます。この場合、モデルバイナリもコンテナ化しています。

以下の結果は、Seldon Core を使用してデプロイできる、名前が sklearn_iris でタグが 0.1 のコンテナです。


```
s2i build . seldonio/seldon-core-s2i-python3:1.19.0-dev sklearn_iris:0.1
```

<!--
##### 5. Deploy to Kubernetes

Similar to what we did with the pre-packaged model server, we define here our deployment structure however we also have to specify the container that we just built, together with any further containerSpec options we may want to add.
-->

##### 5. Kubernetes へのデプロイ

プリパッケージされたモデルサーバーで行ったように、ここでデプロイ構造を定義しますが、さらに、今構築したコンテナを指定し、追加したい任意の containerSpec オプションも指定する必要があります。

```bash
$ kubectl apply -f - << END
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-model
  namespace: model-namespace
spec:
  name: iris
  predictors:
  - componentSpecs:
    - spec:
      containers:
      - name: classifier
        image: sklearn_iris:0.1
  - graph:
      name: classifier
    name: default
    replicas: 1
END
```

<!--
##### 6. Send a request to your deployed model in Kubernetes

Every model deployed exposes a standardised User Interface to send requests using our OpenAPI schema.

This can be accessed through the endpoint http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/ which will allow you to send requests directly through your browser.

![](https://raw.githubusercontent.com/SeldonIO/seldon-core/master/doc/source/images/rest-openapi.jpg)

Or alternatively you can send requests programmatically using our Seldon Python Client or another Linux CLI:
-->

##### 6. Kubernetes にデプロイされたモデルへのリクエスト送信

デプロイされた各モデルは、OpenAPI スキーマを使用してリクエストを送信するための標準化されたユーザーインターフェースを公開します。

これはエンドポイント `http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/` を通じてアクセスでき、ブラウザを通じて直接リクエストを送信できます。

![](https://raw.githubusercontent.com/SeldonIO/seldon-core/master/doc/source/images/rest-openapi.jpg)

または、Seldon Python クライアントや他の Linux CLI を使用してプログラムでリクエストを送信することもできます：


```bash
$ curl -X POST http://<ingress>/seldon/model-namespace/iris-model/api/v1.0/predictions \
    -H 'Content-Type: application/json' \
    -d '{ "data": { "ndarray": [1,2,3,4] } }' | json_pp

{
   "meta" : {},
   "data" : {
      "names" : [
         "t:0",
         "t:1",
         "t:2"
      ],
      "ndarray" : [
         [
            0.000698519453116284,
            0.00366803903943576,
            0.995633441507448
         ]
      ]
   }
}
```

<!--
## Hands on Examples

Below are a set of Jupyter notebooks that you can try out yourself for deploying Seldon Core as well as using some of the more advanced features.
-->

## 実践的な例

以下は、Seldon Core をデプロイするため、またはより高度な機能を使用するために自分で試すことができる一連の Jupyter ノートブックです。

以下リンクあり  
[ここ](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/quickstart.html#hands-on-examples)からたどれます。

