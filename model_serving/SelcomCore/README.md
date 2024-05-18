# Seldon Core: Blazing Fast, Industry-Ready ML

[https://docs.seldon.io/projects/seldon-core/en/latest/workflow/github-readme.html](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/github-readme.html)より。

A platform to deploy your machine learning models on Kubernetes at massive scale.

## Seldon Core V2 Now Available

![scv2_image](https://raw.githubusercontent.com/SeldonIO/seldon-core/master/doc/source/_static/scv2_banner.png)

<!--
Seldon Core V2 is now available. If you’re new to Seldon Core we recommend you start here. Check out the docs here and make sure to leave feedback on our slack community and submit bugs or feature requests on the repo. The codebase can be found in this branch.

Continue reading for info on Seldon Core V1…
-->

Seldon Core V2 が利用可能になりました。Seldon Core を初めて使用する場合は、ここから始めることをお勧めします。[こちらのドキュメント](https://seldon.io/docs/)をチェックして、是非フィードバックを私たちのSlackコミュニティに残し、リポジトリでバグや機能リクエストを提出してください。コードベースは[このブランチ](https://github.com/SeldonIO/seldon-core/tree/v2)で見つけることができます。

Seldon Core V1 についての情報を読み進めるには続けてください…


[video_play_icon](https://www.youtube.com/watch?v=5Q-03We8aDE)

<!--
## Overview

Seldon core converts your ML models (Tensorflow, Pytorch, H2o, etc.) or language wrappers (Python, Java, etc.) into production REST/GRPC microservices.

Seldon handles scaling to thousands of production machine learning models and provides advanced machine learning capabilities out of the box including Advanced Metrics, Request Logging, Explainers, Outlier Detectors, A/B Tests, Canaries and more.

Read the Seldon Core Documentation

Join our community Slack to ask any questions

Get started with Seldon Core Notebook Examples

Join our fortnightly online working group calls : Google Calendar

Learn how you can start contributing

Check out Blogs that dive into Seldon Core components

Watch some of the Videos and Talks using Seldon Core
-->

## 概要

Seldon Core は、ML モデル（Tensorflow、Pytorch、H2o など）または言語ラッパー（Python、Java など）を本番用の REST/GRPC マイクロサービスに変換します。

Seldon は数千の本番機械学習モデルへのスケーリングを処理し、高度なメトリクス、リクエストログ、説明者、異常検出器、A/B テスト、カナリアなどを含む、高度な機械学習機能を提供します。

* [Seldon Core ドキュメント](https://docs.seldon.io/projects/seldon-core/en/latest/)を読む
* 質問がある場合は、[コミュニティ Slack](https://join.slack.com/t/seldondev/shared_invite/zt-vejg6ttd-ksZiQs3O_HOtPQsen_labg) に参加してください
* [Seldon Core ノートブックの例](https://github.com/SeldonIO/seldon-core/tree/master/notebooks) で始める
* [Google カレンダー](https://calendar.google.com/calendar/u/0?cid=c2VsZG9uaW8uY29tX2pramVwY2c4N2ZrZjBwdWt2dGk1YnVlZm9rQGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20) で隔週のオンラインワーキンググループコールに参加する
* [コントリビュートの方法](https://docs.seldon.io/projects/seldon-core/en/latest/developer/contributing.html) を学ぶ
* [Seldon Core コンポーネントを掘り下げるブログ](https://www.seldon.io/blog/) をチェックする
* [Seldon Core を使用したビデオやトーク](https://docs.seldon.io/projects/seldon-core/en/latest/tutorials/videos.html)を視聴する


![](https://raw.githubusercontent.com/SeldonIO/seldon-core/master/doc/source/images/seldon-core-high-level.jpg)

<!--
High Level Features¶
With over 2M installs, Seldon Core is used across organisations to manage large scale deployment of machine learning models, and key benefits include:

Easy way to containerise ML models using our pre-packaged inference servers, custom servers, or language wrappers.

Out of the box endpoints which can be tested through Swagger UI, Seldon Python Client or Curl / GRPCurl.

Cloud agnostic and tested on AWS EKS, Azure AKS, Google GKE, Alicloud, Digital Ocean and Openshift.

Powerful and rich inference graphs made out of predictors, transformers, routers, combiners, and more.

Metadata provenance to ensure each model can be traced back to its respective training system, data and metrics.

Advanced and customisable metrics with integration to Prometheus and Grafana.

Full auditability through model input-output request logging integration with Elasticsearch.

Microservice distributed tracing through integration to Jaeger for insights on latency across microservice hops.

Secure, reliable and robust system maintained through a consistent security & updates policy.
-->

## 高レベル機能
200万回以上のインストールで、Seldon Core は多くの組織で大規模な機械学習モデルのデプロイを管理するために使用されており、主な利点は以下の通りです：

- プリパッケージされた推論サーバー、カスタムサーバー、または言語ラッパーを使用して ML モデルをコンテナ化する簡単な方法。
- Swagger UI、Seldon Python クライアント、または Curl / GRPCurl を通じてテストできる即時利用可能なエンドポイント。
- AWS EKS、Azure AKS、Google GKE、Alicloud、Digital Ocean、および Openshift でテストされたクラウド非依存。
- 予測器、トランスフォーマー、ルーター、コンバイナーなどから成る強力で豊富な推論グラフ。
- モデルがそれぞれのトレーニングシステム、データ、およびメトリクスにまで遡れるメタデータ出自。
- Prometheus と Grafana への統合を持つ高度でカスタマイズ可能なメトリクス。
- Elasticsearch との統合によるモデル入出力リクエストログによる完全な監査性。
- マイクロサービス間のホップにおけるレイテンシの洞察のための Jaeger への統合によるマイクロサービス分散トレース。
- 一貫したセキュリティおよびアップデートポリシーによって維持される安全で信頼性があり堅牢なシステム。


<!--
## Getting Started

Deploying your models using Seldon Core is simplified through our pre-packaged inference servers and language wrappers. Below you can see how you can deploy our “hello world Iris” example. You can see more details on these workflows in our Documentation Quickstart.
-->


## 入門

Seldon Core を使用してモデルをデプロイすることは、プリパッケージされた推論サーバーや言語ラッパーを通じて簡素化されます。以下では、私たちの「ハローワールドアイリス」の例をデプロイする方法を見ることができます。これらのワークフローの詳細は、[ドキュメントクイックスタート](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/quickstart.html)で確認できます。


<!--
### Install Seldon Core

Quick install using Helm 3 (you can also use Kustomize):
-->

### Seldon Core のインストール

Helm 3 を使用したクイックインストール（Kustomize も使用可能）：

```bash
kubectl create namespace seldon-system
```


```bash
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --namespace seldon-system \
    --set istio.enabled=true
    # You can set ambassador instead with --set ambassador.enabled=true
```

<!--
### Deploy your model using pre-packaged model servers

We provide optimized model servers for some of the most popular Deep Learning and Machine Learning frameworks that allow you to deploy your trained model binaries/weights without having to containerize or modify them.

You only have to upload your model binaries into your preferred object store, in this case we have a trained scikit-learn iris model in a Google bucket:
-->

### プリパッケージされたモデルサーバーを使用してモデルをデプロイ

私たちは、ディープラーニングおよび機械学習の最も一般的なフレームワークのいくつかのために最適化されたモデルサーバーを提供しており、コンテナ化や変更を行うことなくトレーニング済みのモデルバイナリ/ウェイトをデプロイできます。

お好みのオブジェクトストアにモデルバイナリをアップロードするだけです。この場合、Google バケットにトレーニング済みの scikit-learn アイリスモデルがあります：


```bash
gs://seldon-models/v1.19.0-dev/sklearn/iris/model.joblib
```

Create a namespace to run your model in:

```bash
kubectl create namespace seldon
```

<!--
We then can deploy this model with Seldon Core to our Kubernetes cluster using the pre-packaged model server for scikit-learn (SKLEARN_SERVER) by running the kubectl apply command below:
-->

次に、以下の `kubectl apply` コマンドを実行することで、プリパッケージされた scikit-learn 用モデルサーバー（SKLEARN_SERVER）を使用して、このモデルを Kubernetes クラスターに Seldon Core と共にデプロイできます：

```bash
$ kubectl apply -f - << END
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-model
  namespace: seldon
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
### Send API requests to your deployed model

Every model deployed exposes a standardised User Interface to send requests using our OpenAPI schema.

This can be accessed through the endpoint http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/ which will allow you to send requests directly through your browser.

![](https://raw.githubusercontent.com/SeldonIO/seldon-core/master/doc/source/images/rest-openapi.jpg)

Or alternatively you can send requests programmatically using our Seldon Python Client or another Linux CLI:
-->

### デプロイされたモデルへの API リクエストの送信

デプロイされた各モデルは、OpenAPI スキーマを使用してリクエストを送信するための標準化されたユーザーインターフェースを公開します。

これはエンドポイント `http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/` を通じてアクセスでき、ブラウザを通じて直接リクエストを送信できます。

![](https://raw.githubusercontent.com/SeldonIO/seldon-core/master/doc/source/images/rest-openapi.jpg)

または、Seldon Python クライアントや他の Linux CLI を使用してプログラムでリクエストを送信することもできます：


```bash
$ curl -X POST http://<ingress>/seldon/seldon/iris-model/api/v1.0/predictions \
    -H 'Content-Type: application/json' \
    -d '{ "data": { "ndarray": [[1,2,3,4]] } }'

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
### Deploy your custom model using language wrappers
For more custom deep learning and machine learning use-cases which have custom dependencies (such as 3rd party libraries, operating system binaries or even external systems), we can use any of the Seldon Core language wrappers.

You only have to write a class wrapper that exposes the logic of your model; for example in Python we can create a file Model.py:
-->

### 言語ラッパーを使用してカスタムモデルをデプロイ

カスタムの依存関係（サードパーティのライブラリ、オペレーティングシステムのバイナリ、または外部システムなど）を持つ、よりカスタムされたディープラーニングおよび機械学習のユースケースのために、Seldon Core の言語ラッパーのいずれかを使用できます。

モデルのロジックを公開するクラスラッパーを書くだけで済みます。例えば、Pythonでは `Model.py` ファイルを作成できます：


```python
import pickle

class Model:
    def __init__(self):
        self._model = pickle.loads( open("model.pickle", "rb") )

    def predict(self, X):
        output = self._model(X)
        return output
```

<!--
We can now containerize our class file using the Seldon Core s2i utils to produce the sklearn_iris image:
-->
Seldon Core の s2i ユーティリティを使用してクラスファイルをコンテナ化し、`sklearn_iris` イメージを生成できます：

```bash
s2i build . seldonio/seldon-core-s2i-python3:0.18 sklearn_iris:0.1
```

<!--
And we now deploy it to our Seldon Core Kubernetes Cluster:
-->
そして、これを Seldon Core Kubernetes クラスターにデプロイします：

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
    graph:
      name: classifier
    name: default
    replicas: 1
END
```

<!--
### Send API requests to your deployed model
Every model deployed exposes a standardised User Interface to send requests using our OpenAPI schema.

This can be accessed through the endpoint http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/ which will allow you to send requests directly through your browser.

![](https://raw.githubusercontent.com/SeldonIO/seldon-core/master/doc/source/images/rest-openapi.jpg)

Or alternatively you can send requests programmatically using our Seldon Python Client or another Linux CLI:
-->


デプロイされたモデルへの API リクエストの送信
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
Dive into the Advanced Production ML Integrations¶
Any model that is deployed and orchestrated with Seldon Core provides out of the box machine learning insights for monitoring, managing, scaling and debugging.

Below are some of the core components together with link to the logs that provide further insights on how to set them up.
-->

## 高度なプロダクションML統合について詳しく学ぶ

Seldon Core でデプロイおよびオーケストレーションされたどのモデルも、監視、管理、スケーリング、デバッグのための機械学習の洞察を提供します。

以下は、それらをセットアップする方法に関する洞察を提供するログへのリンクと共に、いくつかのコアコンポーネントです。

* Standard and custom metrics with prometheus
* Full audit trails with ELK request logging
* Explainers for Machine Learning Interpretability
* Outlier and Adversarial Detectors for Monitoring
* CI/CD for MLOps at Massive Scale
* Distributed tracing for performance monitoring


