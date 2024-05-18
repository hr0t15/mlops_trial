# Seldon Coreのインストール

[Install Locally - Set Up Kind](https://docs.seldon.io/projects/seldon-core/en/latest/install/kind.html#set-up-kind)以降の操作。

## Set Up Kind

```bash
kind create cluster --name seldon
```

<!--
## Install Cluster Ingress

Ingress is a Kubernetes object that provides routing rules for your cluster. It manages the incomming traffic and routes it to the services running inside the cluster.

Seldon Core supports using either Istio or Ambassador to manage incomming traffic. Seldon Core automatically creates the objects and rules required to route traffic to your deployed machine learning models.

Istio is an open source service mesh. If the term service mesh is unfamiliar to you, it’s worth reading a little more about Istio.
-->

## クラスター Ingress のインストール

Ingress は Kubernetes オブジェクトで、クラスターのルーティングルールを提供します。それは受信トラフィックを管理し、クラスター内で実行されているサービスにルーティングします。

Seldon Core は Istio または Ambassador のいずれかを使用して受信トラフィックを管理することをサポートしています。Seldon Core は、デプロイされた機械学習モデルへのトラフィックをルーティングするために必要なオブジェクトとルールを自動的に作成します。

Istio はオープンソースのサービスメッシュです。もしサービスメッシュという用語が馴染みがない場合は、Istio についてもう少し読む価値があります。


<!--
### Download Istio

For Linux and macOS, the easiest way to download Istio is using the following command:
-->


### Istio のダウンロード

Linux と macOS では、Istio をダウンロードする最も簡単な方法は次のコマンドを使用することです：

```bash
curl -L https://istio.io/downloadIstio | sh -
```

Move to the Istio package directory. For example, if the package is istio-1.11.4:

```bash
cd istio-1.11.4
```

Add the istioctl client to your path (Linux or macOS):


```bash
export PATH=$PWD/bin:$PATH
```

<!--
### Install Istio

Istio provides a command line tool istioctl to make the installation process easy. The demo configuration profile has a good set of defaults that will work on your local cluster.
-->

### Istio のインストール

Istio はインストールプロセスを簡単にするためのコマンドラインツール `istioctl` を提供しています。デモ設定プロファイルには、ローカルクラスターで動作する良いデフォルト設定が含まれています。


```bash
istioctl install --set profile=demo -y
```

<!--
The namespace label istio-injection=enabled instructs Istio to automatically inject proxies alongside anything we deploy in that namespace. We’ll set it up for our default namespace:
-->
名前空間ラベル `istio-injection=enabled` は、その名前空間にデプロイされるものすべてに対して Istio が自動的にプロキシを注入するよう指示します。私たちはデフォルトの名前空間にそれを設定します：


```bash
kubectl label namespace default istio-injection=enabled
```

<!--
### Create Istio Gateway

In order for Seldon Core to use Istio’s features to manage cluster traffic, we need to create an Istio Gateway by running the following command:

Warning

You will need to copy the entire command from the code block below
-->

### Istio Gateway の作成

Seldon Core が Istio の機能を使用してクラスタートラフィックを管理するためには、以下のコマンドを実行して Istio Gateway を作成する必要があります：

警告

以下のコードブロックからコマンド全体をコピーする必要があります。


```bash
kubectl apply -f - << END
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway # use istio default controller
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
END
```

<!--
For custom configuration and more details on installing seldon core with Istio please see the Istio Ingress page.
-->

<!--
## Install Seldon Core

Before we install Seldon Core, we’ll create a new namespace seldon-system for the operator to run in:

kubectl create namespace seldon-system
We’re now ready to install Seldon Core in our cluster. Run the following command for your choice of Ingress:
-->

カスタム設定や Istio での Seldon Core のインストールの詳細については、Istio Ingress ページを参照してください。

## Seldon Core のインストール

Seldon Core をインストールする前に、オペレーターが動作する新しい名前空間 seldon-system を作成します：

```bash
kubectl create namespace seldon-system
```

名前空間が正常に作成されたことを確認します。

```bash
kubectl get namespace seldon-system
```


これでクラスターに Seldon Core をインストールする準備が整いました。Ingress の選択に応じて次のコマンドを実行します：

```bash
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --namespace seldon-system
```

<!--
You can check that your Seldon Controller is running by doing:
-->
Seldon Controller が実行中であることを確認するには、次の操作を行います：

```bash
kubectl get pods -n seldon-system
```

<!--
You should see a seldon-controller-manager pod with STATUS=Running.
-->
`STATUS=Running` と表示される seldon-controller-manager ポッドが見えるはずです。

<!--
## Local Port Forwarding

Because your kubernetes cluster is running locally, we need to forward a port on your local machine to one in the cluster for us to be able to access it externally. You can do this by running:
-->
## ローカルポートフォワーディング

Kubernetesクラスターがローカルで実行されているため、外部からアクセスできるようにローカルマシンのポートをクラスタ内のポートに転送する必要があります。これは次のコマンドを実行することで行えます：


```bash
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```
<!--
This will forward any traffic from port 8080 on your local machine to port 80 inside your cluster.

You have now successfully installed Seldon Core on a local cluster and are ready to start deploying models as production microservices.
-->
これにより、ローカルマシンのポート8080からのすべてのトラフィックがクラスタ内のポート80に転送されます。

これで、ローカルクラスターに Seldon Core を正常にインストールし、モデルをプロダクションマイクロサービスとしてデプロイする準備が整いました。
