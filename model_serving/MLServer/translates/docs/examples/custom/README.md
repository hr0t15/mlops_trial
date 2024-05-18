<!--
# Serving a custom model

The `mlserver` package comes with inference runtime implementations for `scikit-learn` and `xgboost` models.
However, some times we may also need to roll out our own inference server, with custom logic to perform inference.
To support this scenario, MLServer makes it really easy to create your own extensions, which can then be containerised and deployed in a production environment.
-->

# カスタムモデルの提供

`mlserver` パッケージには、`scikit-learn` と `xgboost` モデルの推論ランタイム実装が含まれています。
しかし、時には独自の推論サーバーを展開し、推論を行うためのカスタムロジックが必要になることもあります。
このシナリオをサポートするために、MLServer は独自の拡張を非常に簡単に作成できるようにしており、それらをコンテナ化して本番環境にデプロイすることができます。

<!--
## Overview

In this example, we will train a [`numpyro` model](http://num.pyro.ai/en/stable/).
The `numpyro` library streamlines the implementation of probabilistic models, abstracting away advanced inference and training algorithms.

Out of the box, `mlserver` doesn't provide an inference runtime for `numpyro`.
However, through this example we will see how easy is to develop our own.
-->

## 概要

この例では、[`numpyro` モデル](http://num.pyro.ai/en/stable/)をトレーニングします。
`numpyro` ライブラリは、確率モデルの実装を効率化し、高度な推論とトレーニングアルゴリズムを抽象化します。

`mlserver` はデフォルトでは `numpyro` の推論ランタイムを提供していません。
しかし、この例を通じて、私たち自身の推論ランタイムを開発するのがいかに簡単かを見ていきます。

<!--
## Training

The first step will be to train our model.
This will be a very simple bayesian regression model, based on an example provided in the [`numpyro` docs](https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb).

Since this is a probabilistic model, during training we will compute an approximation to the posterior distribution of our model using MCMC.
-->

## トレーニング

最初のステップはモデルをトレーニングすることです。
これは非常にシンプルなベイジアン回帰モデルであり、[`numpyro` ドキュメント](https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb)で提供されている例に基づいています。

これは確率モデルであるため、トレーニング中に MCMC を使用してモデルの事後分布の近似を計算します。


```python
# Original source code and more details can be found in:
# https://nbviewer.jupyter.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb


import numpyro
import numpy as np
import pandas as pd

from numpyro import distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

DATASET_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
dset = pd.read_csv(DATASET_URL, sep=";")

standardize = lambda x: (x - x.mean()) / x.std()

dset["AgeScaled"] = dset.MedianAgeMarriage.pipe(standardize)
dset["MarriageScaled"] = dset.Marriage.pipe(standardize)
dset["DivorceScaled"] = dset.Divorce.pipe(standardize)


def model(marriage=None, age=None, divorce=None):
    a = numpyro.sample("a", dist.Normal(0.0, 0.2))
    M, A = 0.0, 0.0
    if marriage is not None:
        bM = numpyro.sample("bM", dist.Normal(0.0, 0.5))
        M = bM * marriage
    if age is not None:
        bA = numpyro.sample("bA", dist.Normal(0.0, 0.5))
        A = bA * age
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = a + M + A
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=divorce)


# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

num_warmup, num_samples = 1000, 2000

# Run NUTS.
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(
    rng_key_, marriage=dset.MarriageScaled.values, divorce=dset.DivorceScaled.values
)
mcmc.print_summary()
```

<!--
### Saving our trained model

Now that we have _trained_ our model, the next step will be to save it so that it can be loaded afterwards at serving-time.
Note that, since this is a probabilistic model, we will only need to save the traces that approximate the posterior distribution over latent parameters.

This will get saved in a `numpyro-divorce.json` file.
-->

### トレーニングされたモデルの保存

モデルのトレーニングが完了したので、次のステップはそれを保存して、提供時に後でロードできるようにすることです。
これは確率モデルであるため、潜在パラメーターの事後分布を近似するトレースのみを保存する必要があります。

これは `numpyro-divorce.json` ファイルに保存されます。

```python
import json

samples = mcmc.get_samples()
serialisable = {}
for k, v in samples.items():
    serialisable[k] = np.asarray(v).tolist()

model_file_name = "numpyro-divorce.json"
with open(model_file_name, "w") as model_file:
    json.dump(serialisable, model_file)
```

<!--
## Serving

The next step will be to serve our model using `mlserver`.
For that, we will first implement an extension which serve as the _runtime_ to perform inference using our custom `numpyro` model.
-->

## サービング

次のステップは、`mlserver` を使用してモデルを提供することです。
そのために、まず、カスタムの `numpyro` モデルを使用して推論を行う _ランタイム_ として機能する拡張を実装します。


<!--
### Custom inference runtime

Our custom inference wrapper should be responsible of:

- Loading the model from the set samples we saved previously.
- Running inference using our model structure, and the posterior approximated from the samples.
-->

### カスタム推論ランタイム

カスタム推論ラッパーは以下の責任を持つべきです：

- 以前に保存したサンプルセットからモデルをロードする。
- モデル構造とサンプルから近似された事後分布を使用して推論を実行する。

```python
# %load models.py
import json
import numpyro
import numpy as np

from jax import random
from mlserver import MLModel
from mlserver.codecs import decode_args
from mlserver.utils import get_model_uri
from numpyro.infer import Predictive
from numpyro import distributions as dist
from typing import Optional


class NumpyroModel(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        with open(model_uri) as model_file:
            raw_samples = json.load(model_file)

        self._samples = {}
        for k, v in raw_samples.items():
            self._samples[k] = np.array(v)

        self._predictive = Predictive(self._model, self._samples)

        return True

    @decode_args
    async def predict(
        self,
        marriage: Optional[np.ndarray] = None,
        age: Optional[np.ndarray] = None,
        divorce: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        predictions = self._predictive(
            rng_key=random.PRNGKey(0), marriage=marriage, age=age, divorce=divorce
        )

        obs = predictions["obs"]
        obs_mean = obs.mean()

        return np.asarray(obs_mean)

    def _model(self, marriage=None, age=None, divorce=None):
        a = numpyro.sample("a", dist.Normal(0.0, 0.2))
        M, A = 0.0, 0.0
        if marriage is not None:
            bM = numpyro.sample("bM", dist.Normal(0.0, 0.5))
            M = bM * marriage
        if age is not None:
            bA = numpyro.sample("bA", dist.Normal(0.0, 0.5))
            A = bA * age
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        mu = a + M + A
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=divorce)

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
# %load settings.json
{
    "debug": "true"
}

```

#### `model-settings.json`

```python
# %load model-settings.json
{
    "name": "numpyro-divorce",
    "implementation": "models.NumpyroModel",
    "parameters": {
        "uri": "./numpyro-divorce.json"
    }
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
### Send test inference request

We now have our model being served by `mlserver`.
To make sure that everything is working as expected, let's send a request from our test set.

For that, we can use the Python types that `mlserver` provides out of box, or we can build our request manually.
-->

### テスト推論リクエストの送信

これで、モデルは `mlserver` によって提供されています。
すべてが期待通りに動作していることを確認するために、テストセットからリクエストを送信しましょう。

そのために、`mlserver` が提供する Python の型を使用することも、リクエストを手動で構築することもできます。

```python
import requests
import numpy as np

from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec

x_0 = np.array([28.0])
inference_request = InferenceRequest(
    inputs=[
        NumpyCodec.encode_input(name="marriage", payload=x_0)
    ]
)

endpoint = "http://localhost:8080/v2/models/numpyro-divorce/infer"
response = requests.post(endpoint, json=inference_request.dict())

response.json()
```

<!--
## Deployment

Now that we have written and tested our custom model, the next step is to deploy it.
With that goal in mind, the rough outline of steps will be to first build a custom image containing our code, and then deploy it.
-->

## デプロイメント

カスタムモデルの作成とテストが完了したので、次のステップはデプロイです。
その目的を念頭に置いて、大まかなステップはまずコードを含むカスタムイメージを構築し、それからデプロイすることです。

<!--
### Specifying requirements

MLServer will automatically find your requirements.txt file and install necessary python packages
-->

### 必要条件の指定

MLServer は自動的に `requirements.txt` ファイルを見つけて、必要な Python パッケージをインストールします。

```python
# %load requirements.txt
numpy==1.22.4
numpyro==0.8.0
jax==0.2.24
jaxlib==0.3.7

```

<!--
### Building a custom image

```{note}
This section expects that Docker is available and running in the background.
```

MLServer offers helpers to build a custom Docker image containing your code.
In this example, we will use the `mlserver build` subcommand to create an image, which we'll be able to deploy later.

Note that this section expects that Docker is available and running in the background, as well as a functional cluster with Seldon Core installed and some familiarity with `kubectl`.
-->

### カスタムイメージの構築

```{note}
このセクションでは、Docker が利用可能でバックグラウンドで実行されていることを前提としています。
```

MLServer は、コードを含むカスタム Docker イメージを構築するためのヘルパーを提供しています。
この例では、`mlserver build` サブコマンドを使用してイメージを作成します。このイメージは後でデプロイすることができます。

このセクションでは、Docker が利用可能でバックグラウンドで実行されていること、Seldon Core がインストールされた機能的なクラスターが存在し、`kubectl` に精通していることも前提としています。

```bash
%%bash
mlserver build . -t 'my-custom-numpyro-server:0.1.0'
```

<!--
To ensure that the image is fully functional, we can spin up a container and then send a test request. To start the container, you can run something along the following lines in a separate terminal:
-->

イメージが完全に機能していることを確認するために、コンテナを起動してからテストリクエストを送信できます。コンテナを起動するには、別のターミナルで次のようなコマンドを実行できます：


```bash
docker run -it --rm -p 8080:8080 my-custom-numpyro-server:0.1.0
```

```python
import numpy as np

from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec

x_0 = np.array([28.0])
inference_request = InferenceRequest(
    inputs=[
        NumpyCodec.encode_input(name="marriage", payload=x_0)
    ]
)

endpoint = "http://localhost:8080/v2/models/numpyro-divorce/infer"
response = requests.post(endpoint, json=inference_request.dict())

response.json()
```

<!--
As we should be able to see, the server running within our Docker image responds as expected.
-->

サーバーが期待通りに応答しているのが見えるはずです。


<!--
### Deploying our custom image

```{note}
This section expects access to a functional Kubernetes cluster with Seldon Core installed and some familiarity with `kubectl`.
```

Now that we've built a custom image and verified that it works as expected, we can move to the next step and deploy it.
There is a large number of tools out there to deploy images.
However, for our example, we will focus on deploying it to a cluster running [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/).

```{note}
Also consider that depending on your Kubernetes installation Seldon Core might expect to get the container image from a public container registry like [Docker hub](https://hub.docker.com/) or [Google Container Registry](https://cloud.google.com/container-registry). For that you need to do an extra step of pushing the container to the registry using `docker tag <image name> <container registry>/<image name>` and `docker push <container registry>/<image name>` and also updating the `image` section of the yaml file to `<container registry>/<image name>`.
```

For that, we will need to create a `SeldonDeployment` resource which instructs Seldon Core to deploy a model embedded within our custom image and compliant with the [V2 Inference Protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).
This can be achieved by _applying_ (i.e. `kubectl apply`) a `SeldonDeployment` manifest to the cluster, similar to the one below:
-->

### カスタムイメージのデプロイ


```{note}
このセクションでは、Seldon Core がインストールされた機能的な Kubernetes クラスターへのアクセスと、`kubectl` に関するある程度の知識が必要です。
```

カスタムイメージを構築し、期待通りに動作することを確認したので、次のステップに進みデプロイします。
イメージをデプロイするためのツールは数多くありますが、この例では、[Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/) を実行しているクラスターへのデプロイに焦点を当てます。

また、Kubernetes のインストールによっては、Seldon Core が [Docker Hub](https://hub.docker.com/) や [Google Container Registry](https://cloud.google.com/container-registry) のような公開コンテナレジストリからコンテナイメージを取得することを期待しているかもしれません。そのためには、`docker tag <イメージ名> <コンテナレジストリ>/<イメージ名>` と `docker push <コンテナレジストリ>/<イメージ名>` を使ってコンテナをレジストリにプッシュし、yamlファイルの `image` セクションを `<コンテナレジストリ>/<イメージ名>` に更新する追加のステップが必要です。

そのためには、カスタムイメージ内に埋め込まれたモデルをデプロイするよう Seldon Core に指示する `SeldonDeployment` リソースを作成する必要があります。これは、以下のような `SeldonDeployment` マニフェストをクラスターに _適用_ する（つまり `kubectl apply`）ことで実現できます：

```python
%%writefile seldondeployment.yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: numpyro-model
spec:
  protocol: v2
  predictors:
    - name: default
      graph:
        name: numpyro-divorce
        type: MODEL
      componentSpecs:
        - spec:
            containers:
              - name: numpyro-divorce
                image: my-custom-numpyro-server:0.1.0
```

```python

```
