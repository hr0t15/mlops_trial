<!--
# Serving Scikit-Learn models

Out of the box, `mlserver` supports the deployment and serving of `scikit-learn` models.
By default, it will assume that these models have been [serialised using `joblib`](https://scikit-learn.org/stable/modules/model_persistence.html).

In this example, we will cover how we can train and serialise a simple model, to then serve it using `mlserver`.
-->

# Scikit-Learn モデルの提供

`mlserver` は、`scikit-learn` モデルのデプロイと提供をサポートしています。デフォルトでは、これらのモデルが [`joblib` を使用してシリアライズされたものと仮定します](https://scikit-learn.org/stable/modules/model_persistence.html)。

この例では、シンプルなモデルをトレーニングしてシリアライズし、その後 `mlserver` を使用して提供する方法について説明します。

<!--
## Training

The first step will be to train a simple `scikit-learn` model.
For that, we will use the [MNIST example from the `scikit-learn` documentation](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) which trains an SVM model.
-->

## トレーニング

最初のステップは、シンプルな `scikit-learn` モデルをトレーニングすることです。
そのためには、[SVMモデルをトレーニングする `scikit-learn` ドキュメントのMNIST例](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)を使用します。


```python
# Original source code and more details can be found in:
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# The digits dataset
digits = datasets.load_digits()

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)
```

<!--
### Saving our trained model

To save our trained model, we will serialise it using `joblib`.
While this is not a perfect approach, it's currently the recommended method to persist models to disk in the [`scikit-learn` documentation](https://scikit-learn.org/stable/modules/model_persistence.html).

Our model will be persisted as a file named `mnist-svm.joblib`
-->
### トレーニングしたモデルの保存

トレーニングしたモデルを保存するために、`joblib`を使用してシリアライズします。
これは完璧な方法ではありませんが、現在は[`scikit-learn`ドキュメント](https://scikit-learn.org/stable/modules/model_persistence.html)で推奨されているディスクへのモデル永続化方法です。

モデルは`mnist-svm.joblib`というファイル名で保存されます。


```python
import joblib

model_file_name = "mnist-svm.joblib"
joblib.dump(classifier, model_file_name)
```

<!--
## Serving

Now that we have trained and saved our model, the next step will be to serve it using `mlserver`. 
For that, we will need to create 2 configuration files: 

- `settings.json`: holds the configuration of our server (e.g. ports, log level, etc.).
- `model-settings.json`: holds the configuration of our model (e.g. input type, runtime to use, etc.).
-->

## サービング

モデルのトレーニングと保存が完了したので、次のステップは `mlserver` を使用してモデルを提供することです。
そのためには、2つの設定ファイルを作成する必要があります：

- `settings.json`: サーバーの設定（例：ポート、ログレベルなど）を保持します。
- `model-settings.json`: モデルの設定（例：入力タイプ、使用するランタイムなど）を保持します。

### `settings.json`


```python
%%writefile settings.json
{
    "debug": "true"
}
```

### `model-settings.json`


```python
%%writefile model-settings.json
{
    "name": "mnist-svm",
    "implementation": "mlserver_sklearn.SKLearnModel",
    "parameters": {
        "uri": "./mnist-svm.joblib",
        "version": "v0.1.0"
    }
}
```

<!--
### Start serving our model

Now that we have our config in-place, we can start the server by running `mlserver start .`. This needs to either be ran from the same directory where our config files are or pointing to the folder where they are.
-->
### モデルの提供を開始

設定が整ったので、`mlserver start .` を実行してサーバーを起動できます。これは、設定ファイルがある同じディレクトリから実行するか、それらが存在するフォルダを指定して実行する必要があります。

```shell
mlserver start .
```

<!--
Since this command will start the server and block the terminal, waiting for requests, this will need to be ran in the background on a separate terminal.

### Send test inference request

We now have our model being served by `mlserver`.
To make sure that everything is working as expected, let's send a request from our test set.

For that, we can use the Python types that `mlserver` provides out of box, or we can build our request manually.
-->
このコマンドはサーバーを起動し、リクエストを待ってターミナルをブロックするため、別のターミナルでバックグラウンドで実行する必要があります。

### テスト推論リクエストを送信

現在、私たちのモデルは `mlserver` によって提供されています。
すべてが期待通りに動作していることを確認するために、テストセットからリクエストを送信しましょう。

そのためには、`mlserver` が提供するPythonの型を使用することもできますし、手動でリクエストを構築することもできます。


```python
import requests

x_0 = X_test[0:1]
inference_request = {
    "inputs": [
        {
          "name": "predict",
          "shape": x_0.shape,
          "datatype": "FP32",
          "data": x_0.tolist()
        }
    ]
}

endpoint = "http://localhost:8080/v2/models/mnist-svm/versions/v0.1.0/infer"
response = requests.post(endpoint, json=inference_request)

response.json()
```

<!--
As we can see above, the model predicted the input as the number `8`, which matches what's on the test set.
-->
上記からわかるように、モデルは入力を数字の「8」として予測しました。これはテストセットに一致しています。


```python
y_test[0]
```


```python

```
