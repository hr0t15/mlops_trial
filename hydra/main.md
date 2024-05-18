# タスクとパラメータの一元管理で実現するMLOps

## はじめに

enechainでは市場活性化を目的として、機械学習や最適化アルゴリズムを用いて電力や燃料などの商品に関する指標を算出し、社内外に提供しています。本稿では、これらを算出するモデルの構築・運用を効率化するために作成した、タスクランナーinvokeとパラメータ管理ツールhydraを一体化したシステムを紹介します。

## 背景

### タスクランナーを導入するモチベーション

機械学習モデルの構築・運用において、データサイエンティストがモデル開発を、MLエンジニアがパイプライン構築や処理効率化を担当するなど、複数の開発者が適切に役割分担して並列で作業を進めることで、高速な価値提供が期待できます。しかし、開発者それぞれの間で何の取り決めもなく並列で開発を進めてしまうと、お互いの成果物の受け渡しが煩雑になったり、環境の違いによる想定外の不具合が発生したりと、円滑な連携が難しい場合があります。

これらの問題を解消するため、MLパイプラインのステップ単位での「タスク」としてインターフェイスを予め定義しておきます。タスクにはモデルのロジックは記載せずに、処理の並列化や依存関係、パイプラインエンジン特有の定義などを記載します。データサイエンティストとMLエンジニアはそれぞれの開発に集中しながら、異なる環境（ローカル・クラウド）でも 「タスク」を叩くだけでMLパイプラインを実行できることが望ましいです。

### パラメータ管理ツールを導入するモチベーション

モデルのハイパーパラメータや前処理・評価指標の設定など、実験や運用には多くのパラメータ管理が伴います。単純なテキストファイルや環境変数でもある程度は管理できますが、パラメータファイルの階層化・分割や、他パラメータの再利用、環境ごとの値の切り替えといった高度な機能を持ったツールを導入することで、効率的かつ見通し良くパラメータを管理することができます。

## 実現したいこと

本稿では、タスクランナーとパラメータ管理ツールを統合することで、次のような機能の実現を目指します。

### モデルや環境に依存しないタスクによるパイプラインの操作

モデルの「学習」や「推論」などのタスクをCLIから呼び出すことでパイプラインを操作します。モデルや環境の名前、実行時にしか与えられないパラメータ（対象日次など）はコマンドの引数として与えますが、それ以外のモデル・環境別の差異は全てコンフィグファイルに吸収させます。

### 共通部分と環境特有部分を分離したパラメータ定義

localやdevelop, productionなどの複数の環境でモデルを動作させる場合、パラメータの大部分は共通である一方で、環境ごとに異なるパラメータも一部存在します。同じようなパラメータを環境ごとに繰り返し定義するのは冗長であり、またパラメータの変更漏れや環境間の差異が生じるリスクがあります。そこで、ベースのパラメータは1つとし、環境特有の部分は差分として定義することで、パラメータの再利用性を高めます。この考え方はKustomizeのpatchと似たものです。

### パラメータ定義の構造化

パラメータファイルにはモデルのハイパーパラメータや前処理・評価の設定など、多くのパラメータを定義する必要があります。これらのパラメータを効率的に管理するため、パラメータをトピック単位でグループ化し、ディレクトリやYAMLのマッピング形式でネスト構造を作ります。この階層構造により、パラメータ群を論理的に分割でき、可読性と保守性が向上します。

さらに、ある箇所で定義されたパラメータ値を、別の箇所から参照できる機能を活用します。これにより定義の冗長性を排除でき、一か所の変更で関連パラメータを一括で更新できます。

一方で、コード内からはそれらがマージされた単一のコンフィグオブジェクトにアクセスします。パラメータファイルの分割や参照元をコード側で意識することなく、実装をシンプルに保ちます。

## 実装方法

### 利用するツール

タスクランナーにはinvokeを採用しました。invokeはPython製のタスクランナーであり、Pythonの関数をタスクとして登録し、CLIから呼び出すことができます。似たような手段としてsetupによる自作cliやdoit、clickなどが挙げられますが、invokeは拡張機能が豊富で、subcommandや依存関係の定義、yamlでの設定ファイルの記述ができる、などのメリットがあります。

invokeにもパラメータ管理の機能がありますが、パラメータの構造化や参照機能が不足しているため、パラメータ管理にはhydra（とそのバックエンドであるomegaconf）を採用しました。hydraは機械学習モデルのハイパーパラメータ管理を目的としたツールですが、パラメータの構造化や参照機能が強力であり、環境ごとのパラメータの差分管理や、パラメータの再利用性を高めることができます。

invokeとhydraのそれぞれがタスクランナーとパラメータ管理の機能の両方を持っていますが、それぞれ強みがあり、それらを統合して良いとこどりをすることで、効率的なタスクランナーとパラメータ管理システムを構築することができました。

### パラメータファイル

パラメータファイルはyaml形式で、次のようなディレクトリ構造で定義します。

```
.
├── configs
│  ├── pipelines
│  │  ├── pipeline_1.yaml
│  │  └── pipeline_2.yaml
│  ├── pipelines.yaml
│  ├── prod.yaml
│  └── sandbox.yaml
└── invoke.yaml
```

invoke.yamlには全モデル・環境のベースとなるパラメータを定義します。gcp project名やMLflow Trackingのエンドポイントは環境ごとに切り替えられるように定義しておきます。

```python
# invoke.yaml
defaults:
- _self_
- configs@pipelines: pipelines

version: ${rc.pyproject_value:tool.poetry.version} # poetry package version

env: dev
user: ${oc.env:USER, runner}

gcp:
  project: xxx-${env}

train:
  mlflow:
    use: True
    tracking_uri: https://mlflow.${env}.xxx.com
  output_dir: trained_models
```

configs/ディレクトリ下には、モデル（パイプライン）特有のパラメータと環境特有のパラメータを定義します。

```yaml
# configs/prod.yaml
env: prod
```

prod.yamlではinvoke.yamlで定義したパラメータの${env}の部分をprodに上書きするだけです。sandbox環境（開発者個人ごとに用意した実験用環境）のgcp projectは${env}の置き換えだけでは対応できないため、あらためて定義しています。

```yaml
# configs/sandbox.yaml
gcp:
  project: xxx-sandbox-${user}

train:
  mlflow:
    use: False
```

パイプラインごとのパラメータはconfigs/pipelines/ディレクトリ下に定義します。それぞれのモデルに必要なパラメータを定義する想定ですが、空でも大丈夫です。

```yaml
# configs/pipelines.yaml
defaults:
- pipelines@pipeline_1: pipeline_1
- pipelines@pipeline_2: pipeline_2
```

```yaml
# configs/pipelines/pipeline1.yaml
param1: xxx
```


```yaml
# configs/pipelines/pipeline2.yaml

```

これらのパラメータファイルを用意したところで、configを表示する簡単なタスクをinvokeで定義してみます。

```python
# tasks.py
from collections.abc import Iterable

import invoke
import yaml
from invoke.config import DataProxy


def config_to_dict(config: DataProxy, keys: Iterable[str]) -> dict:
    """invokeのconfigをdictに変換する"""
    return {
        key: config_to_dict(val, val.keys())
        if isinstance((val := config.get(key)), DataProxy)
        else val
        for key in keys
    }


@invoke.task
def print_config(c: invoke.Context) -> None:
    print(
        yaml.dump(
            config_to_dict(
                c.config,
                ("version", "env", "user", "gcp", "train", "pipelines"),
            )
        )
    )
```

このタスクを実行してみると次のような結果が得られます。${env}が残っていたり、pipelinesがnullになっていたりと、invokeだけの機能では他パラメータ値の参照や、複数ファイルのマージができないことがわかります。

```
$ inv print-config
env: dev
gcp:
  project: xxx-${env}
pipelines: null
train:
  mlflow:
    tracking_uri: https://mlflow.${env}.xxx.com
    use: true
  output_dir: trained_models
user: ${oc.env:USER, runner}
version: ${rc.pyproject_value:tool.poetry.version}

$ inv -f configs/sandbox.yaml print-config 
env: dev
gcp:
  project: xxx-sandbox-${user}
pipelines: null
train:
  mlflow:
    tracking_uri: https://mlflow.${env}.xxx.com
    use: false
  output_dir: trained_models
user: ${oc.env:USER, runner}
version: ${rc.pyproject_value:tool.poetry.version}
```


### 構造化パラメータのマージ処理の実装

最初に、pyproject.tomlに記載された任意の値を取得する処理をomegaconfのresolverとしてtasks.pyに定義します。実装の詳細は割愛しますが、これによってパラメータファイル内の記述からomegaconfがpyproject.tomlの値を自動的に解決し、参照することができます。

```python
import os
from typing import Any

import toml
from omegaconf import OmegaConf


def get_pyproject_value(
    project_dir: str, args: str | None = None
) -> Any | None:
    """pyprojectで管理している設定値を返す

    Args:
        project_dir: Project Directory Path
        args: Pyproject args string (ex. tool.poetry.version)

    Returns:
        pyproject value
    """
    pyproject_path = f"{project_dir}/pyproject.toml"
    if not args:
        return None
    if not os.path.exists(pyproject_path):
        return None

    with open(pyproject_path) as f:
        pyproject_toml = toml.load(f)
        value = pyproject_toml
        try:
            for arg in args.split("."):
                value = value[arg]
            return value
        except KeyError:
            return None


def _register_omegaconf_resolver(project_dir: str) -> None:
    """OmegaConf Custom Resolverを登録

    Args:
        project_dir: Project Directory Path
    """

    def _get_pyproject_value(args: str | None = None) -> str | None:
        """pyprojectで管理している設定値を返す

        Args:
            args: Pyproject args (ex. tool.poetry.version)

        Returns:
            pyproject value
        """
        return get_pyproject_value(project_dir, args)

    # Register OmegaConf
    OmegaConf.register_new_resolver(
        "rc.pyproject_value", _get_pyproject_value, replace=True
    )
```

次に、invokeのconfigをhydraで読み込んだconfigで上書きする処理をtasks.pyに追加します。一度hydra (omegaconf) でyamlを読み込んでresolverや階層化コンフィグのマージ処理を行ってから、その結果をinvokeのconfigにマージします。

```python
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def _merge_config(
    invoke_config: invoke.Config, hydra_config: DictConfig
) -> None:
    """Invoke configをHydra configの内容で上書きする

    Args:
        invoke_config: Invoke config
        hydra_config: Hydra config
    """
    for key, val in hydra_config.items():
        if isinstance(val, DictConfig):
            if key in invoke_config:
                _merge_config(
                    invoke_config[key],
                    OmegaConf.to_container(val, resolve=True),
                )
            else:
                invoke_config.update(
                    {key: OmegaConf.to_container(val, resolve=True)}
                )
        else:
            invoke_config.update({key: val})


@invoke.task
def update_config(c: invoke.Context) -> None:
    """c.configの中身をHydraで読み込んだもので上書きするtask

    Args:
        c: invoke.Context
    """
    project_dir = getattr(c.config, "_project_prefix")
    runtime_path = getattr(c.config, "_runtime_path")

    _register_omegaconf_resolver(project_dir)

    if os.path.exists(os.path.join(project_dir, "invoke.yaml")):
        # projectのrootにinvoke.yamlがある場合
        with initialize_config_dir(config_dir=project_dir, version_base=None):
            hydra_config = compose("invoke")
    else:
        hydra_config = None

    if runtime_path is not None:
        # -fオプションで異なるyamlが指定された場合はinvoke.yamlの内容を上書きする
        runtime_dir = os.path.dirname(runtime_path)
        runtime_name = os.path.basename(runtime_path)
        with initialize_config_dir(
            config_dir=os.path.join(project_dir, runtime_dir), version_base=None
        ):
            override_config = compose(runtime_name)
        if hydra_config is not None:
            hydra_config = OmegaConf.merge(hydra_config, override_config)
        else:
            hydra_config = override_config

    if hydra_config is not None:
        _merge_config(c.config, hydra_config)
```

最後に、update_configタスクをprint_configタスクの前に実行するようにinvokeのタスク定義を変更すれば完成です。

```
@invoke.task(pre=[update_config])
def print_config(c: invoke.Context) -> None:
    print(
        yaml.dump(
            config_to_dict(
                c.config,
                ("version", "env", "user", "gcp", "train", "pipelines"),
            )
        )
    )
$ inv print-config
env: dev
gcp:
  project: xxx-dev
pipelines:
  pipeline_1:
    param1: xxx
  pipeline_2: {}
train:
  mlflow:
    tracking_uri: https://mlflow.dev.xxx.com
    use: true
  output_dir: trained_models
user: fujimura
version: 0.1.0

$ inv -f configs/prod.yaml print-config 
env: prod
gcp:
  project: xxx-prod
pipelines:
  pipeline_1:
    param1: xxx
  pipeline_2: {}
train:
  mlflow:
    tracking_uri: https://mlflow.prod.xxx.com
    use: true
  output_dir: trained_models
user: fujimura
version: 0.1.0

$ inv -f configs/sandbox.yaml print-config  
env: dev
gcp:
  project: xxx-sandbox-fujimura
pipelines:
  pipeline_1:
    param1: xxx
  pipeline_2: {}
train:
  mlflow:
    tracking_uri: https://mlflow.dev.xxx.com
    use: false
  output_dir: trained_models
user: fujimura
version: 0.1.0
```

実際の動作としてはこれで十分ですが、taskを定義する際に@invoke.task(pre=[update_config])を毎回書くのは面倒かもしれません。その場合は、invoke.taskデコレータをカスタマイズして、デフォルトでupdate_configを前提処理として実行するようにすることができます。

```python
def task(*args, **kwargs) -> Callable[[...], Callable]:
    """Hydraを利用するためのカスタムinvoke task

    @taskを用いたときにdefaultでupdate_configを事前に実行する
    """
    pre = kwargs.pop("pre", [])
    pre.append(update_config)
    return invoke.task(*args, pre=pre, **kwargs)

@task
def print_config(c: invoke.Context) -> None:
    print(
        yaml.dump(
            config_to_dict(
                c.config,
                ("version", "env", "user", "gcp", "train", "pipelines"),
            )
        )
    )
```


## おわりに

本記事では、タスクランナーとパラメータ管理を統合することで、モデルの構築・運用を効率化する方法を紹介しました。タスクランナーにはinvoke、パラメータ管理にはhydraを採用し、それぞれの強みを生かしてシステムを構築しました。これにより、モデルや環境に依存しないタスクによるパイプラインの操作、共通部分と環境特有部分を分離したパラメータ定義、パラメータ定義の構造化といった機能を実現しました。