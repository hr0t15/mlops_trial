

## はじめに

[Seldon Core Documentation - Install Locally](https://docs.seldon.io/projects/seldon-core/en/latest/install/kind.html)に従い、以下のコンポーネントのインストール手順を示す。

* Docker
* Kind
* Kubectl
* Helm


## Docker

まずはDockerを導入するために必要となるパッケージをインストールする。

```bash:terminal
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
```

公式DockerリポジトリのGPGキーをシステムに追加し、DockerリポジトリをAPTソースに追加する。

```bash:terminal
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
```

Dockerのインストールの参照が、デフォルトのUbuntuリポジトリではなく、Dockerリポジトリとなっていることを確認する。

```bash:terminal
apt-cache policy docker-ce
```

例えば、以下の出力となっていればよい。

```
docker-ce:
  Installed: (none)
  Candidate: 5:19.03.9~3-0~ubuntu-focal
  Version table:
     5:19.03.9~3-0~ubuntu-focal 500
        500 https://download.docker.com/linux/ubuntu focal/stable amd64 Packages
```

Dockerをインストールする。

```bash:terminal
sudo apt install -y docker-ce
```

Dockerがインストールされたことを確認する。

```bash:terminal
sudo docker version
```

Dockerはデフォルトではsudoにより実行する必要がある。  
操作ユーザに関しては、sudoなしでDocker実行できるようにするために、dockerグループに追加する。

```bash:terminal
sudo gpasswd -a $(whoami) docker
getent group docker
```

先ほどはsudoをつけてバージョン確認を行ったが、今度はsudoなしでもバージョン確認ができることを確認する。

```bash:terminal
docker version
```

Dockerのサービスの有効化を行う。

```bash:terminal
sudo systemctl enable docker
```

## kind

[Quick Start - Installing From Release Binaries](https://kind.sigs.k8s.io/docs/user/quick-start/#installing-from-release-binaries)より。


```bash
# For AMD64 / x86_64
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.22.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

バージョン確認で実行できることを確認する。

```
kind --version
```

## Kubectl

[https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)を参考に。

Install kubectl binary with curl on Linux
Download the latest release with the command:

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
```

Install kubectl

```bash
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

インストールの確認

```bash
kubectl version
```


## Helm


[Installing Helm - From Apt (Debian/Ubuntu)](https://helm.sh/docs/intro/install/#from-apt-debianubuntu)を参考に。

```bash
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

インストールの確認

```bash
helm version
```

