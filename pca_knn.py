#! /usr/bin/python3

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

dataFilePath = "dataset/iris.data"
imgDir = "imgs"
randomState = 42

species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
variables = ["comprimento de sépala", "largura de sépala", "comprimento de pétala", "largura de pétala"]
colors = {"Iris-setosa": "#2a78d6", "Iris-versicolor": "#eb6834", "Iris-virginica": "#1baf7a"}
markers = {"Iris-setosa": "o", "Iris-versicolor": "s", "Iris-virginica": "^"}

os.makedirs(imgDir, exist_ok=True)

X, y = [], []
with open(dataFilePath) as dataset:
  for line in dataset:
    line = line.strip()
    if len(line.split(",")) != 5:
      continue
    *values, className = line.split(",")
    X.append([float(v) for v in values])
    y.append(className)
X, y = np.array(X), np.array(y)


def styleAxes(ax):
  ax.grid(True, color="#e5e5e5", linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)


def save(fig, name):
  outFile = os.path.join(imgDir, name)
  fig.tight_layout()
  fig.savefig(outFile, dpi=150)
  print("Gráfico salvo em", outFile)


# divisão 80% treino / 20% teste, mantendo a proporção de cada espécie
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, stratify=y, random_state=randomState
)
print("DIVISÃO DO DATASET")
print("#" * 80)
print("treino: {} registros | teste: {} registros".format(len(X_train), len(X_test)))
for s in species:
  print("\t{}: {} treino, {} teste".format(s, (y_train == s).sum(), (y_test == s).sum()))

# PCA ##########################################################################
# a padronização e o PCA são ajustados só com o treino, para o teste não "vazar"
scaler = StandardScaler().fit(X_train)
pca = PCA().fit(scaler.transform(X_train))

print("\nPCA")
print("#" * 80)
accumulated = 0
for i, ratio in enumerate(pca.explained_variance_ratio_):
  accumulated += ratio
  print("PC{}: {:6.2%} da variância (acumulado: {:6.2%})".format(i + 1, ratio, accumulated))

print("\npeso de cada variável nas componentes:")
print("{:<24}".format("") + "".join("{:>9}".format("PC" + str(i + 1)) for i in range(2)))
for j, v in enumerate(variables):
  print("{:<24}".format(v) + "".join("{:>9.3f}".format(pca.components_[i][j]) for i in range(2)))

# projeção de todos os registros nas duas primeiras componentes
fig, ax = plt.subplots(figsize=(7, 5))
for subset, Xs, ys in (("treino", X_train, y_train), ("teste", X_test, y_test)):
  projected = pca.transform(scaler.transform(Xs))
  for s in species:
    mask = ys == s
    ax.scatter(
      projected[mask, 0], projected[mask, 1],
      color=colors[s] if subset == "treino" else "white",
      edgecolors="white" if subset == "treino" else colors[s],
      marker=markers[s], s=40 if subset == "treino" else 55,
      linewidths=0.8 if subset == "treino" else 1.8,
      alpha=0.85 if subset == "treino" else 1,
      label="{} ({})".format(s, subset)
    )
ax.set_title("Projeção nas duas primeiras componentes principais")
ax.set_xlabel("PC1 ({:.1%} da variância)".format(pca.explained_variance_ratio_[0]))
ax.set_ylabel("PC2 ({:.1%} da variância)".format(pca.explained_variance_ratio_[1]))
styleAxes(ax)
ax.legend(frameon=False, fontsize=8)
save(fig, "pca.png")

# K-NN #########################################################################
# compara K-NN com as 4 variáveis originais e com as 2 primeiras componentes do PCA
models = {
  "4 variáveis originais": lambda k: Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k)),
  ]),
  "2 componentes (PCA)": lambda k: Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("knn", KNeighborsClassifier(n_neighbors=k)),
  ]),
}
modelColors = {"4 variáveis originais": "#2a78d6", "2 componentes (PCA)": "#eb6834"}
ks = list(range(1, 32, 2))

# o k é escolhido por validação cruzada (5 partes) dentro do treino;
# o conjunto de teste só é usado na avaliação final
print("\nK-NN")
print("#" * 80)
fig, ax = plt.subplots(figsize=(7, 5))
results = {}
for name, makeModel in models.items():
  cvScores = [cross_val_score(makeModel(k), X_train, y_train, cv=5).mean() for k in ks]
  bestK = ks[int(np.argmax(cvScores))]
  model = makeModel(bestK).fit(X_train, y_train)
  y_pred = model.predict(X_test)
  results[name] = (bestK, y_pred)

  ax.plot(ks, cvScores, color=modelColors[name], linewidth=2, marker="o", markersize=5, label=name)

  print("\n" + name.upper())
  print("melhor k na validação cruzada: {} (acurácia média {:.2%})".format(bestK, max(cvScores)))
  print("acurácia no teste: {:.2%}".format(accuracy_score(y_test, y_pred)))
  print("matriz de confusão (linhas = real, colunas = previsto):")
  cm = confusion_matrix(y_test, y_pred, labels=species)
  print("{:<18}".format("") + "".join("{:>17}".format(s) for s in species))
  for s, row in zip(species, cm):
    print("{:<18}".format(s) + "".join("{:>17}".format(n) for n in row))
  print(classification_report(y_test, y_pred, labels=species, digits=3, zero_division=0))

ax.set_title("Acurácia do K-NN (validação cruzada no treino) por k")
ax.set_xlabel("k (número de vizinhos)")
ax.set_ylabel("Acurácia média")
ax.set_xticks(ks)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "{:.0%}".format(v)))
styleAxes(ax)
ax.legend(frameon=False)
save(fig, "knn_k.png")

# matrizes de confusão no conjunto de teste
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
shortNames = [s.replace("Iris-", "") for s in species]
for ax, (name, (bestK, y_pred)) in zip(axes, results.items()):
  cm = confusion_matrix(y_test, y_pred, labels=species)
  ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
  for i in range(len(species)):
    for j in range(len(species)):
      ax.text(j, i, cm[i, j], ha="center", va="center",
              color="white" if cm[i, j] > cm.max() / 2 else "#1a1a1a", fontsize=12)
  ax.set_xticks(range(len(species)), shortNames)
  ax.set_yticks(range(len(species)), shortNames)
  ax.set_xlabel("Previsto")
  ax.set_ylabel("Real")
  ax.set_title("{} (k = {})\nacurácia no teste: {:.1%}".format(
    name, bestK, accuracy_score(y_test, y_pred)))
save(fig, "knn_confusao.png")

plt.show()
