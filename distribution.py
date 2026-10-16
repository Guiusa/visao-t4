#! /usr/bin/python3

import matplotlib.pyplot as plt

dataFilePath = "dataset/iris.data"

species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
# cor e marcador distintos por espécie (o marcador garante leitura sem depender só da cor)
colors = {"Iris-setosa": "#2a78d6", "Iris-versicolor": "#eb6834", "Iris-virginica": "#1baf7a"}
markers = {"Iris-setosa": "o", "Iris-versicolor": "s", "Iris-virginica": "^"}

agregated = {
  s: {
    "comprimento de sépala": [],
    "largura de sépala": [],
    "comprimento de pétala": [],
    "largura de pétala": []
  } for s in species
}

with open(dataFilePath) as dataset:
  for line in dataset:
    line = line.strip()
    if len(line.split(",")) != 5:
      continue
    sepal_l, sepal_w, petal_l, petal_w, className = line.split(",")
    agregated[className]["comprimento de sépala"].append(float(sepal_l))
    agregated[className]["largura de sépala"].append(float(sepal_w))
    agregated[className]["comprimento de pétala"].append(float(petal_l))
    agregated[className]["largura de pétala"].append(float(petal_w))

# calcula os valores mínimo e máximo de cada variável considerando todas as espécies
minMax = {}
for v in agregated[species[0]]:
  values = [x for s in species for x in agregated[s][v]]
  minMax[v] = {"min": min(values), "max": max(values)}


def plot(part, lengthKey, widthKey, outFile):
  fig, ax = plt.subplots(figsize=(7, 5))
  for s in species:
    ax.scatter(
      agregated[s][lengthKey], agregated[s][widthKey],
      color=colors[s], marker=markers[s], s=40, alpha=0.85,
      edgecolors="white", linewidths=0.8, label=s
    )

  # escala baseada nos valores mínimo e máximo, com uma pequena margem
  lmin, lmax = minMax[lengthKey]["min"], minMax[lengthKey]["max"]
  wmin, wmax = minMax[widthKey]["min"], minMax[widthKey]["max"]
  lpad, wpad = (lmax - lmin) * 0.05, (wmax - wmin) * 0.05
  ax.set_xlim(lmin - lpad, lmax + lpad)
  ax.set_ylim(wmin - wpad, wmax + wpad)

  ax.set_title("Comprimento x largura de " + part)
  ax.set_xlabel("Comprimento de " + part + " (cm)")
  ax.set_ylabel("Largura de " + part + " (cm)")
  ax.grid(True, color="#e5e5e5", linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.legend(title="Espécie", frameon=False)
  fig.tight_layout()
  fig.savefig(outFile, dpi=150)
  print("Gráfico salvo em", outFile)


def plotRatio(part, lengthKey, widthKey, outFile, bins=20):
  ratios = {
    s: [l / w for l, w in zip(agregated[s][lengthKey], agregated[s][widthKey])]
    for s in species
  }

  # mesmos intervalos (bins) para todas as espécies, para que sejam comparáveis
  allRatios = [r for s in species for r in ratios[s]]
  rmin, rmax = min(allRatios), max(allRatios)
  step = (rmax - rmin) / bins
  edges = [rmin + i * step for i in range(bins + 1)]

  fig, ax = plt.subplots(figsize=(7, 5))
  for s in species:
    ax.hist(ratios[s], bins=edges, color=colors[s], alpha=0.25)
    ax.hist(ratios[s], bins=edges, color=colors[s], histtype="step", linewidth=2, label=s)

  ax.set_title("Distribuição da relação comprimento/largura de " + part)
  ax.set_xlabel("Comprimento / largura de " + part)
  ax.set_ylabel("Número de registros")
  ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.legend(title="Espécie", frameon=False)
  fig.tight_layout()
  fig.savefig(outFile, dpi=150)
  print("Gráfico salvo em", outFile)


plot("sépala", "comprimento de sépala", "largura de sépala", "imgs/sepalas.png")
plot("pétala", "comprimento de pétala", "largura de pétala", "imgs/petalas.png")
plotRatio("sépala", "comprimento de sépala", "largura de sépala", "imgs/relacao_sepalas.png")
plotRatio("pétala", "comprimento de pétala", "largura de pétala", "imgs/relacao_petalas.png")

plt.show()
