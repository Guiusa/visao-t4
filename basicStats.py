#! /usr/bin/python3

from statistics import mean, stdev, mode

dataFilePath="dataset/iris.data"
dataset = open(dataFilePath)

agregated = {
  "Iris-setosa": {
    "total_registers": 0,
    "comprimento de sépala": [],
    "largura de sépala": [],
    "comprimento de pétala": [],
    "largura de pétala": []  
  },
  "Iris-versicolor": {
    "total_registers": 0,
    "comprimento de sépala": [],
    "largura de sépala": [],
    "comprimento de pétala": [],
    "largura de pétala": []
},
  "Iris-virginica": {
    "total_registers": 0,
    "comprimento de sépala": [],
    "largura de sépala": [],
    "comprimento de pétala": [],
    "largura de pétala": []
  },
  "Total": {
    "total_registers": 0,
    "comprimento de sépala": [],
    "largura de sépala": [],
    "comprimento de pétala": [],
    "largura de pétala": []
  },
}
while line := dataset.readline():
  if len(line.split(",")) == 1:
    break
  line = line.split("\n")[0]
  sepal_l, sepal_w, petal_l, petal_w, className = line.split(",")
  agregated[className]["total_registers"] += 1
  agregated[className]["comprimento de sépala"].append(float(sepal_l))
  agregated[className]["largura de sépala"].append(float(sepal_w))
  agregated[className]["comprimento de pétala"].append(float(petal_l))
  agregated[className]["largura de pétala"].append(float(petal_w))
  agregated["Total"]["total_registers"] += 1
  agregated["Total"]["comprimento de sépala"].append(float(sepal_l))
  agregated["Total"]["largura de sépala"].append(float(sepal_w))
  agregated["Total"]["comprimento de pétala"].append(float(petal_l))
  agregated["Total"]["largura de pétala"].append(float(petal_w))

print("ESTATÍSTICA DAS VARIÁVEIS")
print("#" * 80, "\n")

for e in "Iris-setosa,Iris-versicolor,Iris-virginica,Total".split(","):
  if e != "Total":
    print("ESPÉCIE: ", end="")
  print(e)
  print("Número de registros:", agregated[e]["total_registers"])
  for v in "comprimento de sépala,largura de sépala,comprimento de pétala,largura de pétala".split(","):
    print(v+":")
    print("\tmédia: {:.2f}cm".format(mean(agregated[e][v])))
    print("\tdesvio: {:.2f}cm".format(stdev(agregated[e][v])))
    print("\tmoda: {:.2f}cm".format(mode(agregated[e][v])))
  print("relação média comprimento/largura de sépala: {:.5f}".format(mean(agregated[e]["comprimento de sépala"]) / mean(agregated[e]["largura de sépala"])))
  print("relação média comprimento/largura de pétala: {:.5f}".format(mean(agregated[e]["comprimento de pétala"]) / mean(agregated[e]["largura de pétala"])))
  print("-" * 80)

print("\n" + "#" *80)
