import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
X = iris.data
sc = StandardScaler()
sc.fit(X)

distorsions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve (Iris Dataset)')
plt.show()
