
import matplotlib.pyplot as plt
import time
from camel import CAMEL
from sklearn import datasets


t1=time.time()

X, y = datasets.make_swiss_roll(n_samples=10000, random_state=None)
Xnew, ynew = datasets.make_swiss_roll(n_samples=5000, random_state=None)

reducer= CAMEL()

X_embedding = reducer.fit_transform(X)

X_transformed = reducer.transform(Xnew)
Xnew, ynew = datasets.make_swiss_roll(n_samples=3000, random_state=None)
X_transformed = reducer.transform(Xnew)
print(time.time()-t1)
y = y.astype(int)
ynew=ynew.astype(int)

#y_transformed = y[1:y.shape[0]]

# Visualization

plt.figure(1)
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='jet', s=0.2)
plt.title('CAMEL Embedding of training data')

plt.tight_layout()
plt.show()


plt.figure(2)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=ynew, cmap='jet', s=0.2)
plt.title('CAMEL Embedding of new data')

plt.tight_layout()
plt.show()