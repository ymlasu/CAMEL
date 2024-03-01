
import matplotlib.pyplot as plt
import time
from camel import CAMEL
from sklearn import datasets


t1=time.time()

X, y = datasets.make_swiss_roll(n_samples=10000, random_state=None)

reducer= CAMEL(target_type='numerical') #as labels are numerical values

X_embedding = reducer.fit_transform(X, y)


print(time.time()-t1)
y = y.astype(int)


#y_transformed = y[1:y.shape[0]]

# Visualization

plt.figure(1)
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='jet', s=0.2)
plt.title('CAMEL Embedding')
plt.tight_layout()
plt.show()

