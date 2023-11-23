# Importer les bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions  # Assurez-vous d'installer mlxtend (pip install mlxtend)

# Charger, explorer et préparer les données
iris_data = pd.read_csv('Iris.csv')
X = iris_data.iloc[:, [1, 2]].values  # Utiliser seulement les deux premiers attributs (longueur et largeur des sépales)
y = iris_data.iloc[:, -1].values

# Diviser le dataset en données d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Entraîner les données avec une SVM linéaire
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Calculer le score d'échantillons bien classifiés sur le jeu de données de test
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Visualiser la surface de décision
plt.figure(figsize=(8, 6))
plot_decision_regions(X_test, y_test, clf=svm_model, legend=2)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Decision Regions - SVM Linear')
plt.show()

# Pour répondre à la question sur l'adaptation du modèle au problème,
# observez la visualisation de la surface de décision. Si elle semble bien
# séparer les classes, le modèle est adapté. Sinon, vous pourriez essayer
# d'autres noyaux (non linéaires) ou ajuster les paramètres du modèle.
