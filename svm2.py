# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Charger les données
data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

# Explorer et préparer les données
# a. Diviser les colonnes de l'ensemble de données en attributs numériques et catégoriques.
numerical_features = data.select_dtypes(include=[np.number])
categorical_features = data.select_dtypes(include=[np.object])
# b. Supprimer les colonnes ID et ZIP.
data = data.drop(['ID', 'ZIP Code'], axis=1)

# c. Analyse mono variable
# i. Créer une fonction qui affiche la densité des variables numériques.
def plot_numerical_density(feature):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Density Plot of {feature}')
    plt.show()

# ii. Créer une fonction qui retourne un Pie Chart et Bar graph diagrammes pour afficher les variables catégoriques.
def plot_categorical_distribution(feature):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    data[feature].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f'Pie Chart of {feature}')
    
    plt.subplot(1, 2, 2)
    data[feature].value_counts().plot(kind='bar')
    plt.title(f'Bar Graph of {feature}')
    
    plt.show()

# iii. Afficher la distribution de la variable Target (Personal loan)
plot_categorical_distribution('Personal Loan')

# d. Analyse multi variables : donner la matrice de corrélation (heatmap)
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# 2. Déclarer les variables caractéristiques X et la variable cible Y.
X = data.drop('Personal Loan', axis=1)
y = data['Personal Loan']

# 3. Diviser le dataset en données d’apprentissage et données de test, en conservant 30% du jeu de données pour l'évaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Entrainer les données avec une machine à vecteurs de support (SVM) linéaire.
svm_linear_model = SVC(kernel='linear')
svm_linear_model.fit(X_train, y_train)

# 5. Prédire avec les données de test.
y_pred = svm_linear_model.predict(X_test)

# 6. Afficher la matrice de confusion « Confusion Matrix ».
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 7. Afficher le rapport de classification « Classification report ».
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# 8. Construire le modèle SVM en changeant la fonction noyau (kernel function) à “radial” et vérifier si les valeurs des “Accuracy” sont meilleures.
svm_radial_model = SVC(kernel='rbf')
svm_radial_model.fit(X_train, y_train)
y_pred_radial = svm_radial_model.predict(X_test)

accuracy_radial = accuracy_score(y_test, y_pred_radial)
print(f"Accuracy with radial kernel: {accuracy_radial}")
