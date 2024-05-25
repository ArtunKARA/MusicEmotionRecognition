import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Özellikleri ve etiketleri yükleyin
X = np.load('features.npy')
y = np.load('labels.npy')

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cross-validation ayarları
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Neural Network modeli eğitme ve cross-validation
nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_cv_scores = cross_val_score(nn_clf, X, y, cv=cv, scoring='accuracy')
nn_clf.fit(X_train, y_train)

# Random Forest modeli eğitme ve cross-validation
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='accuracy')
rf_clf.fit(X_train, y_train)

# Modeli test verisi üzerinde değerlendirme
nn_y_pred = nn_clf.predict(X_test)
rf_y_pred = rf_clf.predict(X_test)

# Performans ölçümleri
print("Neural Network Model Accuracy:", accuracy_score(y_test, nn_y_pred))
print("Neural Network Model Classification Report:")
print(classification_report(y_test, nn_y_pred))

print("Random Forest Model Accuracy:", accuracy_score(y_test, rf_y_pred))
print("Random Forest Model Classification Report:")
print(classification_report(y_test, rf_y_pred))

# Cross-validation sonuçları
print("Neural Network CV Scores:", nn_cv_scores)
print("Neural Network CV Mean Accuracy:", nn_cv_scores.mean())
print("Random Forest CV Scores:", rf_cv_scores)
print("Random Forest CV Mean Accuracy:", rf_cv_scores.mean())

# Loss vs Epoch grafiği için neural network modelini yeniden eğitme
class LossHistory:
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch, loss):
        self.losses.append(loss)

loss_history = LossHistory()

# Neural Network modeli eğitme (loss takibi ile)
nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)
for epoch in range(500):
    nn_clf.fit(X_train, y_train)
    loss = nn_clf.loss_
    loss_history.on_epoch_end(epoch, loss)

# Loss vs Epoch grafiği
plt.plot(range(1, 501), loss_history.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch for Neural Network')
plt.show()

# ROC Curve grafikleri
from sklearn.preprocessing import label_binarize

# Etiketleri binary formatına dönüştürme
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
nn_y_prob = nn_clf.predict_proba(X_test)
rf_y_prob = rf_clf.predict_proba(X_test)

# ROC eğrileri ve AUC hesaplama
fpr_nn = dict()
tpr_nn = dict()
roc_auc_nn = dict()
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()

for i in range(3):
    fpr_nn[i], tpr_nn[i], _ = roc_curve(y_test_bin[:, i], nn_y_prob[:, i])
    roc_auc_nn[i] = auc(fpr_nn[i], tpr_nn[i])
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], rf_y_prob[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# ROC eğrilerini çizme
plt.figure(figsize=(12, 6))

# Neural Network ROC eğrisi
plt.subplot(1, 2, 1)
for i in range(3):
    plt.plot(fpr_nn[i], tpr_nn[i], label=f'Class {i} (AUC = {roc_auc_nn[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network ROC Curve')
plt.legend(loc="lower right")

# Random Forest ROC eğrisi
plt.subplot(1, 2, 2)
for i in range(3):
    plt.plot(fpr_rf[i], tpr_rf[i], label=f'Class {i} (AUC = {roc_auc_rf[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
