import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    classes = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(7, 5))

    if len(classes) == 2:
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    else:
        y_bin = label_binarize(y_test, classes=classes)
        y_score = model.predict_proba(X_test)
        for i, cls in enumerate(classes):
            RocCurveDisplay.from_predictions(y_bin[:, i], y_score[:, i], ax=ax, name=f'Class {cls}')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
    else:
        print(f"Feature importance not available for {model_name}.")
        return

    indices = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], ax=ax, palette='viridis')
    ax.set_title(f'Feature Importance — {model_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.show()

def full_evaluation_report(model, X_test, y_test, feature_names, model_name):
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, feature_names, model_name)
