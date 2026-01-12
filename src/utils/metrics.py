"""
Evaluation metrics for stock prediction models.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    # Handle near-zero values to avoid division by zero
    epsilon = 1e-8  # Small constant to avoid division by zero
    mask = np.abs(y_true) > epsilon  # Only calculate MAPE for non-zero values
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100
    else:
        mape = np.nan  # Return NaN if all values are near zero

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Auto-detect number of classes
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(unique_classes)

    # Precision, Recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=unique_classes, zero_division=0
    )

    # Macro average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    metrics = {
        'Accuracy': accuracy,
        'Precision_Macro': precision_macro,
        'Recall_Macro': recall_macro,
        'F1_Macro': f1_macro,
        'Confusion_Matrix': cm,
        'Num_Classes': num_classes
    }

    # Add per-class metrics dynamically
    class_names = ['DOWN', 'UP', 'NEUTRAL'][:num_classes]
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            metrics[f'Precision_{class_name}'] = precision[i]
            metrics[f'Recall_{class_name}'] = recall[i]
            metrics[f'F1_{class_name}'] = f1[i]

    return metrics


def directional_accuracy(
    y_true_returns: np.ndarray,
    y_pred_returns: np.ndarray
) -> float:
    """
    Calculate directional accuracy (did we predict the right direction of price change?).

    Args:
        y_true_returns: True percentage returns
        y_pred_returns: Predicted percentage returns

    Returns:
        Directional accuracy (0-1)
    """
    # Compare sign of predicted vs actual returns
    true_direction = np.sign(y_true_returns)
    pred_direction = np.sign(y_pred_returns)

    correct = (true_direction == pred_direction).astype(float)

    return correct.mean()


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def maximum_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Array of returns

    Returns:
        Maximum drawdown (as positive percentage)
    """
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max

    return abs(drawdown.min()) * 100


def win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Array of returns

    Returns:
        Win rate (0-1)
    """
    if len(returns) == 0:
        return 0.0

    return (returns > 0).mean()


def profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Array of returns

    Returns:
        Profit factor
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str = "regression"
) -> Dict[str, float]:
    """
    Calculate all relevant metrics based on model type.

    Args:
        y_true: True values/labels (returns for regression, classes for classification)
        y_pred: Predicted values/labels
        model_type: "regression" or "classification"

    Returns:
        Dictionary of all metrics
    """
    if model_type == "regression":
        metrics = regression_metrics(y_true, y_pred)

        # Add directional accuracy for regression (comparing sign of returns)
        dir_acc = directional_accuracy(y_true, y_pred)
        metrics['Directional_Accuracy'] = dir_acc

    else:  # classification
        metrics = classification_metrics(y_true, y_pred)

    return metrics


def print_metrics(metrics: Dict, model_type: str = "regression") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        model_type: "regression" or "classification"
    """
    print("\n" + "=" * 50)
    print(f"EVALUATION METRICS ({model_type.upper()})")
    print("=" * 50)

    if model_type == "regression":
        print(f"RMSE:                    {metrics['RMSE']:.4f}")
        print(f"MAE:                     {metrics['MAE']:.4f}")
        print(f"MAPE:                    {metrics['MAPE']:.2f}%")
        print(f"R²:                      {metrics['R2']:.4f}")

        if 'Directional_Accuracy' in metrics:
            print(f"Directional Accuracy:    {metrics['Directional_Accuracy']:.2%}")

    else:  # classification
        print(f"Accuracy:                {metrics['Accuracy']:.2%}")

        num_classes = metrics.get('Num_Classes', 2)
        class_names = ['DOWN', 'UP', 'NEUTRAL'][:num_classes]

        print(f"\nPer-Class Metrics:")
        for class_name in class_names:
            if f'Precision_{class_name}' in metrics:
                print(f"  {class_name:7s} - Precision: {metrics[f'Precision_{class_name}']:.2%}, "
                      f"Recall: {metrics[f'Recall_{class_name}']:.2%}, "
                      f"F1: {metrics[f'F1_{class_name}']:.2%}")

        print(f"\nMacro Averages:")
        print(f"  Precision: {metrics['Precision_Macro']:.2%}")
        print(f"  Recall:    {metrics['Recall_Macro']:.2%}")
        print(f"  F1:        {metrics['F1_Macro']:.2%}")

        print(f"\nConfusion Matrix:")
        cm = metrics['Confusion_Matrix']

        # Print header
        header = "         " + "  ".join([f"{name:>6s}" for name in class_names])
        print(header)

        # Print rows
        for i, label in enumerate(class_names):
            row_values = "  ".join([f"{cm[i][j]:6d}" for j in range(num_classes)])
            print(f"{label:7s}  {row_values}")

    print("=" * 50 + "\n")
