"""
Custom loss functions for stock prediction.
"""
import torch
import torch.nn as nn


class DirectionalMSELoss(nn.Module):
    """
    Combined loss: MSE for magnitude + directional accuracy bonus.

    Encourages model to:
    1. Predict accurate magnitudes (MSE)
    2. Get the direction right (sign correctness)
    """

    def __init__(self, directional_weight: float = 0.5):
        """
        Args:
            directional_weight: Weight for directional component (0-1)
                              0 = pure MSE, 1 = pure directional
        """
        super().__init__()
        self.directional_weight = directional_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: True values (batch_size, 1)

        Returns:
            Combined loss value
        """
        # MSE component (magnitude accuracy)
        mse_loss = self.mse(predictions, targets)

        # Directional component (sign correctness)
        # Penalize when signs don't match
        signs_match = (torch.sign(predictions) == torch.sign(targets)).float()
        directional_loss = 1.0 - signs_match.mean()

        # Combine losses
        total_loss = (1 - self.directional_weight) * mse_loss + self.directional_weight * directional_loss

        return total_loss


class DirectionalHuberLoss(nn.Module):
    """
    Combined loss: Huber for magnitude + directional accuracy bonus.

    More robust to outliers than DirectionalMSELoss.

    FIXED: Scale directional loss to match Huber loss magnitude
    """

    def __init__(self, delta: float = 1.0, directional_weight: float = 0.5):
        """
        Args:
            delta: Huber loss delta parameter
            directional_weight: Weight for directional component (0-1)
        """
        super().__init__()
        self.delta = delta
        self.directional_weight = directional_weight
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: True values (batch_size, 1)

        Returns:
            Combined loss value
        """
        # Huber component (robust magnitude accuracy)
        huber_loss = self.huber(predictions, targets)

        # Directional component (sign correctness)
        # FIXED: Use MSE-based penalty scaled to match Huber magnitude
        signs_match = (torch.sign(predictions) == torch.sign(targets)).float()

        # Penalty for wrong direction: use squared error of predictions when wrong
        directional_penalty = torch.where(
            signs_match == 1.0,
            torch.zeros_like(predictions),  # No penalty if correct direction
            torch.square(predictions - targets)  # MSE penalty if wrong direction
        ).mean()

        # Combine losses: Huber for all + extra penalty for wrong direction
        total_loss = huber_loss + self.directional_weight * directional_penalty

        return total_loss


class WeightedDirectionalLoss(nn.Module):
    """
    Weighted loss that penalizes directional errors more heavily.

    Gives higher weight to:
    1. Getting the direction right
    2. Large magnitude predictions (encourages model to be less conservative)
    """

    def __init__(self, base_loss: str = 'huber', delta: float = 1.0,
                 directional_weight: float = 0.3, magnitude_scale: float = 1.0):
        """
        Args:
            base_loss: 'huber' or 'mse'
            delta: Huber loss delta (if using huber)
            directional_weight: Weight for directional component
            magnitude_scale: Scale factor for magnitude-based weighting
        """
        super().__init__()
        self.directional_weight = directional_weight
        self.magnitude_scale = magnitude_scale

        if base_loss == 'huber':
            self.base_loss_fn = nn.HuberLoss(delta=delta, reduction='none')
        else:
            self.base_loss_fn = nn.MSELoss(reduction='none')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted directional loss.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: True values (batch_size, 1)

        Returns:
            Weighted loss value
        """
        # Base loss (per-sample)
        base_loss = self.base_loss_fn(predictions, targets)

        # Directional correctness
        signs_match = (torch.sign(predictions) == torch.sign(targets)).float()

        # Magnitude-based weighting (encourage larger predictions)
        # Weight increases with target magnitude - we want to predict large moves
        magnitude_weights = 1.0 + self.magnitude_scale * torch.abs(targets)

        # Apply directional penalty
        # If direction is wrong, increase loss by directional_weight
        directional_penalty = torch.where(
            signs_match == 1.0,
            torch.ones_like(base_loss),  # Correct direction: no penalty
            torch.ones_like(base_loss) * (1.0 + self.directional_weight)  # Wrong direction: penalty
        )

        # Combine: base loss * magnitude weights * directional penalty
        weighted_loss = base_loss * magnitude_weights * directional_penalty

        return weighted_loss.mean()


class QuantileLoss(nn.Module):
    """
    Quantile regression loss for uncertainty estimation.

    Predicts multiple quantiles instead of point estimates.
    Better handles prediction uncertainty.
    """

    def __init__(self, quantiles: list = None):
        """
        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]  # Default: 10th, 50th, 90th percentiles
        self.quantiles = torch.tensor(quantiles)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                quantile_idx: int = 1) -> torch.Tensor:
        """
        Calculate quantile loss.

        Args:
            predictions: Model predictions (batch_size, num_quantiles)
            targets: True values (batch_size, 1)
            quantile_idx: Which quantile to use (default: 1 = median)

        Returns:
            Quantile loss
        """
        quantile = self.quantiles[quantile_idx].to(predictions.device)
        errors = targets - predictions

        loss = torch.where(
            errors >= 0,
            quantile * errors,
            (quantile - 1) * errors
        )

        return loss.mean()


class RankCorrelationLoss(nn.Module):
    """
    PHASE 1: Cross-Sectional Rank Correlation Loss

    This is THE KEY innovation for trading models.

    Instead of optimizing for magnitude accuracy (MSE), this loss optimizes
    for RANKING accuracy - which is what actually matters for long-short portfolios.

    The model learns: "Among stocks A, B, C today, which will outperform?"
    Not: "What is stock A's exact return tomorrow?"

    This naturally reduces turnover because it focuses on relative orderings,
    which are more stable than absolute return predictions.

    Implementation:
    - Uses Spearman rank correlation as differentiable objective
    - Operates on batches of stocks from the same time period
    - Implicitly penalizes unstable rankings
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize rank correlation loss.

        Args:
            temperature: Softmax temperature for soft ranking (default: 1.0)
                        Lower = harder ranking, Higher = softer ranking
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate rank correlation loss.

        This is a differentiable approximation of Spearman's rank correlation.

        Args:
            predictions: Model predictions (batch_size, 1) or (batch_size,)
            targets: True returns (batch_size, 1) or (batch_size,)

        Returns:
            Negative rank correlation (to minimize = maximize correlation)
        """
        # Ensure shapes are correct
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()

        # Get batch size
        n = predictions.shape[0]

        if n < 2:
            # Can't compute correlation with < 2 samples
            return torch.tensor(0.0, device=predictions.device)

        # Compute ranks using soft ranking (differentiable)
        # Rank(x_i) = sum_j sigmoid((x_i - x_j) / temperature)

        # Expand for pairwise comparisons
        pred_diff = predictions.unsqueeze(0) - predictions.unsqueeze(1)  # (n, n)
        target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # (n, n)

        # Soft ranks (differentiable approximation)
        pred_ranks = torch.sigmoid(pred_diff / self.temperature).sum(dim=1)
        target_ranks = torch.sigmoid(target_diff / self.temperature).sum(dim=1)

        # Spearman correlation = Pearson correlation of ranks
        pred_ranks_centered = pred_ranks - pred_ranks.mean()
        target_ranks_centered = target_ranks - target_ranks.mean()

        numerator = (pred_ranks_centered * target_ranks_centered).sum()
        denominator = torch.sqrt(
            (pred_ranks_centered ** 2).sum() * (target_ranks_centered ** 2).sum()
        )

        # Avoid division by zero
        if denominator == 0:
            return torch.tensor(0.0, device=predictions.device)

        correlation = numerator / (denominator + 1e-8)

        # Return negative correlation (we want to maximize correlation = minimize negative)
        return -correlation


class CombinedRankRegressionLoss(nn.Module):
    """
    PHASE 1: Combined Loss = Rank Loss + Regression Loss

    This is the final loss function we'll use.

    Combines:
    1. Cross-sectional rank correlation (70% weight) - for tradeable signal
    2. Huber regression (30% weight) - for magnitude calibration

    This dual objective:
    - Optimizes ranking quality (what drives P&L)
    - Maintains magnitude awareness (prevents degenerate solutions)
    - Naturally reduces turnover (stable rankings)
    """

    def __init__(
        self,
        rank_weight: float = 0.7,
        huber_delta: float = 1.0,
        temperature: float = 1.0
    ):
        """
        Initialize combined loss.

        Args:
            rank_weight: Weight for rank loss (0-1), default 0.7
                        Remaining weight goes to Huber loss
            huber_delta: Delta parameter for Huber loss
            temperature: Temperature for soft ranking
        """
        super().__init__()
        self.rank_weight = rank_weight
        self.rank_loss = RankCorrelationLoss(temperature=temperature)
        self.huber_loss = nn.HuberLoss(delta=huber_delta)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            predictions: Model predictions (batch_size, 1) or (batch_size,)
            targets: True returns (batch_size, 1) or (batch_size,)

        Returns:
            Combined loss value
        """
        # Rank correlation component
        rank_loss_val = self.rank_loss(predictions, targets)

        # Huber regression component
        huber_loss_val = self.huber_loss(
            predictions.squeeze() if predictions.dim() > 1 else predictions,
            targets.squeeze() if targets.dim() > 1 else targets
        )

        # Combine with weights
        total_loss = (
            self.rank_weight * rank_loss_val +
            (1 - self.rank_weight) * huber_loss_val
        )

        return total_loss
