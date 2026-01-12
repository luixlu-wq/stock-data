"""
LSTM models for stock price prediction (regression and classification).
Includes multi-task learning and attention mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMRegressor(nn.Module):
    """LSTM model for stock price prediction (regression)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM regressor.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMRegressor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layers
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Output: single price value
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, sequence_length, input_size)

        Returns:
            Predicted prices (batch_size, 1)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last time step output
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Fully connected layers
        output = self.fc(hidden)

        return output


class LSTMClassifier(nn.Module):
    """LSTM model for stock direction prediction (classification)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM classifier.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (UP=1, DOWN=0, NEUTRAL=2)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layers
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)  # Output: class logits
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, sequence_length, input_size)

        Returns:
            Class logits (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last time step output
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Fully connected layers
        output = self.fc(hidden)

        return output


def create_model(
    model_type: str,
    input_size: int,
    config: dict,
    num_classes: int = 3
) -> nn.Module:
    """
    Create LSTM model based on type and configuration.

    Args:
        model_type: "regression", "classification", or "multitask"
        input_size: Number of input features
        config: Model configuration dictionary
        num_classes: Number of classification classes (auto-detected from data)

    Returns:
        PyTorch model
    """
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.3)
    bidirectional = config.get('bidirectional', False)
    use_attention = config.get('use_attention', True)

    if model_type == "multitask":
        # Multi-task learning: both regression and classification
        model = MultiTaskLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,  # Auto-detected (2 for binary, 3 for three-class)
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention
        )

    elif model_type == "regression":
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

    elif model_type == "classification":
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,  # Auto-detected (2 for binary, 3 for three-class)
            dropout=dropout,
            bidirectional=bidirectional
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


class AttentionLayer(nn.Module):
    """Self-attention mechanism for LSTM outputs."""

    def __init__(self, hidden_size: int):
        """
        Initialize attention layer.

        Args:
            hidden_size: Size of LSTM hidden states
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        """
        Apply attention mechanism.

        Args:
            lstm_out: LSTM outputs (batch_size, seq_len, hidden_size)

        Returns:
            Context vector (batch_size, hidden_size)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)

        # Apply attention weights to LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)

        return context, attention_weights


class MultiTaskLSTM(nn.Module):
    """Multi-task LSTM model for both regression and classification with attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = True
    ):
        """
        Initialize multi-task LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of classification classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        super(MultiTaskLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        if use_attention:
            self.attention = AttentionLayer(lstm_output_size)

        # Task-specific heads
        # Regression head (predicts percentage return)
        self.regression_head = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Classification head (predicts direction: UP/DOWN/NEUTRAL)
        self.classification_head = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for both tasks.

        Args:
            x: Input tensor (batch_size, sequence_length, input_size)

        Returns:
            Tuple of (regression_output, classification_logits)
        """
        # Shared LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention or use last hidden state
        if self.use_attention:
            # Use attention over all time steps
            context, _ = self.attention(lstm_out)
        else:
            # Use last hidden state
            if self.bidirectional:
                context = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                context = hidden[-1]

        # Task-specific predictions
        regression_output = self.regression_head(context)
        classification_logits = self.classification_head(context)

        return regression_output, classification_logits


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
