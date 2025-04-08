import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# --- Activation Function Helper ---
def get_activation(activation_name):
    """Helper to get activation function from name."""
    if activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

# --- Deep Autoencoder (MLP based) ---
class DeepAE(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims=None, activation='relu', dropout_rate=0.1):
        """
        Initializes a Deep Autoencoder.

        Args:
            input_dim (int): Dimension of the input features.
            encoding_dim (int): Dimension of the latent space (bottleneck).
            hidden_dims (list[int], optional): List of dimensions for hidden layers
                                               in the encoder (decoder uses reverse).
                                               Defaults to a single hidden layer if None.
            activation (str, optional): Activation function name ('relu', 'tanh', etc.).
                                        Defaults to 'relu'.
            dropout_rate (float, optional): Dropout rate to apply between layers. Defaults to 0.1.
        """
        super(DeepAE, self).__init__(self);

        if hidden_dims is None:
            # Default to one hidden layer roughly halfway between input and encoding
            hidden_dims = [ (input_dim + encoding_dim) // 2 ]

        act_fn = get_activation(activation)
        layers = OrderedDict()
        last_dim = input_dim

        # Encoder layers
        for i, h_dim in enumerate(hidden_dims):
            layers[f'enc_linear_{i}'] = nn.Linear(last_dim, h_dim)
            layers[f'enc_act_{i}'] = act_fn
            if dropout_rate > 0:
                 layers[f'enc_dropout_{i}'] = nn.Dropout(dropout_rate)
            last_dim = h_dim
        layers['enc_bottleneck'] = nn.Linear(last_dim, encoding_dim)
        # Optional: Add activation to bottleneck? Sometimes done, sometimes not.
        # layers['enc_bottleneck_act'] = act_fn

        self.encoder = nn.Sequential(layers)

        # Decoder layers (reverse order)
        layers = OrderedDict()
        last_dim = encoding_dim
        # Optional: Activation matching the bottleneck
        # layers['dec_bottleneck_act'] = act_fn
        for i, h_dim in enumerate(reversed(hidden_dims)):
            layers[f'dec_linear_{i}'] = nn.Linear(last_dim, h_dim)
            layers[f'dec_act_{i}'] = act_fn
            if dropout_rate > 0:
                 layers[f'dec_dropout_{i}'] = nn.Dropout(dropout_rate)
            last_dim = h_dim
        layers['dec_output'] = nn.Linear(last_dim, input_dim)
        # Optional: Output activation (e.g., Sigmoid if input is normalized 0-1)
        # layers['dec_output_act'] = nn.Sigmoid()

        self.decoder = nn.Sequential(layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Extracts the latent representation."""
        return self.encoder(x)

# --- Variational Autoencoder (MLP based) ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=None, activation='relu', dropout_rate=0.1):
        """
        Initializes a Variational Autoencoder.

        Args:
            input_dim (int): Dimension of the input features.
            latent_dim (int): Dimension of the latent space (bottleneck).
            hidden_dims (list[int], optional): List of dimensions for hidden layers
                                               before the latent space split.
                                               Defaults to a single hidden layer if None.
            activation (str, optional): Activation function name. Defaults to 'relu'.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [ (input_dim + latent_dim) // 2 ]

        act_fn = get_activation(activation)
        encoder_layers = OrderedDict()
        last_dim = input_dim

        # Shared Encoder layers before splitting to mu and log_var
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers[f'enc_linear_{i}'] = nn.Linear(last_dim, h_dim)
            encoder_layers[f'enc_act_{i}'] = act_fn
            if dropout_rate > 0:
                 encoder_layers[f'enc_dropout_{i}'] = nn.Dropout(dropout_rate)
            last_dim = h_dim
        self.encoder_base = nn.Sequential(encoder_layers)

        # Layers for mu and log_var
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_log_var = nn.Linear(last_dim, latent_dim)

        # Decoder layers (reverse order)
        decoder_layers = OrderedDict()
        last_dim = latent_dim
        # Start decoder from latent dim, reverse hidden dims
        for i, h_dim in enumerate(reversed(hidden_dims)):
            decoder_layers[f'dec_linear_{i}'] = nn.Linear(last_dim, h_dim)
            decoder_layers[f'dec_act_{i}'] = act_fn
            if dropout_rate > 0:
                 decoder_layers[f'dec_dropout_{i}'] = nn.Dropout(dropout_rate)
            last_dim = h_dim
        decoder_layers['dec_output'] = nn.Linear(last_dim, input_dim)
        # Optional: Output activation (e.g., Sigmoid)
        # decoder_layers['dec_output_act'] = nn.Sigmoid()

        self.decoder = nn.Sequential(decoder_layers)

    def encode(self, x):
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # Sample from standard normal
        return mu + eps * std # z = mu + epsilon * sigma

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var # Return mu, log_var for loss calculation

# --- LSTM Autoencoder ---
class LSTM_AE(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, latent_dim, num_layers=1,
                 dropout_rate=0.1, bidirectional=False):
        """
        Initializes an LSTM Autoencoder. Assumes input shape (batch, seq_len, input_dim).

        Args:
            input_dim (int): Number of features at each time step.
            seq_len (int): Length of the input sequence.
            hidden_dim (int): Number of features in the LSTM hidden state.
            latent_dim (int): Dimension of the final encoded latent vector.
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
            dropout_rate (float, optional): Dropout rate for LSTM layers (if num_layers > 1). Defaults to 0.1.
            bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Defaults to False.
        """
        super(LSTM_AE, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        lstm_dropout = dropout_rate if num_layers > 1 else 0

        # --- Encoder ---
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Input shape: (batch, seq_len, input_dim)
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )

        # Calculate dimension after LSTM before latent layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        # Option 1: Use final hidden state (requires careful handling of bidirectional)
        # Option 2: Flatten last output step -> latent (simpler)
        # Option 3: Flatten all output steps -> latent
        # Let's use Option 3 for now (flatten entire output sequence)
        # flatten_dim = lstm_output_dim * seq_len # Can be large
        # Alternative: Use final output step or final hidden state
        self.encoder_fc = nn.Linear(lstm_output_dim, latent_dim) # Map final output step to latent

        # --- Decoder ---
        self.decoder_fc = nn.Linear(latent_dim, lstm_output_dim) # Map latent back to LSTM hidden size

        # Need a way to generate sequence from latent vector.
        # Common approach: Repeat latent vector 'seq_len' times as input to decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=lstm_output_dim, # Input is the projected latent state
            hidden_size=hidden_dim,     # Output hidden state size matches encoder
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional # Decoder usually matches encoder bidirectionality
        )

        # Final layer to map LSTM outputs back to input feature dimension
        self.output_fc = nn.Linear(lstm_output_dim, input_dim)


    def encode(self, x):
        # x shape: (batch, seq_len, input_dim)
        # outputs shape: (batch, seq_len, num_directions * hidden_size)
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # cell shape: (num_layers * num_directions, batch, hidden_size)
        outputs, (hidden, cell) = self.encoder_lstm(x)

        # Extract relevant part for latent space
        # Using the output of the *last* time step
        last_output = outputs[:, -1, :] # Shape: (batch, num_directions * hidden_size)

        latent = self.encoder_fc(last_output) # Shape: (batch, latent_dim)
        return latent

    def decode(self, z):
        # z shape: (batch, latent_dim)
        # Prepare input for decoder LSTM
        # Project latent vector back to LSTM hidden size dimension
        decoder_input_base = F.relu(self.decoder_fc(z)) # Shape: (batch, lstm_output_dim)

        # Repeat this vector for each time step to feed the decoder LSTM
        # Shape: (batch, seq_len, lstm_output_dim)
        decoder_lstm_input = decoder_input_base.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decode using LSTM
        # outputs shape: (batch, seq_len, num_directions * hidden_size)
        decoder_outputs, _ = self.decoder_lstm(decoder_lstm_input)

        # Map LSTM outputs back to the original feature dimension for each time step
        # Input shape: (batch * seq_len, lstm_output_dim)
        # Output shape: (batch * seq_len, input_dim)
        recon_x = self.output_fc(decoder_outputs.reshape(-1, decoder_outputs.size(2)))

        # Reshape back to sequence: (batch, seq_len, input_dim)
        recon_x = recon_x.reshape(decoder_outputs.size(0), self.seq_len, self.input_dim)
        return recon_x

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x
