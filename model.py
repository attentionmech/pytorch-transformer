import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """
    The LayerNormalization class implements the Layer Normalization technique.
    Layer normalization normalizes the input features across the last dimension (typically the feature dimension).
    It helps stabilize and accelerate training by reducing internal covariate shift.

    Attributes:
    - alpha (nn.Parameter): A learnable scaling parameter, used to scale the normalized output.
    - bias (nn.Parameter): A learnable bias parameter added to the normalized output.
    - eps (float): A small constant added to the standard deviation for numerical stability (to prevent division by zero).
    """
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        """
        Initializes the LayerNormalization module with the following parameters:
        
        Args:
        - features (int): The number of features (typically `d_model` or hidden_size) to normalize.
        - eps (float): A small constant added to the standard deviation for numerical stability (default: 1e-6).
        """
        super().__init__()
        
        # `eps` is used for numerical stability when dividing by the standard deviation.
        self.eps = eps
        
        # Learnable scaling parameter (`alpha`), initialized to ones with the size of `features`.
        self.alpha = nn.Parameter(torch.ones(features))  # Shape: (features,)
        
        # Learnable bias parameter (`bias`), initialized to zeros with the size of `features`.
        self.bias = nn.Parameter(torch.zeros(features))  # Shape: (features,)

    def forward(self, x):
        """
        Forward pass for LayerNormalization. This normalizes the input tensor `x` along the last dimension 
        (typically the feature dimension).
        
        Args:
        - x (Tensor): The input tensor to normalize. The shape of `x` is typically 
          (batch_size, seq_len, hidden_size), where `hidden_size` refers to the `features`.
        
        Returns:
        - Tensor: The normalized tensor, with the same shape as the input tensor.
        """
        # x: (batch_size, seq_len, hidden_size)
        # The input tensor `x` can have 3 dimensions (e.g., batch_size, sequence_length, hidden_size).
        
        # Calculate the mean of `x` along the last dimension (the feature dimension).
        # The `mean` will be computed across the hidden_size dimension (dim=-1).
        # `mean` will have the shape (batch_size, seq_len, 1), as we keep the sequence length and batch dimensions.
        mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch_size, seq_len, 1)
        
        # Calculate the standard deviation of `x` along the last dimension (the feature dimension).
        # Similar to the mean, `std` will be computed across the hidden_size dimension (dim=-1).
        # `std` will also have shape (batch_size, seq_len, 1).
        std = x.std(dim=-1, keepdim=True)  # Shape: (batch_size, seq_len, 1)
        
        # The formula for Layer Normalization is:
        # normalized_x = alpha * ((x - mean) / (std + eps)) + bias
        # where `alpha` is the learnable scaling parameter, and `bias` is the learnable bias parameter.

        # `std + self.eps` ensures that the division doesn't cause issues when the standard deviation is very small.
        normalized_x = (x - mean) / (std + self.eps)  # Shape: (batch_size, seq_len, hidden_size)
        
        # Scale the normalized values by `alpha` and add the `bias`.
        # `alpha` and `bias` are applied element-wise along the hidden_size dimension.
        return self.alpha * normalized_x + self.bias  # Shape: (batch_size, seq_len, hidden_size)


class FeedForwardBlock(nn.Module):
    """
    This class implements a feed-forward neural network block, which consists of two linear layers 
    with ReLU activation and dropout applied between them, as described in the Transformer architecture.
    
    Attributes:
    - linear_1 (nn.Linear): The first linear layer that transforms the input to a higher dimension.
    - dropout (nn.Dropout): A dropout layer used for regularization between the two linear layers.
    - linear_2 (nn.Linear): The second linear layer that maps the higher-dimensional output back to the original dimension.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Initializes the FeedForwardBlock with the following parameters:
        
        Args:
        - d_model (int): The dimensionality of the input and output vectors (embedding size).
        - d_ff (int): The number of units in the hidden layer (typically larger than `d_model`).
        - dropout (float): The dropout probability used for regularization between the two linear layers.
        """
        super().__init__()

        # First linear transformation: from d_model to d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)  # (d_model, d_ff)
        # Dropout layer to regularize the activations
        self.dropout = nn.Dropout(dropout)
        # Second linear transformation: from d_ff back to d_model
        self.linear_2 = nn.Linear(d_ff, d_model)  # (d_ff, d_model)

    def forward(self, x):
        """
        Forward pass through the FeedForwardBlock.
        
        Args:
        - x (Tensor): The input tensor with shape (batch_size, seq_len, d_model).
        
        Returns:
        - Tensor: The output tensor with shape (batch_size, seq_len, d_model), after applying the two 
                  linear transformations, ReLU activation, and dropout.
        """
        # Apply the first linear layer followed by ReLU activation, then dropout and the second linear layer
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):
    """
    This class is responsible for converting input tokens (e.g., word indices) into dense vectors (embeddings)
    and scaling them according to the square root of the model dimension, as described in the original Transformer paper.
    
    Attributes:
    - d_model (int): The dimensionality of the output embedding vectors.
    - vocab_size (int): The size of the vocabulary (i.e., the total number of unique tokens).
    - embedding (nn.Embedding): The embedding layer that converts input indices to vectors.
    """
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the InputEmbeddings class with the given parameters.
        
        Args:
        - d_model (int): The dimension of the output embedding vectors (size of the embedding space).
        - vocab_size (int): The size of the vocabulary (number of unique tokens).
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # nn.Embedding creates a matrix of size (vocab_size, d_model) where each row is an embedding vector
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass through the embedding layer, which converts token indices into dense vectors.
        
        Args:
        - x (Tensor): The input tensor containing token indices with shape (batch_size, seq_len).
        
        Returns:
        - Tensor: The output tensor with shape (batch_size, seq_len, d_model), 
                  where each token is represented by its embedding.
        """
        # Multiply by sqrt(d_model) to scale the embeddings, as done in the original Transformer paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    This class implements positional encodings as described in the Transformer paper.
    Positional encoding adds information about the order of tokens in the sequence since the Transformer is agnostic to token order.

    Attributes:
    - d_model (int): The dimensionality of the positional encoding vectors.
    - seq_len (int): The length of the input sequence.
    - dropout (nn.Dropout): The dropout layer to apply to the positional encoding.
    - pe (Tensor): The precomputed positional encoding matrix.
    """
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Initializes the PositionalEncoding class.
        
        Args:
        - d_model (int): The dimensionality of the positional encoding vectors.
        - seq_len (int): The length of the input sequence (maximum number of positions).
        - dropout (float): The dropout probability for regularization.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix for the positional encodings of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector for positions (0, 1, 2, ..., seq_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector for the scaling factor of each dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        
        # Register the positional encoding as a buffer (no gradients)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass that adds positional encodings to the input tensor.
        
        Args:
        - x (Tensor): The input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
        - Tensor: The input tensor with positional encodings added, shape (batch_size, seq_len, d_model).
        """
        # Add positional encoding to the input tensor (broadcasting across batch dimension)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    """
    This class implements a residual connection followed by layer normalization, as used in the Transformer model.
    The purpose of residual connections is to allow gradients to flow more easily through the network, avoiding the vanishing gradient problem.
    
    Attributes:
    - dropout (nn.Dropout): The dropout layer used for regularization.
    - norm (LayerNormalization): The LayerNormalization module applied after the residual connection.
    """
    
    def __init__(self, features: int, dropout: float) -> None:
        """
        Initializes the ResidualConnection class.
        
        Args:
        - features (int): The number of features (or model dimension) for the input tensor.
        - dropout (float): Dropout probability to apply to the residual connections.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """
        Forward pass through the residual connection followed by layer normalization.
        
        Args:
        - x (Tensor): The input tensor with shape (batch_size, seq_len, features).
        - sublayer (callable): A function (or submodule) that is applied to the input tensor.
        
        Returns:
        - Tensor: The output tensor after applying the residual connection and normalization.
        """
        # Apply the sublayer to the normalized input tensor and add the residual connection
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """
    This class implements the Multi-Head Attention mechanism from the Transformer architecture.
    Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

    Attributes:
    - d_model (int): The dimensionality of the input and output vectors (i.e., the size of the embedding space).
    - h (int): The number of attention heads in the multi-head attention mechanism.
    - d_k (int): The dimension of each attention head's query/key/value vectors.
    - w_q, w_k, w_v, w_o (nn.Linear): Linear layers for generating the query, key, value, and output projections.
    - dropout (nn.Dropout): Dropout layer for regularization applied to attention scores.
    """
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        Initializes the MultiHeadAttentionBlock with the following parameters:
        
        Args:
        - d_model (int): The dimension of the input feature space (typically the model size, e.g., 512).
        - h (int): The number of attention heads.
        - dropout (float): Dropout probability for attention scores (used for regularization).
        """
        super().__init__()

        # Assert that the dimension of the model (d_model) is divisible by the number of heads (h)
        assert d_model % h == 0, "d_model must be divisible by the number of heads (h)."

        # Calculate the dimension of each attention head's query, key, and value vectors
        self.d_k = d_model // h  # The dimension of each attention head's query/key/value

        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)

        # Dropout layer for regularizing the attention scores
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Implements the scaled dot-product attention as described in the original paper.
        
        Args:
        - query (Tensor): Query tensor with shape (batch, h, seq_len, d_k).
        - key (Tensor): Key tensor with shape (batch, h, seq_len, d_k).
        - value (Tensor): Value tensor with shape (batch, h, seq_len, d_k).
        - mask (Tensor or None): Mask tensor to prevent attention to certain positions (usually padding tokens).
        - dropout (nn.Dropout): Dropout layer to apply to attention scores for regularization.

        Returns:
        - Tensor: The weighted sum of the values after applying attention.
        - Tensor: The attention scores (for visualization or analysis).
        """
        d_k = query.shape[-1]  # The dimension of each attention head's key/query vectors (d_k)

        # Compute attention scores as the scaled dot-product of the query and key
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled by sqrt(d_k)

        # If a mask is provided, apply it to the attention scores (set masked positions to a very low value)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)  # Set masked positions to -inf for softmax

        # Apply softmax over the last dimension (seq_len) to compute the attention weights
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)

        # Apply dropout to the attention scores
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute the final weighted sum of values using the attention scores
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Forward pass for MultiHeadAttentionBlock. This method computes the attention and outputs the result.

        Args:
        - q (Tensor): The query tensor with shape (batch, seq_len, d_model).
        - k (Tensor): The key tensor with shape (batch, seq_len, d_model).
        - v (Tensor): The value tensor with shape (batch, seq_len, d_model).
        - mask (Tensor): The mask tensor used to avoid attention to certain positions (e.g., padding).

        Returns:
        - Tensor: The output tensor after applying the multi-head attention mechanism.
        """
        
        # Apply linear transformations to query, key, and value tensors
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Reshape and transpose the query, key, and value to split into multiple heads
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Compute attention and get the weighted sum of the values
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine the results from all attention heads into a single tensor
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Apply the output projection (Wo) to get the final output
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class EncoderBlock(nn.Module):
    # The EncoderBlock class represents a single encoder block in the transformer.
    # It consists of a self-attention layer followed by a feed-forward network.
    # Each block applies residual connections around the attention and feed-forward layers.
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        # `features` represents the hidden size or `d_model` (the dimensionality of the input and output of the block).
        # `self_attention_block` is an instance of a multi-head self-attention layer.
        # `feed_forward_block` is an instance of a position-wise feed-forward network.
        # `dropout` is the dropout rate to apply in the residual connections for regularization.

        super().__init__()
        # Store the self-attention and feed-forward blocks as attributes.
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        
        # Create residual connections around the self-attention and feed-forward layers.
        # `ResidualConnection` applies a residual connection to the input of each layer.
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # The forward pass defines how the input `x` (shape: batch_size x seq_len x d_model) is processed.
        # `src_mask` is the mask applied to prevent attending to padding tokens in the source sequence.

        # Apply the first residual connection (around self-attention block).
        # The lambda function wraps the self-attention block to pass the required arguments.
        # The self-attention block computes self-attention with the input `x` three times (Q, K, V are all `x`).
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Apply the second residual connection (around the feed-forward block).
        # The feed-forward block operates independently on each position in the sequence.
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        # Return the output after applying both layers and their residual connections.
        return x


class Encoder(nn.Module):
    # The Encoder class represents the full encoder in the transformer.
    # It consists of multiple `EncoderBlock`s stacked together.
    # Each block applies self-attention followed by a feed-forward network.
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        # `features` represents the size of the model's hidden state (`d_model`).
        # `layers` is a list of `EncoderBlock` instances that will be stacked in the encoder.
        
        super().__init__()
        # Store the layers (encoder blocks) as a list.
        self.layers = layers
        
        # LayerNormalization is applied to the final output after processing all layers.
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        # The forward pass processes the input sequence `x` (shape: batch_size x seq_len x d_model)
        # through each encoder block.
        # `mask` is the mask applied to the input sequence (to handle padding tokens).
        
        # Pass the input `x` through each encoder block sequentially.
        for layer in self.layers:
            x = layer(x, mask)
        
        # After all layers, apply layer normalization to the output of the last encoder block.
        return self.norm(x)


class DecoderBlock(nn.Module):
    # The DecoderBlock class represents a single block in the decoder.
    # It consists of a self-attention layer, a cross-attention layer, and a feed-forward network.
    # Like the encoder block, it also uses residual connections around each layer.

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        # `features` represents the dimensionality of the model's hidden states (`d_model`).
        # `self_attention_block` is the multi-head self-attention block for the decoder's own input sequence.
        # `cross_attention_block` is the multi-head cross-attention block, attending to the encoder's output.
        # `feed_forward_block` is the position-wise feed-forward network for the decoder.
        # `dropout` is the dropout rate used for regularization in the residual connections.
        
        super().__init__()
        
        # Store the self-attention, cross-attention, and feed-forward blocks as attributes.
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
        # Create residual connections around all three blocks (self-attention, cross-attention, and feed-forward layers).
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # The forward pass processes the input target sequence `x` (shape: batch_size x seq_len x d_model)
        # with the output from the encoder `encoder_output` (shape: batch_size x seq_len x d_model).
        # `src_mask` is the mask for the source sequence (applied to the encoder's output).
        # `tgt_mask` is the mask for the target sequence (used to prevent attending to future tokens).

        # Apply the first residual connection (around self-attention block).
        # The self-attention block computes self-attention with the target sequence itself.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        # Apply the second residual connection (around cross-attention block).
        # The cross-attention block computes attention between the target sequence `x`
        # and the encoder output `encoder_output`.
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        # Apply the third residual connection (around feed-forward block).
        # The feed-forward block operates independently on each position in the sequence.
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        # Return the output after applying all layers and their residual connections.
        return x

class Decoder(nn.Module):
    # The Decoder class processes the target sequence in the transformer architecture.
    # It takes an input `x` (target sequence) and the `encoder_output` (from the encoder) to generate the final output.
    # It applies a series of decoder layers and performs layer normalization at the end.

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        # `features` represents the number of channels or the size of the hidden state (`d_model`).
        # `layers` is a list of individual decoder layers (could be multiple layers of self-attention, cross-attention, and feed-forward networks).

        super().__init__()
        # Store the decoder layers in the class.
        self.layers = layers
        
        # LayerNormalization applied to the final output of the decoder layers.
        # This helps stabilize training and improve generalization.
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # The `forward` method defines the forward pass of the decoder.
        # `x` is the input target sequence (shape: batch_size x seq_len x d_model).
        # `encoder_output` is the output from the encoder (shape: batch_size x seq_len x d_model).
        # `src_mask` is the source sequence mask (to prevent attending to padding).
        # `tgt_mask` is the target sequence mask (used to prevent attending to future tokens in autoregressive tasks).

        # Loop through each layer in the `self.layers` list and pass `x` through them.
        # This allows for stacking multiple layers of attention and feed-forward networks in the decoder.
        for layer in self.layers:
            # Each layer processes the target sequence `x`, encoder output `encoder_output`,
            # and applies both source (`src_mask`) and target (`tgt_mask`) masks during attention operations.
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # After all layers, apply layer normalization to the output of the last layer.
        # This stabilizes the learning and makes the training process more robust.
        return self.norm(x)


class ProjectionLayer(nn.Module):
    # The ProjectionLayer class is a simple fully connected layer that projects the output of the decoder
    # into the target vocabulary space. The output is used for predicting the next word in the sequence.
    
    def __init__(self, d_model, vocab_size) -> None:
        # `d_model` is the size of the model's hidden state (the feature dimension).
        # `vocab_size` is the number of words in the target vocabulary, i.e., the number of possible output classes.
        
        super().__init__()
        
        # Define a linear layer that transforms the hidden state into the target vocabulary size.
        # This will output logits, which are raw scores for each word in the vocabulary.
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # The `forward` method defines the forward pass of the projection layer.
        # `x` is the output from the last decoder layer (shape: batch_size x seq_len x d_model).
        # We apply the linear transformation to map it to the vocabulary size.

        # Output shape: (batch_size, seq_len, vocab_size), where vocab_size is the number of words in the target vocabulary.
        # Each entry in the output represents the logit (raw score) for a word in the vocabulary.
        return self.proj(x)

class Transformer(nn.Module):
    # The constructor (__init__) initializes the Transformer model.
    # It takes in encoder, decoder, embedding layers, positional encoding, and the projection layer as arguments.
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()

        # Encoder is the stack of encoder layers (multi-head attention + feed-forward networks)
        self.encoder = encoder

        # Decoder is the stack of decoder layers (self-attention, cross-attention, feed-forward networks)
        self.decoder = decoder

        # Embedding layer for the source (input) sequence. Converts word indices into embeddings.
        self.src_embed = src_embed

        # Embedding layer for the target (output) sequence. Converts word indices into embeddings.
        self.tgt_embed = tgt_embed

        # Positional encoding for the source sequence, which adds information about token positions.
        self.src_pos = src_pos

        # Positional encoding for the target sequence, which adds information about token positions.
        self.tgt_pos = tgt_pos

        # The final projection layer converts the decoder output into the size of the target vocabulary.
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # The `encode` function processes the source input sequence (src).
        # It first embeds the source tokens into a dense representation, then applies positional encoding.
        # After that, it passes the embedded source through the encoder layers to generate the encoded representation.
        
        # (batch, seq_len) -> (batch, seq_len, d_model)
        src = self.src_embed(src)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model) with positional encoding added
        src = self.src_pos(src)
        
        # The encoded source is passed through the encoder (which consists of multiple layers)
        # src_mask is used to mask certain parts of the input, e.g., padding tokens.
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # The `decode` function generates the target sequence (tgt) from the encoded source (encoder_output).
        # It first embeds the target tokens, then applies positional encoding.
        # After that, it passes the embedded target and the encoder output through the decoder layers to produce the final output.

        # (batch, seq_len) -> (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model) with positional encoding added
        tgt = self.tgt_pos(tgt)

        # The target and encoded source are passed through the decoder to generate the output sequence.
        # The tgt_mask is used to ensure that the decoder does not look at future tokens in the target sequence (autoregressive property).
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # The `project` function takes the decoder output (of shape (batch, seq_len, d_model)) and projects it into the vocabulary space.
        # The projection layer maps the hidden states into logits of the size of the vocabulary.
        # This is the final step to generate probabilities over the target vocabulary.

        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers for the source and target sequences.
    # These layers will map token IDs to dense vector representations of size `d_model`.
    src_embed = InputEmbeddings(d_model, src_vocab_size)  # Source sequence embedding
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)  # Target sequence embedding

    # Create positional encoding layers for both source and target sequences.
    # These layers will inject positional information to the embeddings.
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)  # Positional encoding for the source
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)  # Positional encoding for the target
    
    # Create the encoder blocks. Each block contains a self-attention mechanism and a feed-forward network.
    encoder_blocks = []
    for _ in range(N):
        # Each encoder block has a multi-head self-attention layer followed by a feed-forward block.
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # Self-attention for encoder
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # Feed-forward network for encoder
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)  # Combined encoder block
        encoder_blocks.append(encoder_block)  # Add the block to the encoder stack

    # Create the decoder blocks. Each block contains a self-attention mechanism, cross-attention to the encoder, and a feed-forward network.
    decoder_blocks = []
    for _ in range(N):
        # Each decoder block has:
        # - A self-attention mechanism to attend to previous target tokens
        # - A cross-attention mechanism to attend to the encoder's output
        # - A feed-forward network
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # Self-attention for decoder
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # Cross-attention to encoder
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # Feed-forward network for decoder
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)  # Combined decoder block
        decoder_blocks.append(decoder_block)  # Add the block to the decoder stack
    
    # Create the encoder and decoder modules by passing in the blocks.
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))  # Stack of encoder blocks
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))  # Stack of decoder blocks
    
    # Create the projection layer which maps the decoder output to the target vocabulary size.
    # This will generate the logits for each token in the target sequence.
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)  # Mapping to target vocabulary size
    
    # Create the full transformer model by combining the encoder, decoder, and other components.
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters of the transformer model using Xavier uniform initialization.
    # This helps in faster convergence by setting initial weights properly.
    for p in transformer.parameters():
        if p.dim() > 1:  # Check if the parameter is a weight matrix (dim > 1)
            nn.init.xavier_uniform_(p)  # Initialize with Xavier uniform distribution
    
    # Return the fully constructed transformer model.
    return transformer
