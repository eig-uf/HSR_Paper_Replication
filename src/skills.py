import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans


def encode_instructions(instructions, model_name="bert-base-uncased"):
    """
        Encode task instructions into fixed-size numerical embeddings using BERT.

        BERT produces a 768-dimensional embedding for each token. We use masked
        mean pooling to average the token embeddings into one vector per instruction,
        excluding padding tokens.

        Shapes:
            Tokenized inputs:  (N, L)
            BERT hidden states: (N, L, 768)
            Attention mask:     (N, L, 1)
            Final embeddings:   (N, 768)

        Where:
            N = Number of task instructions
            L = Number of tokens per instruction (after padding)
            768 = BERT-base hidden dimension

        Returns:
            NumPy array of shape (N, 768), suitable for K-means clustering.

        Example:
            90 LIBERO task instructions -> (90, 768) embedding matrix.
    """
    # Load the pretrained BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Tokenize instructions:
    # padding=True: Pad shorter instructions to match the longest
    # truncation=True: Truncate instructions exceeding BERT's maximum length
    # return_tensors="pt": Return PyTorch tensors
    inputs = tokenizer(
        instructions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Add a dimension to the attention mask for broadcasting
    # Shape: (batch, tokens) -> (batch, tokens, 1)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling: Average token embeddings while ignoring padding
    # Multiply by mask to zero out padding, sum tokens, divide by valid token count
    mask = inputs["attention_mask"].unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

    return embeddings.numpy()



def cluster_skills(embeddings, k=10):
    """
    Cluster task instruction embeddings into K skill groups.

    Input:  (N, 768) embedding matrix
    Output: (N,) cluster labels ranging from 0 to K-1
    """
    # Cluster semantically similar task embeddings
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    return kmeans.fit_predict(embeddings)