"""MrBERT model configuration - BERT with MrT5-style delete gates."""

from transformers import BertConfig


class MrBertConfig(BertConfig):
    """
    Configuration class for MrBERT model.
    
    This extends the standard BERT configuration with parameters for the
    delete gate mechanism from the MrT5 paper. The delete gate learns to
    selectively remove tokens during encoding to improve efficiency.
    
    Args:
        sigmoid_mask_scale (`float`, *optional*, defaults to -30.0):
            Scale factor for the sigmoid activation in the delete gate.
            More negative values lead to stronger deletion signals.
            MrT5 paper uses -30.0.
        gate_layer_norm (`bool`, *optional*, defaults to True):
            Whether to apply layer normalization before the delete gate.
        deletion_threshold (`float`, *optional*, defaults to -15.0):
            Threshold for counting a token as deleted. Tokens with gate values 
            below this threshold are considered deleted for metrics.
            MrT5 paper uses sigmoid_mask_scale / 2 = -15.0.
        delete_gate_layer (`int`, *optional*, defaults to 3):
            The encoder layer index where the delete gate is applied.
            Layer indexing starts at 0. MrT5 paper uses layer 3.
        use_softmax1 (`bool`, *optional*, defaults to True):
            Whether to use the softmax1 variant (softmax with n+1 denominator)
            for attention score normalization. Recommended by the MrT5 paper.
        deletion_type (`str`, *optional*, defaults to "scaled_sigmoid"):
            Type of delete gate to use. Options:
            - "scaled_sigmoid": Learnable gate with scaled sigmoid activation
            - "log_sigmoid": Learnable gate with log sigmoid activation
            - "random": Random deletion for ablation studies
            - "fixed": Fixed deletion pattern for ablation studies
        random_deletion_probability (`float`, *optional*, defaults to 0.5):
            Probability of deletion when using random deletion type.
        fixed_deletion_amount (`float`, *optional*, defaults to 0.5):
            Fraction of tokens to delete when using fixed deletion type.
        use_gumbel_noise (`bool`, *optional*, defaults to False):
            Whether to add Gumbel noise to delete gate logits during training.
            This can help with exploration during training.
    """
    
    model_type = "mrbert"
    
    def __init__(
        self,
        sigmoid_mask_scale: float = -30.0,  # MrT5 default
        gate_layer_norm: bool = True,
        deletion_threshold: float = -15.0,  # MrT5 default (sigmoid_mask_scale / 2)
        delete_gate_layer: int = 3,  # MrT5 default (layer 3, 0-indexed)
        use_softmax1: bool = True,
        deletion_type: str = "scaled_sigmoid",
        random_deletion_probability: float = 0.5,
        fixed_deletion_amount: float = 0.5,
        use_gumbel_noise: bool = False,
        bypass_gate: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigmoid_mask_scale = sigmoid_mask_scale
        self.gate_layer_norm = gate_layer_norm
        self.deletion_threshold = deletion_threshold
        self.delete_gate_layer = delete_gate_layer
        self.use_softmax1 = use_softmax1
        self.deletion_type = deletion_type
        self.random_deletion_probability = random_deletion_probability
        self.fixed_deletion_amount = fixed_deletion_amount
        self.use_gumbel_noise = use_gumbel_noise
        self.bypass_gate = bypass_gate
