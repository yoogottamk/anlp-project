import torch
from torch import nn

from anlp_project.models import utils


def CopyParams(base_layer, target_layer, decoder=False):
    if target_layer.self_attn.qkv_same_dim:
        target_layer.self_attn.in_proj_weight = base_layer.self_attn.in_proj_weight
    else:
        target_layer.self_attn.k_proj_weight = base_layer.self_attn.k_proj_weight
        target_layer.self_attn.v_proj_weight = base_layer.self_attn.v_proj_weight
        target_layer.self_attn.q_proj_weight = base_layer.self_attn.q_proj_weight
    if base_layer.self_attn.bias_k is not None:
        target_layer.self_attn.bias_k = base_layer.self_attn.bias_k
    if base_layer.self_attn.bias_v is not None:
        target_layer.self_attn.bias_v = base_layer.self_attn.bias_v
    if target_layer.self_attn.in_proj_bias is not None:
        target_layer.self_attn.in_proj_bias = base_layer.self_attn.in_proj_bias
    target_layer.self_attn.out_proj = base_layer.self_attn.out_proj
    target_layer.fc1 = base_layer.fc1
    target_layer.fc2 = base_layer.fc2
    target_layer.self_attn_layer_norm = base_layer.self_attn_layer_norm
    target_layer.final_layer_norm = base_layer.final_layer_norm
    if decoder:
        if target_layer.encoder_attn.qkv_same_dim:
            target_layer.encoder_attn.in_proj_weight = (
                base_layer.encoder_attn.in_proj_weight
            )
        else:
            target_layer.encoder_attn.k_proj_weight = (
                base_layer.encoder_attn.k_proj_weight
            )
            target_layer.encoder_attn.v_proj_weight = (
                base_layer.encoder_attn.v_proj_weight
            )
            target_layer.encoder_attn.q_proj_weight = (
                base_layer.encoder_attn.q_proj_weight
            )
        target_layer.encoder_attn.out_proj = base_layer.encoder_attn.out_proj
        if base_layer.encoder_attn.bias_k is not None:
            target_layer.encoder_attn.bias_k = base_layer.encoder_attn.bias_k
        if base_layer.encoder_attn.bias_v is not None:
            target_layer.encoder_attn.bias_v = base_layer.encoder_attn.bias_v
        if target_layer.encoder_attn.in_proj_bias is not None:
            target_layer.encoder_attn.in_proj_bias = base_layer.self_attn.in_proj_bias
        target_layer.encoder_attn_layer_norm = base_layer.encoder_attn_layer_norm
    return target_layer


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True


class FairseqIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders.

    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.

    The :class:`FairseqIncrementalDecoder` interface also defines the
    :func:`reorder_incremental_state` method, which is used during beam search
    to select and reorder the incremental state based on the selection of beams.

    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs
    ):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        seen = set()

        def apply_reorder_incremental_state(module):
            if (
                module != self
                and hasattr(module, "reorder_incremental_state")
                and module not in seen
            ):
                seen.add(module)
                module.reorder_incremental_state(incremental_state, new_order)

        self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, "_beam_size", -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if (
                    module != self
                    and hasattr(module, "set_beam_size")
                    and module not in seen
                ):
                    seen.add(module)
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size
