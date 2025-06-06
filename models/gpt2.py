import torch
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask


class GPT2Model(GPTPreTrainedModel):
  """
  The GPT model returns the final embeddings for each token in a sentence.

  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n GPT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """

  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # GPT-2 layers.
    self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    # Final layer norm.
    self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    inputs_embeds = None

    ### YOUR CODE HERE
    input_embeds = self.word_embedding(input_ids)

    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = None

    ### TODO: Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    ###       Then, add two embeddings together; then apply dropout and return.
    ### YOUR CODE HERE
    pos_embeds = self.pos_embedding(pos_ids)

    hidden_states = input_embeds + pos_embeds
    hidden_states = self.embed_dropout(hidden_states)

    return hidden_states


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.gpt_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of GPTLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    sequence_output = self.final_layer_norm(sequence_output)

    # Get the hidden state of the final token.
    last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last index
    last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

    return {'last_hidden_state': sequence_output, 'last_token': last_token}

  def hidden_state_to_token(self, hidden_state):
    """
    GPT-2 uses weight tying with the input word embeddings. The logits are the dot product between output hidden states
    and the word embedding weights:

      return hidden_state(s) * E^T
    """
    ### YOUR CODE HERE
    E = self.word_embedding.weight
    
    return torch.matmul(hidden_state, E.t())

  @classmethod
  def from_pretrained(cls, model_name: str = "gpt2"):
    # 1) Hugging Face GPT-2 모델과 config 가져오기
    hf_model = OpenAIGPT2Model.from_pretrained(model_name).eval()
    hf_cfg   = hf_model.config   # 여기엔 .n_embd, .n_layer, .n_head 등이 있다

    # 2) 우리 GPT2Config 에 "올바른 속성 이름"으로 채워 넣기
    our_cfg = GPT2Config(
      hidden_size = hf_cfg.n_embd,                     # n_embd를 hidden_size로
      num_hidden_layers   = hf_cfg.n_layer,            # n_layer
      num_attention_heads = hf_cfg.n_head,             # n_head
      intermediate_size   = hf_cfg.n_embd * 3,          # (원래는 4×지만, 예제에선 3×)
      hidden_dropout_prob = hf_cfg.resid_pdrop,
      layer_norm_eps      = hf_cfg.layer_norm_epsilon,
      vocab_size          = hf_cfg.vocab_size,
      max_position_embeddings = hf_cfg.n_positions,
      pad_token_id        = hf_cfg.eos_token_id,
    )

    # 3) LoRA가 포함된 구조로 모델 생성 (랜덤 초기화 상태)
    our_model = GPT2Model(our_cfg).eval()

    # 4) Embedding weight 덮어쓰기
    our_model.word_embedding.load_state_dict(
      hf_model.get_input_embeddings().state_dict()
    )
    # fMJK
    # our_model.pos_embedding.load_state_dict(
    #   hf_model.get_position_embeddings().state_dict()
    # )
   # GPT2Model 내부에서 positional embedding은 transformer.wpe 에 있음
    our_model.pos_embedding.load_state_dict(
      hf_model.wpe.state_dict()
    )
    # 5) Transformer 레이어별 weight 복사: 'hf_cfg.n_layer'만큼 순회
    for i in range(hf_cfg.n_layer):
      our_layer = our_model.gpt_layers[i]
      hf_layer  = hf_model.h[i]

      # 5-a) c_attn (Q/K/V 묶음) → Query/Key/Value로 나눠서 복사
      c_attn_w = hf_layer.attn.c_attn.weight.data   # [n_embd, 3*n_embd]
      c_attn_b = hf_layer.attn.c_attn.bias.data     # [3*n_embd]
      d = hf_cfg.n_embd

      # Query
      our_layer.self_attention.query.weight.data.copy_(c_attn_w[:, :d].T)
      our_layer.self_attention.query.bias.data.copy_(c_attn_b[:d])
      # Key
      our_layer.self_attention.key.weight.data.copy_(c_attn_w[:, d:2*d].T)
      our_layer.self_attention.key.bias.data.copy_(c_attn_b[d:2*d])
      # Value
      our_layer.self_attention.value.weight.data.copy_(c_attn_w[:, 2*d:3*d].T)
      our_layer.self_attention.value.bias.data.copy_(c_attn_b[2*d:3*d])

      # 5-b) attention_dense (c_proj)
      our_layer.attention_dense.weight.data.copy_(hf_layer.attn.c_proj.weight.data.T)
      our_layer.attention_dense.bias.data.copy_(hf_layer.attn.c_proj.bias.data)

      # 5-c) attention LayerNorm (ln_1)
      our_layer.attention_layer_norm.weight.data.copy_(hf_layer.ln_1.weight.data)
      our_layer.attention_layer_norm.bias.data.copy_(hf_layer.ln_1.bias.data)

      # 5-d) MLP (c_fc + c_proj)
      our_layer.interm_dense.weight.data.copy_(hf_layer.mlp.c_fc.weight.data.T)
      our_layer.interm_dense.bias.data.copy_(hf_layer.mlp.c_fc.bias.data)
      our_layer.out_dense.weight.data.copy_(hf_layer.mlp.c_proj.weight.data.T)
      our_layer.out_dense.bias.data.copy_(hf_layer.mlp.c_proj.bias.data)

      # 5-e) Feed-forward LayerNorm (ln_2)
      our_layer.out_layer_norm.weight.data.copy_(hf_layer.ln_2.weight.data)
      our_layer.out_layer_norm.bias.data.copy_(hf_layer.ln_2.bias.data)

    # 6) 마지막 final LayerNorm (ln_f)
    our_model.final_layer_norm.weight.data.copy_(hf_model.ln_f.weight.data)
    our_model.final_layer_norm.bias.data.copy_(hf_model.ln_f.bias.data)

    return our_model