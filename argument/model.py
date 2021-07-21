from torch import nn
import torch

class Argument_Extraction(nn.Module):
    """multi query mrc"""

    def __init__(self, encoder, input_size):
        super().__init__()

        self.encoder = encoder
        self.input_size = input_size
        self.fc = nn.Linear(input_size, 2)
        # self.att_merge = AttentionMerge(input_size,768,0.1)

    def forward(self, input_ids, token_type_ids, attn_masks):
        output = self.encoder(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids)
        hidden_state = output[0]
        logits = self.fc(hidden_state)
        mask = token_type_ids.unsqueeze(-1)
        mask = mask.expand(logits.shape)
        logits = torch.sigmoid(logits) * mask.float()

        return logits
