from torch import nn
import torch


class EventDetection(nn.Module):

    def __init__(self, encoder, num_event, input_size=1024):
        super().__init__()

        self.input_size = input_size
        self.encoder = encoder
        self.fc = nn.Linear(input_size, num_event)

    def forward(self, input_ids, segment_ids, attn_masks):
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=segment_ids)[0]
        cls_repsentation = hidden_state[:, 0, :]
        out = self.fc(cls_repsentation)
        logits = torch.sigmoid(out)
        # logits = logits*segment_ids
        return logits


class Argument_Extraction(nn.Module):
    """multi query mrc"""

    def __init__(self, encoder, input_size):
        super().__init__()

        self.encoder = encoder
        self.input_size = input_size
        self.fc = nn.Linear(input_size, 2)
        # self.att_merge = AttentionMerge(input_size,768,0.1)

    def forward(self, input_ids, token_type_ids, attn_masks):
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids)[0]

        logits = self.fc(hidden_state)
        mask = token_type_ids.unsqueeze(-1)
        mask = mask.expand(logits.shape)
        logits = torch.sigmoid(logits) * mask.float()

        return logits
