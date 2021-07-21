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
        return logits
