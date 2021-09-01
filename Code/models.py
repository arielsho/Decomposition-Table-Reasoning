import os
import pdb
import torch
import torch.nn as nn
from transformers import TapasConfig, TapasForSequenceClassification
from attention_sigmoid import AttentionLayer_sigmoid

class DoubleBERT(nn.Module):
    def __init__(self, args):
        super(DoubleBERT, self).__init__()
        self.args = args

        # BERT
        self.load_dir = args.bert_model

        # modules
        self.Tapas = TapasForSequenceClassification.from_pretrained(args.model)
        self.Bert_evd = TapasForSequenceClassification.from_pretrained(args.bert_model)

        if args.load_tapas_model:
            self.Tapas.load_state_dict(torch.load(args.load_tapas_model))

        self.attention_layer_sigmoid = AttentionLayer_sigmoid(self.args)
        self.out_proj = nn.Linear(1536, 2, bias=True)

        # to cuda
        self.Tapas = self.Tapas.cuda()
        self.Bert_evd = self.Bert_evd.cuda()
        self.attention_layer_sigmoid = self.attention_layer_sigmoid.cuda()
        self.out_proj = self.out_proj.cuda()

    def forward(self, input_ids, attention_mask, token_type_ids, labels,
                input_ids_evd, input_mask_evd, segment_ids_evd, cls_ids_evd, cls_mask_evd):

        outputs_tapas = self.Tapas(input_ids=input_ids, token_type_ids=token_type_ids,
                                   attention_mask=attention_mask, labels=None,
                                   output_hidden_states=True)
        last_layer_tapas = outputs_tapas.hidden_states[-1]
        query = last_layer_tapas[:, 0]

        # Bert
        new_segment_ids_evd = segment_ids_evd.unsqueeze(dim=-1).repeat(1,1,7)
        outputs_bert = self.Bert_evd(input_ids=input_ids_evd, token_type_ids=new_segment_ids_evd,
                                 attention_mask=input_mask_evd, labels=None,
                                 output_hidden_states=True)

        last_layer_bert = outputs_bert.hidden_states[-1]

        memory = last_layer_bert[torch.arange(last_layer_bert.size(0)).unsqueeze(1), cls_ids_evd]

        # Attn Layer
        o = self.attention_layer_sigmoid(query, memory).view(-1, self.args.mem_dim)
        logits = self.out_proj(torch.cat((query, o), dim=1))

        return logits
