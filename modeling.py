from transformers import PreTrainedModel, RobertaModel, RobertaConfig
import torch
import torch.nn as nn
BertLayerNorm = torch.nn.LayerNorm

import code

class DualRobertaForDotProduct(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size//2),
            nn.ReLU(),
            nn.Linear(config.hidden_size//2, 1),
        )
        self.init_weights()

    def forward(
        self,
        user_input_ids=None,
        user_attention_mask=None,
        user_token_type_ids=None,
        user_position_ids=None,
        user_head_mask=None,
        user_inputs_embeds=None,
        user_output_attentions=None,
        user_output_hidden_states=None,
        user_return_dict=None,
        item_input_ids=None,
        item_attention_mask=None,
        item_token_type_ids=None,
        item_position_ids=None,
        item_head_mask=None,
        item_inputs_embeds=None,
        labels=None,
        item_output_attentions=None,
        item_output_hidden_states=None,
        item_return_dict=None,
    ):
        user_outputs = self.roberta(
            user_input_ids,
            attention_mask=user_attention_mask,
            token_type_ids=user_token_type_ids,
            position_ids=user_position_ids,
            head_mask=user_head_mask,
            inputs_embeds=user_inputs_embeds,
            output_attentions=user_output_attentions,
            output_hidden_states=user_output_hidden_states,
        )
        item_outputs = self.roberta(
            item_input_ids,
            attention_mask=item_attention_mask,
            token_type_ids=item_token_type_ids,
            position_ids=item_position_ids,
            head_mask=item_head_mask,
            inputs_embeds=item_inputs_embeds,
            output_attentions=item_output_attentions,
            output_hidden_states=item_output_hidden_states,
        )

        user_rep = user_outputs[1]
        item_rep = item_outputs[1]
        # score = torch.sum(user_rep * item_rep, -1)
        score = self.fc(torch.cat([user_rep, item_rep], -1)).view(-1)

        # code.interact(local=locals())
        loss = torch.nn.functional.mse_loss(score, labels.float(), reduction="mean")
        reg = torch.sum(user_rep*user_rep, -1).mean() + torch.sum(item_rep*item_rep, -1).mean()
        loss = loss + reg

        # code.interact(local=locals())

        return (loss, score)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

