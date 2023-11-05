import torch
import torch.nn as nn


class FeedForwardClassifier(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.output_layer = nn.Linear(self.hidden_size, 2, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()

        self.linear.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        torch.nn.init.xavier_uniform_(self.output_layer.weight.data)

    def compute_output(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        encoded = torch.mean(hidden_states, dim=1)

        ff = self.linear(encoded)
        ff = self.gelu(ff)

        output = self.output_layer(ff)

        return output

    def forward(self, mode, inputs):
        if not self.args.predict:
            (input_ids, attention_mask, token_type_ids, labels) = inputs

            input_ids = input_ids.long()
            attention_mask = attention_mask.long()
            token_type_ids = token_type_ids.long()
            labels = labels.long()

        if self.args.predict:
            (input_ids, attention_mask, token_type_ids) = inputs

            input_ids = input_ids.long()
            attention_mask = attention_mask.long()
            token_type_ids = token_type_ids.long()

        # short_input_ids = []
        # short_attention_mask = []
        # short_token_type_ids = []
        #
        # long_input_ids = []
        # long_attention_mask = []
        # long_token_type_ids = []
        #
        # order = []

        # for i in range(input_ids.size(0)):
        #     if attention_mask[i].count_nonzero(dim=0) <= 512:
        #         short_input_ids.append(input_ids[i])
        #         short_attention_mask.append(attention_mask[i])
        #         short_token_type_ids.append(token_type_ids[i])
        #         order.append(i)
        #
        # for i in range(input_ids.size(0)):
        #     if attention_mask[i].count_nonzero(dim=0) > 512:
        #         long_input_ids.append(input_ids[i])
        #         long_attention_mask.append(attention_mask[i])  # Fixed typo here
        #         long_token_type_ids.append(token_type_ids[i])
        #         order.append(i)
        #
        # # this is for short
        # short_input_ids = torch.cat([x[:512].unsqueeze(0) for x in short_input_ids],dim=0)
        # short_attention_mask = torch.cat([x[:512].unsqueeze(0) for x in short_attention_mask],dim=0)
        # short_token_type_ids = torch.cat([x[:512].unsqueeze(0) for x in short_token_type_ids],dim=0)
        #
        # short_output = self.compute_output(short_input_ids, short_attention_mask, short_token_type_ids)
        #
        # # this is for long
        # if long_input_ids:
        #     long_input_ids = list(torch.cat([x.unsqueeze(0) for x in long_input_ids], dim=0).split(512, dim=1))
        #     long_attention_mask = list(
        #         torch.cat([x.unsqueeze(0) for x in long_attention_mask], dim=0).split(512, dim=1))
        #     long_token_type_ids = list(
        #         torch.cat([x.unsqueeze(0) for x in long_token_type_ids], dim=0).split(512, dim=1))
        #
        #     final_long_output = None
        #     for i in range(len(long_input_ids)):
        #         input_ids = long_input_ids[i]
        #         attention_mask = long_attention_mask[i]
        #         token_type_ids = long_attention_mask[i]
        #
        #         output = self.compute_output(input_ids, attention_mask, token_type_ids)
        #
        #         if final_long_output is None:
        #             final_long_output = output
        #         else:
        #             for j in range(output.size(0)):
        #                 if output[j][1] > output[j][0]:
        #                     final_long_output[j] = output[j]
        # else:
        #     final_long_output = torch.tensor([])  # or other appropriate tensor

        # combine short and long outputs
        # all_outputs = list(output.split(1, dim=0))
        # if long_output.nelement() != 0:  # Check if final_long_output is not empty
        #     all_outputs += list(final_long_output.split(1, dim=0))
        # output = []
        # for i in order:
        #     output.append(all_outputs[i])
        # output = torch.cat(output, dim=0)

        # if mode == "train":
        #
        #     loss = self.loss_fct(output, labels)
        #
        #     return loss, output
        if mode == "eval":
            output = self.compute_output(input_ids, attention_mask, token_type_ids)
            prediction = torch.argmax(output, dim=1)
            print(prediction)
            return prediction
