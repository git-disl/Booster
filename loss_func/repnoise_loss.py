import torch
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def register_activation_hook(model):
    activations = {}
    hooks = []
    for name, param in model.named_modules():
        param.name = name
        def _hook(module, __, val):
            activations[module.name] = val
            # print(module.name)
        hooks += [param.register_forward_hook(_hook)]
        
    return activations, hooks 


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
 
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
 
    def forward(self, source, target, xy_only=False):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss



def masked_token_ce_loss(
    logits,
    labels,
    mask
):
    # # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # # apply mask
    # # shifted masks
    shift_logit_mask = mask[..., :-1].contiguous()
    expanded_mask = shift_logit_mask.unsqueeze(-1).expand(-1, -1, shift_logits.size(-1))
    shift_label_mask = mask[..., 1:].contiguous()
    shift_logits = shift_logits * expanded_mask
    shift_labels = shift_labels * shift_label_mask
    shift_labels[shift_labels == 0] = -100
    shift_labels = shift_labels.type(torch.LongTensor)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).to(DEVICE), shift_labels.view(-1).to(DEVICE))
    # loss_fct = torch.nn.CrossEntropyLoss()
    # loss = loss_fct(logits.view(-1, logits.size(-1)).to(DEVICE), labels.view(-1).to(DEVICE))
    return loss


def adapt_dimension_b2a(a,b):
    import torch.nn.functional as F
    import copy 
    b= copy.deepcopy(b)
    a= copy.deepcopy(a)
    # Handle the first dimension
    if b.shape[0] < a.shape[0]:
        padding_first_dim = (0, 0, 0, a.shape[0] - b.shape[0])
        b = F.pad(b, padding_first_dim, mode='constant', value=0)
    elif b.shape[0] > a.shape[0]:
        b = b[:a.shape[0], :]

    # Handle the second dimension
    if b.shape[1] < a.shape[1]:
        padding_second_dim = (0, a.shape[1] - b.shape[1])
        b = F.pad(b, padding_second_dim, mode='constant', value=0)
    elif b.shape[1] > a.shape[1]:
        b = b[:, :a.shape[1]]
    return b

def rep_noise_loss(
    model, harmful_batch, harmless_batch=None,
    beta=0.001, alpha=1
):
    """ Calculate the representation noise loss

    Args:
        model (Pytorch Model): The model to calculate the loss, i.e. an LLM loaded with Huggingface Transformers
        harmful_batch (Dataloader): the paired harmful batch
        harmless_batch (Dataloader): the paired harmless batch
        beta (float, optional): _description_. Defaults to 0.001.
        alpha (int, optional): _description_. Defaults to 1.
    """
    mmd_loss = MMD_loss()
    activations, hooks = register_activation_hook(model)
    harmful_outputs = model(harmful_batch['input_ids'], attention_mask=harmful_batch['attention_mask'], output_hidden_states=True)
    # harmful_outputs = model(harmful_batch['input_ids'], attention_mask=harmful_batch['attention_mask'])
    harmful_activations = []
    for i in range(len(model.base_model.model.model.layers)):
        harmful_activations.append(activations[f'base_model.model.model.layers.{i}.mlp'])
    # print(harmful_batch['input_ids'].shape)
    harmless_batch_truncate_or_pad =adapt_dimension_b2a(harmful_batch['input_ids'], harmless_batch['input_ids'] )
    mask = ~torch.eq(harmful_batch['input_ids'], harmless_batch_truncate_or_pad)
    noise_loss = 0
    for i, hidden in enumerate(harmful_activations):
        hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
        hiddens = hidden * hiddens_mask
        gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
        noise_loss += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
    noise_loss /= len(harmful_activations) # len(layer_idxs)

    mask = mask.to(DEVICE).to(torch.bfloat16)
    # print(mask.dtype)
    
    harmful_outputs_loss = masked_token_ce_loss(
        harmful_outputs.logits,
        harmful_batch['input_ids'],
        mask
    )
    for hook in hooks:
        hook.remove()
    hooks = []
    
    output_embeddings = model.get_output_embeddings()
    norm = model.base_model.model.model.norm

    harmful_losses = harmful_outputs_loss
    # mask = torch.ones_like(harmful_batch['input_ids'])
    # comment out this hidden state adversarial loss because of NaN
    for i, h in enumerate(harmful_outputs.hidden_states):
        # print(h.dtype)
        h = h.to(DEVICE).to(torch.bfloat16)
        out = output_embeddings(norm(h))
        loss = masked_token_ce_loss(
            out.to(DEVICE), harmful_batch['input_ids'].to(DEVICE),
            mask
        )
        harmful_losses += loss
    harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1
    
    
    
    harmful_batch_truncate_or_pad =adapt_dimension_b2a(harmless_batch['input_ids'], harmful_batch['input_ids'] )
    mask = ~torch.eq(harmless_batch['input_ids'], harmful_batch_truncate_or_pad)
    # mask = torch.ones_like(harmless_batch['input_ids'])
    harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'])
    harmless_outputs_loss = masked_token_ce_loss(
        harmless_outputs.logits,
        harmless_batch['input_ids'],
        mask
    )

    harmless_losses = harmless_outputs_loss

   
    
    # print(harmful_losses)
    # loss = harmless_losses + beta * noise_loss - alpha * torch.log(harmful_losses)
    loss = harmless_losses + beta * noise_loss - alpha * torch.log(harmful_losses)
    # loss = harmless_losses
    # print(noise_loss)
    # loss =  beta * noise_loss - alpha * torch.log(harmful_losses)
    return loss
