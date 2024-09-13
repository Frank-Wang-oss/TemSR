import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from einops import rearrange
from torch.autograd import Function

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, device, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss

# def EntropyLoss(input_):
#     mask = input_.ge(0.0000001)
#     mask_out = torch.masked_select(input_, mask)
#     entropy = - (torch.sum(mask_out * torch.log(mask_out)))
#     return entropy / float(input_.size(0))
#
# def EntropyLoss_single (input_):
#     mask = input_.ge(0.0000001)
#     mask_out = torch.masked_select(input_, mask)
#     entropy = - ((mask_out * torch.log(mask_out)))
#     return entropy

# def EntropyLoss(input_):
#     input_ = nn.Softmax(dim=1)(input_)
#     mask = input_.ge(0.0000001)
#     mask_out = torch.masked_select(input_, mask)
#     entropy = - (torch.sum(mask_out * torch.log(mask_out)))
#     return entropy / float(input_.size(0))

# def EntropyLoss_single(input_):
#     input_ = nn.Softmax(dim=1)(input_)
#     mask = input_.ge(0.0000001)
#     mask_out = torch.masked_select(input_, mask)
#     out = mask_out * torch.log(mask_out)
#     print(mask_out.shape)
#     entropy = - (torch.sum(mask_out * torch.log(mask_out),-1))
#     return entropy


def EntropyLoss(input_):
    input_ = nn.Softmax(dim=1)(input_)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return torch.mean(entropy)
def EntropyLoss_single(input_):
    input_ = nn.Softmax(dim=1)(input_)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy
def SKL_loss(out1, out2):
    out2_t = out2.clone()
    out2_t = out2_t.detach()
    out1_t = out1.clone()
    out1_t = out1_t.detach()
    return (F.kl_div(F.log_softmax(out1), out2_t, reduction='none') +
            F.kl_div(F.log_softmax(out2), out1_t, reduction='none')) / 2
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def Temporal_EntropyLoss_v1(Recovered_inputs, bs):
    ## recovered_inputs are list, each having (n*bs, out)
    ## n represents how many segments are used ti capture temporal dependency

    entropy_single = EntropyLoss_single(Recovered_inputs)
    entropy_single = rearrange(entropy_single, '(b n) -> b n',b=bs)
    entropy_1 = entropy_single.unsqueeze(1)
    entropy_2 = entropy_single.unsqueeze(2)

    entropy_diff = torch.abs(entropy_1 - entropy_2)


    entropy_src_like = EntropyLoss(Recovered_inputs)

    loss_entropy_diff = torch.mean(entropy_diff)

    return entropy_src_like + 0.1*loss_entropy_diff


def Temporal_EntropyLoss_v2(Recovered_inputs, bs): ### Use only the full data for entropy computation
    ## recovered_inputs are list, each having (n*bs, out)
    ## n represents how many segments are used ti capture temporal dependency

    entropy_single = EntropyLoss_single(Recovered_inputs)
    entropy_single = rearrange(entropy_single, '(b n) -> b n',b=bs)
    entropy_1 = entropy_single.unsqueeze(1)
    entropy_2 = entropy_single.unsqueeze(2)

    entropy_diff = torch.abs(entropy_1 - entropy_2)


    entropy_src_like = EntropyLoss(Recovered_inputs)
    return torch.mean(entropy_diff), entropy_src_like
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1- (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()

    return loss

def temporal_consistency_loss(recovered_data):
    loss_fn = nn.MSELoss()
    target = recovered_data[:, 1:]  # Predict next time steps
    prediction = recovered_data[:, :-1]
    loss = loss_fn(prediction, target)
    return loss

def Eu_distance(sample1, sample2):
    ## input size (bs, N, L)
    sample1 = rearrange(sample1,'b N L -> b (N L)', b = sample1.size(0))

    sample2 = rearrange(sample2,'b N L -> b (N L)', b = sample2.size(0))
    diff = torch.norm(sample1 - sample2, dim=-1)
    return diff



def Bank_info_max_anchor_min(model, original_data, recovered_data, recovered_data_feat, entropy_bank, sample_bank,
                               classifier, percent, temperature, detach):
    # model.eval()

    num_samples = recovered_data.size(0)

    # Compute number of samples with lowest entropy values for anchor
    num_low_entropy_samples = int(num_samples * percent)
    num_low_entropy_samples = max(1, num_low_entropy_samples)

    recovered_data_pred = classifier(recovered_data_feat)
    recovered_data_entropy = EntropyLoss_single(recovered_data_pred)

    # Select lowest entropy samples
    _, low_indices = torch.topk(entropy_bank, num_low_entropy_samples, largest=False)
    anchor_recovered_samples = sample_bank[low_indices]
    anchor_sample = anchor_recovered_samples.mean(dim=0, keepdim=True)

    # Obtain features from the model
    _, anchor_sample_feat = model(anchor_sample)
    _, original_data_feat = model(original_data)
    # _, recovered_data_feat = model(recovered_data)  # Already provided as an argument

    # Normalize features
    anchor_sample_feat = F.normalize(anchor_sample_feat, dim=-1)
    original_data_feat = F.normalize(original_data_feat, dim=-1)
    recovered_data_feat = F.normalize(recovered_data_feat, dim=-1)

    # Compute similarities (positive and negative)
    positive_similarity = torch.mm(recovered_data_feat, anchor_sample_feat.t()) / temperature

    negative_similarity_original = torch.mm(recovered_data_feat, original_data_feat.t()) / temperature
    negative_similarity_recovered = torch.mm(recovered_data_feat, recovered_data_feat.t()) / temperature

    # Mask the diagonal to avoid comparing the same samples
    mask = torch.eye(num_samples, device=negative_similarity_original.device).bool()

    negative_similarity_original = negative_similarity_original.masked_select(mask).view(num_samples, 1)

    negative_similarity_recovered = negative_similarity_recovered.masked_select(~mask).view(num_samples,
                                                                                            num_samples - 1)
    # Combine negative similarities
    negative_similarity = torch.cat([negative_similarity_original, negative_similarity_recovered], dim=1)

    # Label for cross-entropy (all positives are first)
    labels = torch.zeros(num_samples, dtype=torch.long, device=recovered_data_feat.device)

    # Combine positive and negative similarities
    logits = torch.cat([positive_similarity, negative_similarity], dim=1)

    log_probs = F.log_softmax(logits, dim=1)

    # Gather the log-probabilities corresponding to the correct class
    log_probs_for_labels = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)

    # Compute the weighted loss
    if detach:
        recovered_data_entropy = recovered_data_entropy.detach()
    weighted_loss = - (recovered_data_entropy * log_probs_for_labels).mean()

    # Return the total loss and additional metrics
    return weighted_loss, recovered_data_entropy


def Local_info_max_anchor_min(model, original_data, recovered_data, recovered_data_feat,
                               classifier, percent, temperature, detach):
    # model.eval()

    num_samples = recovered_data.size(0)

    # Compute number of samples with lowest entropy values for anchor
    num_low_entropy_samples = int(num_samples * percent)
    num_low_entropy_samples = max(1, num_low_entropy_samples)

    recovered_data_pred = classifier(recovered_data_feat)
    recovered_data_entropy = EntropyLoss_single(recovered_data_pred)

    # Select lowest entropy samples
    _, low_indices = torch.topk(recovered_data_entropy, num_low_entropy_samples, largest=False)
    anchor_sample = recovered_data[low_indices].mean(dim=0,keepdim=True)

    # Obtain features from the model
    _, anchor_sample_feat = model(anchor_sample)
    _, original_data_feat = model(original_data)
    # _, recovered_data_feat = model(recovered_data)  # Already provided as an argument

    # Normalize features
    anchor_sample_feat = F.normalize(anchor_sample_feat, dim=-1)
    original_data_feat = F.normalize(original_data_feat, dim=-1)
    recovered_data_feat = F.normalize(recovered_data_feat, dim=-1)

    # Compute similarities (positive and negative)
    positive_similarity = torch.mm(recovered_data_feat, anchor_sample_feat.t()) / temperature

    negative_similarity_original = torch.mm(recovered_data_feat, original_data_feat.t()) / temperature
    negative_similarity_recovered = torch.mm(recovered_data_feat, recovered_data_feat.t()) / temperature

    # Mask the diagonal to avoid comparing the same samples
    mask = torch.eye(num_samples, device=negative_similarity_original.device).bool()

    negative_similarity_original = negative_similarity_original.masked_select(mask).view(num_samples, 1)

    negative_similarity_recovered = negative_similarity_recovered.masked_select(~mask).view(num_samples,
                                                                                            num_samples - 1)
    # Combine negative similarities
    negative_similarity = torch.cat([negative_similarity_original, negative_similarity_recovered], dim=1)

    # Label for cross-entropy (all positives are first)
    labels = torch.zeros(num_samples, dtype=torch.long, device=recovered_data_feat.device)

    # Combine positive and negative similarities
    logits = torch.cat([positive_similarity, negative_similarity], dim=1)

    log_probs = F.log_softmax(logits, dim=1)

    # Gather the log-probabilities corresponding to the correct class
    log_probs_for_labels = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)

    # Compute the weighted loss
    if detach:
        recovered_data_entropy = recovered_data_entropy.detach()
    weighted_loss = - (recovered_data_entropy * log_probs_for_labels).mean()

    # Return the total loss and additional metrics
    return weighted_loss, recovered_data_entropy
def info_max(model, original_data, recovered_data, recovered_data_feat):
    # model.eval()

    num_samples = recovered_data.size(0)


    # Obtain features from the model
    _, original_data_feat = model(original_data)
    # _, recovered_data_feat = model(recovered_data)  # Already provided as an argument

    # Normalize features
    original_data_feat = F.normalize(original_data_feat, dim=-1)
    recovered_data_feat = F.normalize(recovered_data_feat, dim=-1)

    # Compute similarities (positive and negative)

    negative_similarity_original = torch.mm(recovered_data_feat, original_data_feat.t())
    negative_similarity_recovered = torch.mm(recovered_data_feat, recovered_data_feat.t())

    # Mask the diagonal to avoid comparing the same samples
    mask = torch.eye(num_samples, device=negative_similarity_original.device).bool()

    negative_similarity_original = negative_similarity_original.masked_select(mask).view(num_samples, 1)

    negative_similarity_recovered = negative_similarity_recovered.masked_select(~mask).view(num_samples,
                                                                                            num_samples - 1)
    # Combine negative similarities
    negative_similarity = torch.cat([negative_similarity_original, negative_similarity_recovered], dim=1)

    # Label for cross-entropy (all positives are first)

    loss = negative_similarity.mean()

    # Return the total loss and additional metrics
    return loss
def CL_recovered_original(x_a, x_p, x_n, model, temperature=0.07):

    _, x_n = model(x_n)
    x_p = rearrange(x_p, '(b N) L -> b N L', b=x_a.size(0))

    bs, num_positives, dim = x_p.shape

    # Normalize all vectors to ensure cosine similarity is within the correct range
    x_a = F.normalize(x_a, dim=-1)
    x_p = F.normalize(x_p, dim=-1)
    x_n = F.normalize(x_n, dim=-1)

    # Compute similarity between anchor and positive samples
    pos_sim = torch.bmm(x_p, x_a.unsqueeze(-1)).squeeze(-1) / temperature  # (bs, 3)
    # print(pos_sim)

    # Compute similarity between anchor and negative samples
    neg_sim = torch.mm(x_a, x_n.t()) / temperature  # (bs, bs)

    # Concatenate positive and negative similarities
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # (bs, 3 + bs)

    # Apply softmax to the similarity matrix
    softmax_sim = F.log_softmax(logits, dim=1)  # Log-Softmax

    # Calculate the InfoNCE loss
    infonce_loss = -softmax_sim[:, :3].mean()  # Only consider positive samples' log-probabilities

    return infonce_loss, pos_sim.mean(), neg_sim.mean()


def CL_recovered_original_v2(x_a, x_p, x_n, model, temperature=0.07):

    _, x_n = model(x_n)
    x_p = rearrange(x_p, '(b N) L -> b N L', b=x_a.size(0))

    bs, num_positives, dim = x_p.shape

    # Normalize all vectors to ensure cosine similarity is within the correct range
    x_a = F.normalize(x_a, dim=-1)
    x_p = F.normalize(x_p, dim=-1)
    x_n = F.normalize(x_n, dim=-1)

    # Compute similarity between anchor and positive samples
    pos_sim = torch.bmm(x_p, x_a.unsqueeze(-1)).squeeze(-1) / temperature  # (bs, 3)
    # print(pos_sim)

    # Compute similarity between anchor and negative samples
    neg_sim = torch.mm(x_a, x_n.t()) / temperature  # (bs, bs)
    neg_sim = torch.diag(neg_sim).unsqueeze(dim=1)
    # Concatenate positive and negative similarities
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # (bs, 3 + bs)

    # Apply softmax to the similarity matrix
    softmax_sim = F.log_softmax(logits, dim=1)  # Log-Softmax

    # Calculate the InfoNCE loss
    infonce_loss = -softmax_sim[:, :3].mean()  # Only consider positive samples' log-probabilities

    return infonce_loss, pos_sim.mean(), neg_sim.mean()


def CL_recovered_original_v3(x_a, x_p, x_n, model):

    _, x_n = model(x_n)
    x_p = rearrange(x_p, '(b N) L -> b N L', b=x_a.size(0))

    bs, num_positives, dim = x_p.shape

    # Normalize all vectors to ensure cosine similarity is within the correct range
    x_a = F.normalize(x_a, dim=-1)
    x_p = F.normalize(x_p, dim=-1)
    x_n = F.normalize(x_n, dim=-1)



    distances_to_pos = torch.norm(x_a.unsqueeze(1) - x_p, dim=-1).mean(-1)

    # Compute distances to original samples
    distances_to_neg = torch.norm(x_a - x_n, dim=-1)
    # Compute loss for maximizing distances to original samples and minimizing distances to anchor samples
    # Use log to moderate the differences between the two distances.

    total_loss = torch.log((distances_to_pos+1e-5)/(distances_to_neg+1e-5)+1).mean()


    return total_loss, distances_to_pos.mean(), distances_to_neg.mean()


def CL_recovered_original_v4(x_a, x_p, x_n, model, temperature):


    x_p = rearrange(x_p, '(b N) L -> b N L', b=x_a.size(0))

    x = torch.cat((x_a.unsqueeze(1), x_p),dim=1)

    bs, num_positive, f = x.shape
    x = F.normalize(x, dim=-1)

    # Compute positive pairs (bs, num_positive, f) * (bs, f, num_positive) -> (bs, num_positive, num_positive)
    positive_pairs = torch.matmul(x, x.transpose(1, 2)) / temperature

    # Compute negative pairs
    _, x_n = model(x_n)
    x_n = F.normalize(x_n, dim=-1)

    n = x_n.unsqueeze(-1)  # Expand to (bs, f, 1)
    negative_pairs = torch.matmul(x, n).squeeze(-1) / temperature  # (bs, num_positive, 1) -> (bs, num_positive)

    # Reshape to concatenate positive and negative pairs
    positive_pairs = positive_pairs.view(bs * num_positive,
                                         num_positive)  # (bs, num_positive, num_positive) -> (bs*num_positive, num_positive)
    negative_pairs = negative_pairs.view(bs * num_positive, 1)  # (bs, num_positive) -> (bs*num_positive, 1)

    # Concatenate positive and negative pairs -> (bs*num_positive, num_positive + 1)
    logits = torch.cat([negative_pairs, positive_pairs], dim=1)

    # Labels: 0 means the first position is the positive pair
    labels = torch.zeros(bs * num_positive, dtype=torch.long, device=x.device)

    # Compute the InfoNCE loss
    loss = F.cross_entropy(logits, labels)

    return loss, positive_pairs.mean(), negative_pairs.mean()




class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)
        return loss


def kl_divergence_batch(source_batch, target_batch, epsilon=1e-6):
    """
    Compute KL divergence between source and target batch distributions.

    Args:
    - source_batch (torch.Tensor): Batch of source data of shape (bs, f).
    - target_batch (torch.Tensor): Batch of target data of shape (bs, f).
    - epsilon (float): Small value to avoid division by zero.

    Returns:
    - kl_div (torch.Tensor): KL divergence for each feature in the batch (bs,).
    """
    # Add a small epsilon to avoid log(0)
    f_dim = source_batch.size(1)
    if source_batch.size(0) == target_batch.size(0):
        source_batch = source_batch + epsilon
        target_batch = target_batch + epsilon
    else:
        source_batch = source_batch.mean(dim=0) + epsilon
        target_batch = target_batch.mean(dim=0) + epsilon
    # Compute KL divergence for each feature
    kl_div = F.kl_div(source_batch.log(), target_batch, reduction='batchmean')
    return kl_div / f_dim
class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class AdversarialLoss(nn.Module):
    '''
    Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
    '''

    def __init__(self, input_dim=16, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, **kwargs):
        super(AdversarialLoss, self).__init__()
        self.domain_classifier = Discriminator(input_dim)
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)

    def forward(self, source, target):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        source_loss = self.get_adversarial_result(source, True, lamb)
        target_loss = self.get_adversarial_result(target, False, lamb)
        adv_loss = 0.5 * (source_loss + target_loss)
        return adv_loss

    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        if source:
            domain_label = torch.ones(len(x), 1).long()
        else:
            domain_label = torch.zeros(len(x), 1).long()
        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
if __name__ == '__main__':
    # bs, num_channel, time_length = 3, 5, 8
    # x = torch.rand(bs, num_channel, time_length)

    bs, out_dim = 3, 2
    x = torch.rand(bs, out_dim)

    out = EntropyLoss(x)
    print(out)
    out = Entropy(x)
    print(out)
    out = EntropyLoss_single(x)
    print(out.size())