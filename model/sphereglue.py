from copy import deepcopy
from typing import List, Tuple

import torch
from torch import nn
from torch_geometric.nn import ChebConv, knn_graph

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.kpt_encoder = MLP([feature_dim + 4] + layers + [feature_dim])
        nn.init.constant_(self.kpt_encoder[-1].bias, 0.0)

    def forward(self, descriptor, kpts, scores):
        inputs = [descriptor.transpose(1,2), kpts.transpose(1,2), scores.unsqueeze(1)]
        out = self.kpt_encoder(torch.cat(inputs, dim=1))
        return out


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SphericalChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, aggr):
        super().__init__()
        self.conv1 = ChebConv(in_channels, out_channels, K=K, aggr=aggr)
        

    def forward(self, x, position, knn):
        edges = knn_graph(position.squeeze(0), k=knn, flow= 'target_to_source')
        x = self.conv1(x.transpose(2,1), edges)
        return x.transpose(2,1)

class SphereGlue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.keypoint_encoder = [self.config['descriptor_dim']//4, self.config['descriptor_dim']//2, self.config['descriptor_dim'], self.config['descriptor_dim']*2]

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.keypoint_encoder)

        self.chebconv = SphericalChebConv(
            in_channels=self.config['descriptor_dim'], out_channels=self.config['output_dim'], K=self.config['K'], aggr=self.config['aggr'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['output_dim'], layer_names=self.config['GNN_layers']*9)

        self.final_proj = nn.Conv1d(
            self.config['output_dim'], self.config['output_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        

    def forward(self, data):
        """Run ChebGlue on a pair of keypoints and descriptors"""
        desc1, desc2 = data['h1'], data['h2']
        kpts1, kpts2 = data['unitCartesian1'], data['unitCartesian2']
        scores1, scores2 = data['scores1'], data['scores2']

        if kpts1.shape[1] == 0 or kpts2.shape[1] == 0:  # no keypoints
            shape1, shape2 = kpts1.shape[:-1], kpts2.shape[:-1]
            return {
                'matches0': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matches1': kpts2.new_full(shape2, -1, dtype=torch.int),
                'matching_scores0': kpts1.new_zeros(shape1),
                'matching_scores1': kpts2.new_zeros(shape2),
            }
        
        if kpts1.shape[1] >= self.config['max_kpts'] or kpts2.shape[1] >= self.config['max_kpts']:  
            kpts1, kpts2 = kpts1[:, :self.config['max_kpts'], :], kpts2[:, :self.config['max_kpts'], :]
            desc1, desc2 = desc1[:, :self.config['max_kpts'], :], desc2[:, :self.config['max_kpts'], :]
            scores1, scores2 = scores1[:, :self.config['max_kpts']], scores2[:, :self.config['max_kpts']]

        # Keypoint MLP encoder.
        desc1 = self.kenc(desc1, kpts1, scores1) 
        desc2 = self.kenc(desc2, kpts2, scores2) 

        # Chebyshev convolution
        desc1 = self.chebconv(desc1, kpts1, self.config['knn'])
        desc2 = self.chebconv(desc2, kpts2, self.config['knn'])
        
        # Multi-layer Transformer network.
        desc1, desc2 = self.gnn(desc1, desc2)

        # Final MLP projection.
        mdesc1, mdesc2 = self.final_proj(desc1), self.final_proj(desc2)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc1, mdesc2)
        scores = scores / self.config['output_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        max1, max2 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices1, indices2 = max1.indices, max2.indices
        mutual1 = arange_like(indices1, 1)[None] == indices2.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores1 = torch.where(mutual1, max1.values.exp(), zero)
        valid1 = mutual1 & (mscores1 > self.config['match_threshold'])
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'context_descriptors0': mdesc1,
            'context_descriptors1': mdesc2,
            'scores': scores,
            'matches0': indices1,  # use -1 for invalid match
            'matching_scores0': mscores1,
        }

