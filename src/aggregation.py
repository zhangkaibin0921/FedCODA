import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median

try:
    from sklearn.cluster import SpectralClustering
except ImportError:
    SpectralClustering = None 


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []
        
         
    def aggregate_updates(self, global_model, agent_updates_dict):


        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':    
            aggregated_updates = self.agg_avg(agent_updates_dict)

        elif self.args.aggr == 'alignins':
            aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'signguard':
            aggregated_updates = self.agg_signguard(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        elif self.args.aggr == 'fedcoda':
            aggregated_updates = self.agg_fedcoda(agent_updates_dict, global_model, cur_global_params)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict, neurotoxin_mask

    def agg_rfa(self, agent_updates_dict):
        local_updates = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)

        n = len(local_updates)
        temp_updates = torch.stack(local_updates, dim=0)
        weights = torch.ones(n).to(self.args.device)  
        gw = compute_geometric_median(local_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(temp_updates-gw, dim=1)))
            gw = compute_geometric_median(local_updates, weights).median

        aggregated_model = gw
        return aggregated_model

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_updates = [agent_updates_dict[i] for i in score_index]


        return sum(return_updates)/len(return_updates)

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        update = self.agg_avg(current_dict)

        return update
   
    def agg_foolsgold(self, agent_updates_dict):
        def foolsgold(updates):
            """
            :param updates:
            :return: compute similatiry and return weightings
            """
            n_clients = updates.shape[0]
            cs = smp.cosine_similarity(updates) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        client_updates = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        update_len = np.array(client_updates[0].shape).prod()
        # print("client_updates size", client_models[0].parameters())
        # update_len = len(client_updates)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, update_len))
        if len(names) < len(client_updates):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, update_len))
        updates = np.zeros((num_clients, update_len))

        for i in range(len(client_updates)):
            # updates[i] = np.reshape(client_updates[i][-2].cpu().data.numpy(), (update_len))
            updates[i] = np.reshape(client_updates[i], (update_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += updates[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(updates[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += updates
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(updates)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        print(len(client_updates), len(wv))


        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model


    def agg_fedcoda(self, agent_updates_dict, global_model, flat_global_model):
        # ========== 第一部分：FedCODA z-score筛选 ==========
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        client_ids = malicious_id + benign_id
        num_clients = len(client_ids)
        if num_clients == 0:
            return torch.zeros_like(flat_global_model)
        ordered_updates = [agent_updates_dict[cid] for cid in client_ids]
        if num_clients == 1:
            return ordered_updates[0]

        stacked = torch.stack(ordered_updates, dim=0)

        tmp = torch.median(stacked, dim=0).values

        sparsity = float(getattr(self.args, "sparsity", 0.3))
        lambda_s = float(getattr(self.args, "lambda_s", 1.0))
        lambda_c = float(getattr(self.args, "lambda_c", 1.0))
        eps = float(getattr(self.args, "eps", 1e-12))

        major_sign = torch.sign(tmp)
        tmp_norm = torch.norm(tmp).item()
        if tmp_norm <= eps:
            return tmp

        dim = tmp.numel()
        topk_dim = max(1, int(dim * sparsity))

        mpsa_scores = []
        tda_scores = []
        for i in range(num_clients):
            vec = stacked[i]
            _, topk_idx = torch.topk(torch.abs(vec), k=topk_dim)

            sign_vec = torch.sign(vec[topk_idx])
            agree = torch.sum(sign_vec == major_sign[topk_idx]).item()
            mpsa = agree / float(topk_dim)
            mpsa_scores.append(mpsa)

            vec_norm = torch.norm(vec).item()
            if vec_norm > eps:
                tda = float(torch.dot(vec, tmp).item() / (vec_norm * tmp_norm))
            else:
                tda = 0.0
            tda_scores.append(tda)

        def _mz_filter(scores, lam):
            arr = np.array(scores, dtype=np.float64)
            if arr.size <= 1:
                return set(range(len(scores)))
            std = np.std(arr)
            med = np.median(arr)
            std = std if std > 1e-12 else 1.0

            mz_low = (med - arr) / std
            keep_mask = mz_low < lam

            keep = set(np.argwhere(keep_mask).flatten().astype(int).tolist())
            return keep

        keep_mpsa = _mz_filter(mpsa_scores, lambda_s)
        keep_tda = _mz_filter(tda_scores, lambda_c)

        keep_idx_set = keep_mpsa.intersection(keep_tda)
        keep_idx = sorted(list(keep_idx_set))

        if len(keep_idx) == 0:
            return torch.zeros_like(flat_global_model)

        selected_updates_dict = {client_ids[i]: ordered_updates[i] for i in keep_idx}

        # ========== 第二部分 FedCODA Spectral 聚类筛选 ==========
        align_client_ids = list(selected_updates_dict.keys())
        align_local_updates = [selected_updates_dict[cid] for cid in align_client_ids]
        align_stacked = torch.stack(align_local_updates, dim=0)
        n_clients, dim = align_stacked.shape

        topk_ratio = float(getattr(self.args, "align_topk_ratio", 0.3))
        topk_dim = max(1, int(dim * topk_ratio))
        abs_updates = torch.abs(align_stacked)
        topk_indices = torch.topk(abs_updates, k=topk_dim, dim=1).indices
        topk_mask = torch.zeros_like(align_stacked, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)

        sign_updates = torch.sign(align_stacked)
        num_corrupt = int(getattr(self.args, "num_corrupt", 0))
        eps_align = 1e-8

        align_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        cosine_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        feature_matrix = np.zeros((n_clients, n_clients, 2), dtype=np.float64)

        for i in range(n_clients):
            for j in range(i, n_clients):
                if i == j:
                    align_matrix[i, j] = 1.0
                    cosine_matrix[i, j] = 0.0
                    feature_matrix[i, j] = [0.0, 1.0]
                    continue

                common_mask = topk_mask[i] & topk_mask[j]
                intersect_count = int(common_mask.sum().item())

                if intersect_count == 0:
                    align_score = 0.0
                    cosine_sim = 0.0
                else:
                    same_sign = torch.sum(sign_updates[i][common_mask] == sign_updates[j][common_mask]).item()
                    align_score = float(same_sign / intersect_count)

                    vec_i = align_stacked[i][common_mask]
                    vec_j = align_stacked[j][common_mask]
                    norm_i = torch.norm(vec_i).item()
                    norm_j = torch.norm(vec_j).item()
                    if norm_i > eps_align and norm_j > eps_align:
                        cosine_sim = float(torch.dot(vec_i, vec_j).item() / (norm_i * norm_j))
                    else:
                        cosine_sim = 0.0

                align_matrix[i, j] = align_matrix[j, i] = align_score
                cosine_matrix[i, j] = cosine_matrix[j, i] = cosine_sim
                feature_matrix[i, j] = feature_matrix[j, i] = [cosine_sim, align_score]

        triu_indices = np.triu_indices_from(cosine_matrix, k=1)
        cosine_vals = cosine_matrix[triu_indices]
        align_vals = align_matrix[triu_indices]

        eps_norm = 1e-12
        if len(cosine_vals) > 0:
            cmin, cmax = np.min(cosine_vals), np.max(cosine_vals)
        else:
            cmin, cmax = 0.0, 1.0
        if abs(cmax - cmin) < eps_norm:
            cosine_matrix_normalized = cosine_matrix.copy()
        else:
            cosine_matrix_normalized = (cosine_matrix - cmin) / (cmax - cmin)
            np.fill_diagonal(cosine_matrix_normalized, 0.0)

        if len(align_vals) > 0:
            amin, amax = np.min(align_vals), np.max(align_vals)
        else:
            amin, amax = 0.0, 1.0
        if abs(amax - amin) < eps_norm:
            align_matrix_normalized = align_matrix.copy()
        else:
            align_matrix_normalized = (align_matrix - amin) / (amax - amin)
            np.fill_diagonal(align_matrix_normalized, 1.0)

        feature_matrix_normalized = np.zeros((n_clients, n_clients, 2), dtype=np.float64)
        for i in range(n_clients):
            for j in range(n_clients):
                feature_matrix_normalized[i, j] = [cosine_matrix_normalized[i, j], align_matrix_normalized[i, j]]

        feature_dist_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    feature_dist_matrix[i, j] = 0.0
                else:
                    feat_i = feature_matrix_normalized[i, j]
                    cosine_sim = feat_i[0]
                    align_score = feat_i[1]
                    dist = np.sqrt((1.0 - cosine_sim) ** 2 + (1.0 - align_score) ** 2)
                    feature_dist_matrix[i, j] = dist

        selected_indices = list(range(n_clients))
        cluster_labels = None

        if n_clients >= 2:
            if SpectralClustering is not None:
                cluster_k = max(2, min(2, n_clients))
                similarity_matrix = 1.0 / (1.0 + feature_dist_matrix)
                np.fill_diagonal(similarity_matrix, 1.0)
                spectral = SpectralClustering(
                    n_clusters=cluster_k, affinity="precomputed", assign_labels="kmeans", random_state=42
                )
                cluster_labels = spectral.fit_predict(similarity_matrix)

        if cluster_labels is not None:
            unique_labels = sorted(set(cluster_labels))
            cluster_stats = []
            for cluster_id in unique_labels:
                member_idx = np.where(cluster_labels == cluster_id)[0]
                if len(member_idx) == 0:
                    continue
                member_ids = [align_client_ids[idx] for idx in member_idx]
                benign_cnt = sum(1 for cid in member_ids if cid >= num_corrupt)
                malicious_cnt = len(member_ids) - benign_cnt
                majority_type = "Benign" if benign_cnt >= malicious_cnt else "Malicious"
                cluster_stats.append((cluster_id, len(member_idx), majority_type, member_idx))

            best_cluster = None
            best_score = None
            best_avg_cos = None

            for stat in cluster_stats:
                cluster_id, size, majority_type, member_idx = stat
                if size == 0:
                    continue

                member_tensor_idx = np.array(member_idx, dtype=int)
                sub_cos = cosine_matrix_normalized[np.ix_(member_tensor_idx, member_tensor_idx)]
                tril = np.tril_indices_from(sub_cos, k=-1)
                if len(tril[0]) > 0:
                    pair_vals = sub_cos[tril]
                    avg_pair_cos = float(pair_vals.mean())
                else:
                    avg_pair_cos = 1.0

                score = size * (1.0 + avg_pair_cos)

                if best_score is None or score > best_score:
                    best_score = score
                    best_avg_cos = avg_pair_cos
                    best_cluster = stat

            if best_cluster is not None:
                chosen_cluster_id, chosen_size, _, chosen_member_idx = best_cluster
                selected_indices = np.array(chosen_member_idx, dtype=int).tolist()

        final_selected_client_ids = [align_client_ids[idx] for idx in selected_indices]
        final_selected_updates = {cid: selected_updates_dict[cid] for cid in final_selected_client_ids}

        # ========== 计算最终选中的客户端的 FPR 和 TPR ==========
        correct = 0
        for cid in final_selected_client_ids:
            if cid in benign_id:
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0

        FPR = 0.0
        if len(malicious_id) > 0:
            wrong = 0
            for cid in final_selected_client_ids:
                if cid in malicious_id:
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('[FedCODA] benign update index:   %s' % str(benign_id))
        logging.info('[FedCODA] selected update index: %s' % str(final_selected_client_ids))
        logging.info('[FedCODA] FPR:       %.4f' % FPR)
        logging.info('[FedCODA] TPR:       %.4f' % TPR)

        # ========== 第三部分：聚合最终选中的客户端更新 ==========
        aggregated_update = self.agg_avg(final_selected_updates)
        return aggregated_update