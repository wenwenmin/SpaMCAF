import numpy as np
from sklearn.metrics import adjusted_rand_score
from utils import *
from model import *
from dataset import *
from config import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
# section_id = "mouse_brain"
# save_model_path = f"../T2/{section_id}"
# section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
#                     "151675", "151676", "mouse_brain", "Breast Cancer", "FFPE"]
# ## 508 669 670 671 672
# for section_id in section_list:
section_id = "FFPE"
save_model_path = f"../T1/{section_id}"
adata, _ = get_sectionData(section_id)
adata_TriHRGE = sc.read_h5ad(save_model_path + '/recovered_data.h5ad')
print(adata_TriHRGE)
adata_sample = sc.read_h5ad(save_model_path + '/sampled_data.h5ad')
pr_stage = np.zeros(adata_TriHRGE.shape[1])
P_value = np.ones(adata_TriHRGE.shape[1])
mse_values = np.zeros(adata_TriHRGE.shape[1])
mae_values = np.zeros(adata_TriHRGE.shape[1])
used_gene = adata_TriHRGE.var.index

for it in tqdm(range(adata_TriHRGE.shape[1])):
    pr_stage[it], P_value[it] = \
        pearsonr(adata_TriHRGE[:, used_gene[it]].X.toarray().squeeze(), adata[:, used_gene[it]].X.toarray().squeeze())
    mse_values[it] = mean_squared_error(adata_TriHRGE[:, used_gene[it]].X.toarray().squeeze(),
                                            adata[:, used_gene[it]].X.toarray().squeeze())
    mae_values[it] = mean_absolute_error(adata_TriHRGE[:, used_gene[it]].X.toarray().squeeze(),
                                             adata[:, used_gene[it]].X.toarray().squeeze())
mask = ~np.isnan(pr_stage)
pr_stage_n = pr_stage[mask]
used_gene_n = used_gene[mask]
p_value = P_value[mask]
print("section_id:", section_id, "PCC:", np.mean(pr_stage_n))
print("section_id:", section_id, "AVG MSE:", np.mean(mse_values))
print("section_id:", section_id, "AVG MAE:", np.mean(mae_values))

    #show_gene = ["Ttr", "Mbp", "Nrgn"]

    # import os
    #
    # save_fig_path = rf"C:/Users/DELL/Desktop/1"
    # os.makedirs(save_fig_path, exist_ok=True)
    # section_id = "mouse_brain"
    # save_model_path = f"../T1/{section_id}"
    # adata_sample = sc.read_h5ad(f"../T1/{section_id}/sampled_data.h5ad")
    # adata, _ = get_sectionData(section_id)
    # sc.set_figure_params(dpi=80, figsize=(2.8, 3))
    # sc.pl.embedding(adata_sample, basis="coord", color=show_gene, s=30, show=True)
    #
    # adata_recover = sc.read_h5ad(f"../T1/{section_id}/recovered_data.h5ad")
    #
    # sc.set_figure_params(dpi=80, figsize=(2.8, 3))
    # sc.pl.embedding(adata_recover, basis="coord", color=show_gene, s=30, show=True)
    # break
# print(adata_TriHRGE, adata_sample)
# with open(save_model_path + '/h_features.pkl', 'rb') as f:
#     h_features = pickle.load(f)
# # #
# # # 打印特征形状以确认加载成功
# print(f'Loaded features shape: {h_features.shape}')
# adata_H = sc.AnnData(h_features)
# adata_H.obsm["spatial"] = adata.obsm["spatial"]
# adata_TriHRGE.obsm["spatial"] = adata.obsm["spatial"]
# #使用 KMeans 进行聚类
# ARI_TriHRGE = {}
# ARI_h = {}
#
# n_clusters = 5
#
# kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=123) #661
# kmeans_labels = kmeans.fit_predict(adata_TriHRGE.X) # h_features
#
# adata_TriHRGE.obs['kmeans'] = kmeans_labels.astype(str)
#
# ari = adjusted_rand_score(adata.obs['layer'].astype(str), adata_TriHRGE.obs['kmeans'].astype(str))
# # if ari < 0.18:
# sc.pl.spatial(adata_TriHRGE, color='kmeans', title='KMeans Clustering', spot_size=150, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\1/{section_id}_{ari}.pdf', format='pdf')
#
# ARI_TriHRGE[i] = ari
# print(i, ari)
# break
# for i in range(1000):
#
#     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=320)  # 661
#     kmeans_labels = kmeans.fit_predict(h_features)  # h_features
#
#     adata_H.obs['kmeans'] = kmeans_labels.astype(str)
#
#     ari = adjusted_rand_score(adata.obs['layer'].astype(str), adata_H.obs['kmeans'].astype(str))
#     sc.pl.spatial(adata_H, color='kmeans', title='KMeans Clustering', spot_size=150, show=False)
#     # plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\TriHRGE/{section_id}_{ari}.pdf', format='pdf')
#     ARI_h[i] = ari
#     print(i, ari)
#     break
#
#
# max_i = max(ARI_TriHRGE, key=ARI_TriHRGE.get)
# max_ari = ARI_TriHRGE[max_i]
# print(f"Max ARI_TriHRGE: {max_ari} at PCA components: {max_i}")
#
# max_i_h = max(ARI_h, key=ARI_h.get)
# max_ari_h = ARI_h[max_i_h]
# print(f"Max ARI_H: {max_ari_h} at PCA components: {max_i_h}")

# ARI = {}
# for i in range(100, 200):
#     sc.pp.pca(adata_TriHRGE, n_comps=100)
#     sc.tl.tsne(adata_TriHRGE)
#     # print(adata_TriHRGE)
#     adata_TriHRGE.obsm["spatial"] = adata.obsm["spatial"]
#     kmeans_adata_stage = KMeans(n_clusters=7, init='k-means++', random_state=0).fit(adata_TriHRGE.obsm["X_pca"])
#     adata_TriHRGE.obs['kmeans'] = kmeans_adata_stage.labels_.astype(str)
#     ari = adjusted_rand_score(adata.obs['layer'].astype(str), adata_TriHRGE.obs['kmeans'].astype(str))
#     sc.pl.spatial(adata_TriHRGE, basis="spatial", color='kmeans', title=f'KMeans+{ari}', spot_size=150)
#     ARI[i] = ari
#     break
# # 找出最大ARI值及其对应的i
# max_i = max(ARI, key=ARI.get)
# max_ari = ARI[max_i]
# print(f"Max ARI: {max_ari} at PCA components: {max_i}")

# pr_stage = np.zeros(adata_TriHRGE.shape[1])
# P_value = np.ones(adata_TriHRGE.shape[1])
# mse_values = np.zeros(adata_TriHRGE.shape[1])
# mae_values = np.zeros(adata_TriHRGE.shape[1])
# used_gene = adata_TriHRGE.var.index
#
# for it in tqdm(range(adata_TriHRGE.shape[1])):
#     pr_stage[it], P_value[it] = \
#         pearsonr(adata_TriHRGE[:, used_gene[it]].X.toarray().squeeze(), adata[:, used_gene[it]].X.toarray().squeeze())
#     mse_values[it] = mean_squared_error(adata_TriHRGE[:, used_gene[it]].X.toarray().squeeze(),
#                                         adata[:, used_gene[it]].X.toarray().squeeze())
#     mae_values[it] = mean_absolute_error(adata_TriHRGE[:, used_gene[it]].X.toarray().squeeze(),
#                                          adata[:, used_gene[it]].X.toarray().squeeze())
mask = ~np.isnan(pr_stage)
pr_stage_n = pr_stage[mask]
used_gene_n = used_gene[mask]
# p_value = P_value[mask]
# print("section_id:", section_id, "PCC:", np.mean(pr_stage_n))
# print("section_id:", section_id, "AVG MSE:", np.mean(mse_values))
# print("section_id:", section_id, "AVG MAE:", np.mean(mae_values))
#
# 指定要输出的基因名称
genes_to_output = ["AZGP1"]

# 查找这些基因对应的PCC值
for gene in genes_to_output:
    # 查找基因的索引
    gene_index = np.where(used_gene_n == gene)[0]

    # 如果找到了该基因
    if len(gene_index) > 0:
        pcc_value = pr_stage_n[gene_index[0]]
        print(f"PCC Value for {gene}: {pcc_value}")
    else:
        print(f"{gene} not found in the data.")

sorted_indices = np.argsort(pr_stage_n)[::-1][:5]
top_genes = [used_gene_n[idx] for idx in sorted_indices]
top_pcc_values = [pr_stage_n[idx] for idx in sorted_indices]
# #
# #
for i, gene in enumerate(top_genes):
    print(f"Top {i + 1}: Gene Name = {gene}, PCC Value = {top_pcc_values[i]}")

show_gene = ["AZGP1"]

import os

save_fig_path = rf"C:/Users/DELL/Desktop/1"
os.makedirs(save_fig_path, exist_ok=True)
save_model_path = f"../T1/{section_id}"
adata_sample = sc.read_h5ad(f"../T1/{section_id}/sampled_data.h5ad")
adata, _ = get_sectionData(section_id)
sc.set_figure_params(dpi=80, figsize=(2.8, 3))
sc.pl.embedding(adata_sample, basis="coord", color=show_gene, s=30, show=True)

adata_recover = sc.read_h5ad(f"../T1/{section_id}/generated_data_8x.h5ad")

sc.set_figure_params(dpi=80, figsize=(2.8, 3))
sc.pl.embedding(adata_recover, basis="coord", color=show_gene, s=8, show=True)

save_fig_path = rf"C:\Users\DELL\Desktop\1/"
os.makedirs(save_fig_path, exist_ok=True)

sc.set_figure_params(dpi=80, figsize=(2.8, 3))
sc.pl.embedding(adata, basis="coord", color=show_gene, s=30, show=True)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
plt.savefig(save_fig_path + "GeneTruth.pdf", format='pdf', bbox_inches="tight")
#
#
sc.set_figure_params(dpi=80, figsize=(2.8, 3))
sc.pl.embedding(adata_sample, basis="coord", color=show_gene, s=30, show=False)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
plt.savefig(save_fig_path + "GeneSample.pdf", format='pdf', bbox_inches="tight")
#
#
#
# pcc_show_gene = {gene: pearsonr(adata_TriHRGE[:, gene].X.toarray().squeeze(),
#                                 adata[:, gene].X.toarray().squeeze())[0]
#                  for gene in show_gene}
#
# # %%
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_TriHRGE, basis="coord", color=show_gene, s=30, show=True)
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1)
#
# positions = {
#     show_gene[0]: (10, 10),
#     show_gene[1]: (20, 20),
#     show_gene[2]: (30, 30),
# }
# #Add PCC annotations to the plot
# for gene in show_gene:
#     x, y = positions[gene]
#     ax.text(x, y, f"{gene}\nPCC: {pcc_show_gene[gene]:.2f}",
#             fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))
#
# plt.savefig(save_fig_path + "GeneRecovery.pdf", format='pdf', bbox_inches="tight")
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_TriHRGE, basis="coord", color=show_gene, s=30, show=False)
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1)
# plt.savefig(save_fig_path + "GeneRecovery.pdf", format='pdf', bbox_inches="tight")