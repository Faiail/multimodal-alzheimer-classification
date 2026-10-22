# src/experiment_config.py

# Supported models definition
DL_MODELS = [
    # 'GAT', 
    'GCN',
    # 'dualgraph'
]

# Mapping models to architectural flags
DL_ARCHITECTURES = {
    # 'FullKAN':     {'use_kan_backbone': True,  'use_kan_head': True},  # kan completo
    'backbonekan': {'use_kan_backbone': True, 'use_kan_head': False},
    # 'headkan':     {'use_kan_backbone': False, 'use_kan_head': True},
    # 'base':        {'use_kan_backbone': False, 'use_kan_head': False},
}

N_LAYERS_OPTIONS = [
    # 3,
    2
]  # Network depth test

MODALITIES = ['Imaging', 'Multimodal']

GRAPH_TYPES = [
    # 'multigraph', #1
    # 'zhang',      #2
    # 'knn',        #3
    'static',     #4
    # 'hybrid'
]

# Reference for Zhang statistics calculation
ZHANG_REFERENCE_OPTIONS = [
    # 'all_train',
    # 'cn_only',
    'mci_only'
]

# Fusion strategies for multigraph
MULTIGRAPH_FUSION_OPTIONS = [
    'hetero_sum',
]

LOSS_CONFIGS = {
    # 'cross_entropy': {'type': 'cross_entropy'},
    # 'focal_1.0': {'type': 'focal', 'gamma': 1.0, 'alpha': none},
    # 'focal_1.0_balanced': {'type': 'focal'
    # 'focal_1.0_sqrt':     {'type': 'focal'
    # 'focal_1.5': {'type': 'focal', 'gamma': 1.5, 'alpha': none},
    # 'focal_1.5_balanced': {'type': 'focal'
    # 'focal_1.5_sqrt':     {'type': 'focal'
    # 'focal_2.0': {'type': 'focal', 'gamma': 2.0, 'alpha': none},
    # 'focal_2.0_balanced': {'type': 'focal', 'gamma': 2.0, 'alpha': 'balanced'},
    'focal_2.0_sqrt':     {'type': 'focal', 'gamma': 2.0, 'alpha': 'sqrt_balanced'},
} 

def get_experiment_schedule(test_mode=False):
    experiments = []
    
    if test_mode:
        print("⚡ TEST MODE: one heterogeneous GCN configuration")
        arch_name, arch_flags = next(iter(DL_ARCHITECTURES.items()))
        loss_name = next(iter(LOSS_CONFIGS))
        return [{
            'Family': 'DL',
            'Model': 'GCN',
            'Arch': arch_name,
            'Flags': arch_flags,
            'Data': 'Std',
            'Modality': 'Multimodal',
            'Graph': 'multigraph',
            'NLayers': 2,
            'ZhangRef': 'all_train',
            'MultigraphFusion': 'hetero_sum',
            'Loss': loss_name,
        }]

    # graph topologies comparison for base gcn (without kan)
    for loss_name, loss_flags in LOSS_CONFIGS.items():
        for dl_model in DL_MODELS:
            for arch_name, arch_flags in DL_ARCHITECTURES.items():
                for n_layers in N_LAYERS_OPTIONS:
                    for graph_type in GRAPH_TYPES:
                        zhang_refs = ZHANG_REFERENCE_OPTIONS if graph_type in ('zhang', 'hybrid', 'multigraph') else ['all_train']
                        for zhang_ref in zhang_refs:
                            fusion_options = MULTIGRAPH_FUSION_OPTIONS if graph_type == 'multigraph' else ['hetero_sum']
                            for mg_fusion in fusion_options:
                                exp = {
                                    'Family': 'DL',
                                    'Model': dl_model,
                                    'Arch': arch_name,
                                    'Flags': arch_flags,
                                    'Data': 'Std',
                                    'Modality': 'Multimodal',
                                    'Graph': graph_type,
                                    'NLayers': n_layers,
                                    'Loss': loss_name,
                                }
                                if graph_type in ('zhang', 'hybrid', 'multigraph'):
                                    exp['ZhangRef'] = zhang_ref
                                if graph_type == 'multigraph':
                                    exp['MultigraphFusion'] = mg_fusion
                                experiments.append(exp)

    return experiments