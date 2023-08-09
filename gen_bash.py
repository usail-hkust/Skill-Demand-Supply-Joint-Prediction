def gen_script():
    dataset = ["it"]
    # dataset = ["it"]
    # hyp_comb = ["add", "res", "pool","lambda"]
    model = ["static", "adaptive", "cross-view", "hier"]
    sk_emb_dim = [32]
    lr = [0.001]

    for d in dataset:
        for m in model:
            for emb in sk_emb_dim:
                for l in lr:
                    if m in ["static"]:
                        emb = 16
                    print(f'python train.py --config config.json --name {d}_{m}_{emb} --sk_emb_dim {emb} -um Skill_Evolve_Hetero --emb_dim {emb}  -m {m} --dataset {d} -lr {l} -dev 3 -gcn 2 --wandb true --dropout 0.3')

# gen_script()

def gen_script_hyp():
    dataset = ["it"]
    # dataset = ["it"]
    hyp_comb = ["add"]
    model = ["hier"]
    sk_emb_dim = [4,8,16,32,64]
    # 4,8,16,32,64
    lr = [0.001]
    gcn = [2]
    delta = [0.1]
    # 0.5, 0.1,0.01,0.001,0.0001
    for d in dataset:
        for m in model:
            for emb in sk_emb_dim:
                for l in lr:
                    for hc in hyp_comb:
                        for g in gcn:
                            for delt in delta:
                                print(f'python train.py --config config.json --name "{d}_{m}_{hc}_{emb}_{delt}" --sk_emb_dim {emb} -um Skill_Evolve_Hetero --emb_dim {emb}  -m {m} --dataset {d} -lr {l} --delta {delt} -gcn {g} -dev 1 --wandb true --dropout 0.3 -hyp True')

# gen_script_hyp()
# -hyp true -hypcomb {hc}
def gen_single_variate_script():
    dataset = ["it","fin","cons"]
    
    model = ["Transformer"]
    sk_emb_dim = [32]

    for d in dataset:
        for m in model:
            for emb in sk_emb_dim:
                print(f'python train.py --config config.json --name {d}_{m}_{emb} --univarmodel {m} --sk_emb_dim {emb} --emb_dim {emb} --dataset {d} -lr 0.001 --layer_num 3 -dev 0 --wandb true --dropout 0.3')

# gen_single_variate_script()

def gen_script_wave():
    dataset = ["it", "fin", "cons"]
    # dataset = ["it"]
    model = ["Fedformer"]
    sk_emb_dim = [32]
    lr = [0.001]

    for d in dataset:
        for m in model:
            for emb in sk_emb_dim:
                for l in lr:
                    print(f'python train.py --config config.json --name {d}_{m}_{emb} --sk_emb_dim {emb} -um Graph_Baseline --emb_dim {emb} -m {m} --dataset {d} -lr {l} -dev 1 --wandb true --dropout 0.3')

gen_script_wave()