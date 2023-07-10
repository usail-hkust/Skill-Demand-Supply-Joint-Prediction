def gen_script():
    dataset = ["it"]
    # dataset = ["it"]
    # hyp_comb = ["add", "res", "pool","lambda"]
    model = ["static", "adaptive", "multi-view", "hier"]
    sk_emb_dim = [64]
    lr = [0.005]

    for d in dataset:
        for m in model:
            for emb in sk_emb_dim:
                for l in lr:
                        print(f'python train.py --config config.json --name {d}_{m}_{emb} --sk_emb_dim {emb} -um Skill_Evolve_Hetero --emb_dim {emb}  -m {m} --dataset {d} -lr {l} --subgraph 7446 --skill_num 7446 -dev 2 --wandb true')
def gen_script_hyp():
    dataset = ["it"]
    # dataset = ["it"]
    hyp_comb = ["add", "res", "pool","lambda"]
    model = ["hier"]
    sk_emb_dim = [64]
    lr = [0.005]

    for d in dataset:
        for m in model:
            for emb in sk_emb_dim:
                for l in lr:
                    for hc in hyp_comb:
                        print(f'python train.py --config config.json --name {d}_{m}_{hc}_{emb} --sk_emb_dim {emb} -um Skill_Evolve_Hetero --emb_dim {emb}  -m {m} --dataset {d} -lr {l} --subgraph 7446 --skill_num 7446 -dev 2 --wandb true -hyp true -hypcomb {hc}')

gen_script()

def gen_single_variate_script():
    dataset = ["it","fin", "cons"]
    
    model = ["LSTM","GRU","RNN"]
    sk_emb_dim = [32]

    for d in dataset:
        for m in model:
            for emb in sk_emb_dim:
                print(f'python train.py --config config.json --name {d}_{m}_{emb} --univarmodel {m} --emb_dim {emb} --dataset {d} -lr 0.001 --subgraph 7446 --skill_num 7446 --layer_num 3 -dev 0 --wandb true')

# gen_single_variate_script()

def gen_script_wave():
    dataset = ["it", "fin", "cons"]
    # dataset = ["it"]
    model = ["wavenet"]
    sk_emb_dim = [32]
    lr = [0.005]

    for d in dataset:
        for m in model:
            for emb in sk_emb_dim:
                for l in lr:
                    print(f'python train.py --config config.json --name {d}_{m}_{emb} --sk_emb_dim {emb} -um Graph_Baseline --emb_dim {emb} -m {m} --dataset {d} -lr {l} --subgraph 7446 --skill_num 7446 -dev 1 --wandb true')

# gen_script_wave()