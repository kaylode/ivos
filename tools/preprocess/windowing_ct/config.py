from typing import List 
cfg = {
    "head and neck": {
        "brain": [{"W": 80, "L": 40}],
        "stroke": [{"W": 8, "L": 32}, {"W": 40, "L": 40}],
        "temporal bones": [{"W": 2800, "L": 600}, {"W": 4000, "L": 700}],
        # "subdural": [{"W": w, "L": l} for w in range(130, 300) for l in range(50, 100)],
        # "soft tissues": [
        #     {"W": w, "L": l} for w in range(350, 400) for l in range(20, 60)
        # ],
    },
    "chest": {"lungs": [{"W": 1500, "L": -600}], "mediastinum": [{"W": 350, "L": 50}]},
    "abdomen": {"soft tissues": [{"W": 400, "L": 50}], "liver": [{"W": 150, "L": 30}]},
    "spine": {"soft tissues": [{"W": 250, "L": 50}], "bone": [{"W": 1800, "L": 400}]},
}


def print_cfg(cfg):
    for k, v in cfg.items():
        print(k)
        for k1, v1 in v.items():
            print(f"\t{k1}")
            for v2 in v1:
                print(f'\t\t{v2["W"]} {v2["L"]}')


def get_cfg(key, cfg=cfg):
    if isinstance(key, str):
        if "-" in key:
            key = key.split("-")
            key = [k.strip() for k in key]

        for k in key:
            if k in cfg:
                cfg = cfg[k]
            else:
                raise ValueError(f"{k} not in cfg")
        return cfg
    
    if isinstance(key, List):
        res = []
        for c in key:
            res += get_cfg(c, cfg)
        return res


if __name__ == "__main__":
    # print_cfg(cfg)
    print(get_cfg("spine-bone", cfg))
    print(get_cfg("abdomen-liver", cfg))
