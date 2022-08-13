from theseus.utilities.download import download_from_drive

MODEL_CONFIGS = {
    'swin_tiny_patch4_window7_224': {
        'patch_size': 4,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        "window_size": 8,
        "mlp_ratio": 4.,
        "qkv_bias": True,
        "qk_scale": None,
        'drop_rate': 0,
        'drop_path_rate': 0.1,
        "ape": False,
        "patch_norm": True,
        "final_upsample": "expand_first"
    }
}

PRETRAINED_MODELS = {
    'swin_tiny_patch4_window7_224': '1TyMf0_uvaxyacMmVzRfqvLLAWSOE2bJR'
}


def get_config(model_name):
    assert model_name in MODEL_CONFIGS.keys(), f"{model_name} not available. Availabe models are {MODEL_CONFIGS}"
    return MODEL_CONFIGS[model_name]

def load_pretrained_model(name:str = 'swin_tiny_patch4_window7_224'):
    assert name in PRETRAINED_MODELS, f"Cannot find pretrained for {name}. Available models are: {PRETRAINED_MODELS.keys()}"
    return download_from_drive(PRETRAINED_MODELS[name], output=None, cache=True) 