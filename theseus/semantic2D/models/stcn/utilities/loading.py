from theseus.utilities.download import download_from_drive


PRETRAINED_MODELS = {
    'stcn': '1mRrE0uCI2ktdWlUgapJI_KmgeIiF2eOm'
}

def load_pretrained_model(name:str = 'stcn'):
    assert name in PRETRAINED_MODELS, f"Cannot find pretrained for {name}. Available models are: {PRETRAINED_MODELS.keys()}"
    return download_from_drive(PRETRAINED_MODELS['stcn'], output=None, cache=True) 