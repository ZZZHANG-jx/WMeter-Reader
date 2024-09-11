import torchvision.models as models
from models.cnn_exctractor_pretrain import Net_64,Net_64_distribution
from models.transformer_im_parallel_distributionsrc import TransformerModel_im_distributionsrc


def get_model(name,input_nc=3, output_nc=2, input_size=(512,512)):
    model = _get_model_instance(name)
    if name == 'cnn_extractor_pretrain_distribution':
        model = model()
    elif name == 'transformer_im_distribution':
        model = model()
    return model


def _get_model_instance(name):
    try:
        return {
            'cnn_extractor_pretrain_distribution':Net_64_distribution,
            'transformer_im_distribution':TransformerModel_im_distributionsrc
        }[name]
    except:
        print('Model {} not available'.format(name))
