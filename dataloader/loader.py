import tensorflow as tf

__all__ = ['get_dataloader']

def get_dataloader(args):

    name = args.model.name

    if name == 'lstm_enc_dec':
        if args.settings.dataloader.env_only:
            from .lstm_enc_dec_dataloader import dataloader4lstm_enc_dec_env
            dataloader = dataloader4lstm_enc_dec_env(args)
        else:
            from .lstm_enc_dec_dataloader import dataloader4lstm_enc_dec
            dataloader = dataloader4lstm_enc_dec(args)
    
    elif name == 'rf':
        from .rf_dataloader import rf_dataloader
        dataloader = rf_dataloader(args)

    else:
        raise ValueError(args.model.name)

    return dataloader
    