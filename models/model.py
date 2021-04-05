import tensorflow as tf


__all__ = ['get_model']


def get_model(args, config=None):
    model = None
    name = args.model.name

    if name == 'lstm_enc_dec':
        if args.settings.dataloader.env_only:
            from .lstm_enc_dec import Encoder_Decoder_Env
            model = Encoder_Decoder_Env(input_shapes=args.settings.dataloader.input_shapes, 
                                        output_shape=args.settings.dataloader.output_shape)
        else:
            from .lstm_enc_dec import Encoder_Decoder
            model = Encoder_Decoder(input_shapes=args.settings.dataloader.input_shapes, 
                                    output_shape=args.settings.dataloader.output_shape)
    elif name == 'rf':
        from .rf import get_rf_model
        model = get_rf_model(args)
    else:
        raise ValueError(args.model.name)
    
    return model