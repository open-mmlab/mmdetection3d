from mmdet.apis.train import parse_losses


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    if 'img_meta' in data:
        num_samples = len(data['img_meta'].data)
    else:
        num_samples = len(data['img'].data)
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

    return outputs
