TSLIB_MODELS = (
    "Autoformer",
    "iTransformer",
)


def forward_batch(model, batch_x, device, model_name):
    if model_name in TSLIB_MODELS:
        outputs = model(
            batch_x["x_enc"].to(device),
            batch_x["x_mark_enc"].to(device),
            batch_x["x_dec"].to(device),
            batch_x["x_mark_dec"].to(device),
        )
    else:
        x_input = batch_x["x_enc"] if isinstance(batch_x, dict) else batch_x
        outputs = model(x_input.to(device))

    return outputs[0] if isinstance(outputs, tuple) else outputs


def match_target_shape(outputs, target):
    if outputs.shape[1] > target.shape[1]:
        outputs = outputs[:, -target.shape[1]:]
    if outputs.ndim == 2 and target.ndim == 3 and target.shape[-1] == 1:
        outputs = outputs.unsqueeze(-1)
    if outputs.shape != target.shape:
        raise ValueError(
            f"Model output shape {tuple(outputs.shape)} does not match target shape {tuple(target.shape)}"
        )
    return outputs
