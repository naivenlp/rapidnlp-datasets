import torch


def padding_features(features, pad_id=0, max_length=512):
    padded_features = {}
    for feature in features:
        for k in feature.keys():
            if k not in padded_features:
                padded_features[k] = []
            value = feature[k]
            padded_value = value + [pad_id] * (max_length - len(value))
            padded_features[k].append(padded_value)
    # to tensor
    return {k: torch.LongTensor(v) for k, v in padded_features.items()}
