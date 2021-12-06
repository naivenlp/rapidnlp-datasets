import torch


def padding_dict(datas, pad_id=0, max_length=512):
    padded_features = {}
    for data in datas:
        for k in data.keys():
            if k not in padded_features:
                padded_features[k] = []
            value = data[k]
            padded_value = value + [pad_id] * (max_length - len(value))
            padded_features[k].append(padded_value)
    # to tensor
    return {k: torch.LongTensor(v) for k, v in padded_features.items()}


def padding_features(features, pad_id=0, max_length=512):
    return padding_dict(features, pad_id=pad_id, max_length=max_length)


def padding_labels(labels, pad_id=0, max_length=512):
    return padding_dict(labels, pad_id=pad_id, max_length=max_length)
