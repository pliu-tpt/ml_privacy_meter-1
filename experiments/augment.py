import numpy as np
import torch


def get_signal_on_augmented_data(model_pm, data, targets, method=None):
    if method == "augmented":
        # implement your own way of using the augmented data for inferring membership
        reflect = True
        shift = 0
        stride = 1

        outs = []
        for aug in [data, data.flip(dims=[3])][: reflect + 1]:
            aug_pad = torch.nn.functional.pad(aug, [shift] * 4, mode="reflect")
            for dx in range(0, 2 * shift + 1, stride):
                for dy in range(0, 2 * shift + 1, stride):
                    this_x = aug_pad[:, :, dx : dx + 32, dy : dy + 32]
                    logits = model_pm.get_rescaled_logits(this_x, targets)
                    outs.append(logits)
        return np.transpose(np.array(outs), (1, 0))  # (batch size, number of aug)

    else:
        return model_pm.get_rescaled_logits(data, targets)



def get_rpop_on_augmented_data(model_pm, data, targets, method=None):
    if method == "augmented":
        # implement your own way of using the augmented data for inferring membership
        reflect = True
        shift = 0
        stride = 1

        outs = []
        for aug in [data, data.flip(dims=[3])][: reflect + 1]:
            aug_pad = torch.nn.functional.pad(aug, [shift] * 4, mode="reflect")
            for dx in range(0, 2 * shift + 1, stride):
                for dy in range(0, 2 * shift + 1, stride):
                    this_x = aug_pad[:, :, dx : dx + 32, dy : dy + 32]
                    logits = model_pm.get_softmax(this_x, targets) # model_pm.get_taylor_softmax_margin(this_x, targets)
                    outs.append(logits)
        if len(np.array(outs).shape) == 3:
            return np.transpose(np.array(outs)[:,:,0], (1, 0))  # (batch size, number of aug)
        else:
            return np.transpose(np.array(outs)[:,:], (1, 0))  # (batch size, number of aug)

    else:
        return model_pm.get_softmax(data, targets) # model_pm.get_taylor_softmax_margin(data, targets)


def get_augmented_data(data, targets, method=None):
    if method is None:
        return data, targets
    elif method == "augmented":
        reflect = True
        shift = 0
        stride = 1

        targets_list = []
        data_list = []
        for aug in [data, data.flip(dims=[3])][: reflect + 1]:
            aug_pad = torch.nn.functional.pad(aug, [shift] * 4, mode="reflect")
            for dx in range(0, 2 * shift + 1, stride):
                for dy in range(0, 2 * shift + 1, stride):
                    this_x = aug_pad[:, :, dx : dx + 32, dy : dy + 32]
                    data_list.append(this_x)
                    targets_list.append(targets)
        return data, targets

    else:
        raise NotImplementedError
