import config
import random


def print_f(str, f=None):
    if f is not None:
        print(str, file=f)
        if random.randint(0, 20) < 3:
            f.flush()
    print(str)


# ---- Some functions of lr operation ---- #
def lr_poly(base_lr, iter, epoch, max_iter, base_iter, power, change):
    """
        update learning rate
    """
    if change:
        if epoch >= 30:
            new_lr = base_lr * ((1 - float(iter - base_iter) / (max_iter - base_iter)) ** (power))
        else:
            new_lr = base_lr
    else:
        new_lr = base_lr * ((1 - float(iter) / max_iter) ** (power))

    # ---- bound it to 1e-5 ---- #
    if new_lr <= 1e-5:
        new_lr = 1e-5
    return new_lr


def adjust_learning_rate(optimizer, i_iter, epoch, whole_steps, base_steps, change=True):
    """
        update learning rate by steps
    """
    lr = lr_poly(config.LEARNING_RATE, i_iter, epoch, whole_steps, base_steps, config.POWER, change)

    optimizer.param_groups[0]['lr'] = lr

    return lr
