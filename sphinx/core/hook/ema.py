import copy
import torch


class EMAHook(object):

    def __init__(self, model, ema_states=None, enabled=False, alpha=0.5, start_epoch=0):
        self.enabled = enabled
        self.model = model
        self.alpha = alpha
        self.start_epoch = start_epoch
        if ema_states is None:
            self.ema_states = copy.deepcopy(model.state_dict(keep_vars=False))
        elif ema_states is not None:
            self.ema_states = ema_states
        else:
            raise ValueError("unexpected ema_states state")
        # TODO: move ema states to model states and stored to pth, for resume purpose if needed.
        self.restore_states = None

    def step(self, epoch):
        if not self.enabled:
            return
        model_state = self.model.state_dict()
        for name, param in model_state.items():
            if epoch < self.start_epoch:
                self.ema_states[name].data.copy_(param.data)
            elif epoch >= self.start_epoch:
                self.ema_states[name] = torch.lerp(param, self.ema_states[name], self.alpha).detach()
            else:
                raise ValueError(f"unexpected epoch state: {epoch}")

    def state_dict(self):
        if not self.enabled:
            return self.model.state_dict()
        elif self.enabled:
            return self.ema_states
        else:
            raise ValueError(f"unexpected ema enabled state: {self.enabled}")

    def apply_ema(self):
        if not self.enabled:
            return
        # print('applying ema states to model for evaluation')
        self.restore_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema_states)

    def restore(self):
        if not self.enabled:
            return
        # print('restoring model after applying ema states.')
        self.model.load_state_dict(self.restore_states)
        self.restore_states = None
