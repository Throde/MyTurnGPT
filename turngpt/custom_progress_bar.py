# DH: for windows display (partial support for unicode, smooth block will not be shown)
# customize the progress bar to fall back to ascii-only progress bar.
# This 'tqdm' and super class 'ProgressBar' is originally defined in pytorch_lighning/core/lightning.py
# Source code: https://github.com/PyTorchLightning/pytorch-lightning

import importlib
import sys

from pytorch_lightning.callbacks import ProgressBar

if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm


_PAD_SIZE = 5


class tqdm(_tqdm):
    """
    Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from flickering
    """

    @staticmethod
    def format_num(n) -> str:
        """ Add additional padding to the formatted numbers """
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
        if should_be_padded and 'e' not in n:
            if '.' not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += '.'
            n += "0" * (_PAD_SIZE - len(n))
        return n


class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc='Validation sanity check',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True, # DH mod
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            ascii=True, # DH mod
        )
        return bar

    def init_predict_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for predicting. """
        bar = tqdm(
            desc='Predicting',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            ascii=True, # DH mod
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True, # DH mod
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True, # DH mod
        )
        return bar

# usage:
# bar = LitProgressBar()
# trainer = Trainer(callbacks=[bar])