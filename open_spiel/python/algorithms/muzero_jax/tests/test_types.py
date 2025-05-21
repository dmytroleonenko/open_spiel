import jax
import pytest
from open_spiel.python.algorithms.muzero_jax.types import ModelOutput, MuZeroModel


def test_modeloutput_dataclass_fields():
    mo = ModelOutput(hidden_state='hs', reward='rw', policy_logits='pl', value='v')
    assert mo.hidden_state == 'hs'
    assert mo.reward == 'rw'
    assert mo.policy_logits == 'pl'
    assert mo.value == 'v'


def test_muzero_model_protocol_methods_execute():
    model = MuZeroModel()
    # initial_inference returns Ellipsis by design
    res_initial = model.initial_inference(observation='obs', rng_key=None, training=True)
    assert res_initial is Ellipsis
    # recurrent_inference returns Ellipsis by design
    res_recurrent = model.recurrent_inference(hidden_state='hs', action='a', rng_key=None, training=False)
    assert res_recurrent is Ellipsis 