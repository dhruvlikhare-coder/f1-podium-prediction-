""" Tests the adaptive stack class .Run: pytest -q test_Adaptive _stack.py"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble import AdaptiveStack, make_synthethic_f1

FEATS = ["grid" , "quali_gap" , "driver_form" , "team_form"]
CTX = ["circuit_type"]

def small_models():
    return[
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    RandomForestClassifier(n_estimators= 30 , max_depth= 5 , random_state= 0)
    ]
    
@pytest.fixture(scope ="module")
def data():
    df = make_synthethic_f1()
    return df[df["season"]<= 2020].copy()

@pytest.fixture(scope ="module")
def fitted(data):
    train = data[data["season"]< 2020]
    return AdaptiveStack(base_model=small_models() ,replay_window= 10 ).fit(train , FEATS , context_cols= CTX )


def test_default_base_model_seprate_estimator():
    models= AdaptiveStack().make_base_models()
    assert len(models) == 3 
    kinds = [(m.steps[-1][-1] if hasattr(m ,"steps")else m ).__class__.__name__ for m in models ]
    assert kinds == ['LogisticRegression' , 'RandomForestClassifier' , 'XGBClassifier' ]
    assert all (hasattr(m , "predict_proba") for m in models)
    
def test_every_method_return_valid_probabilites(fitted , data): 
    race= data[(data["season"] == 2020) & (data["round"]== 1 )]
    for method in fitted.available_methods:
        p = fitted.predict_proba(race, method)
        assert len(p) == len(race) and p.between(0, 1).all() and p.notna().all(), method
        
def test_unkown_method_raises( fitted , data):
    with pytest.raises(ValueError):
        fitted.predict_proba(data.head(20) , " does_not_ exist")
        
def test_update_changes_only(data):
    train = data[data["season"] < 2020]
    model = AdaptiveStack(base_model= small_models() , learning_rate= 1e-2 ).fit(
        train , FEATS , context_cols= CTX 
    )
    r1 = data[(data["season"] == 2020) & (data["round"] == 1)]
    r2 = data[(data["season"] == 2020) & (data["round"] == 2)]
    
    before = {m :model.predict_proba(r2 ,m) for m in model.available_methods}
    model.update(r1)
    after = {m: model.predict_proba(r2, m) for m in model.available_methods}
    
    for m in model.available_methods:
        same = np.allclose(before[m] , after[m])
        assert same == (m != "mlp_adaptive") , m 
        
def test_replay_window_limits_the_buffer(fitted):
    assert len(np.unique(fitted._buf_race)) <= fitted.replay_window 
    
def test_mlp_architecture_is_configurable(data):
    train = data[data["season"] < 2020]
    model = AdaptiveStack(base_model=small_models(), hidden=(8, 4), mlp_alpha=0.05).fit(
        train, FEATS, context_cols=CTX
    )
    assert tuple(model.mlp.get_params()["hidden_layer_sizes"]) == (8, 4)
    assert model.mlp_alpha == 0.05
    # input = one probability per base model + the context features
    assert model.mlp.n_features_in_ == len(small_models()) + len(CTX)
    
