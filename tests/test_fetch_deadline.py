import time
import pytest
from scripts import fetch_real_samples as fetch


def candidate(name="scene", cloud=0):
    return {"id":name,"bbox":[115,28,117,30],
            "properties":{"datetime":"2020-06-01T00:00:00Z","eo:cloud_cover":cloud}}


def test_expired_budget_makes_no_additional_requests(monkeypatch):
    monkeypatch.setattr(fetch,"_DEADLINE",time.time()-1)
    calls=[]
    with pytest.raises(TimeoutError):
        fetch._with_retry(lambda:calls.append(1))
    assert calls==[]


@pytest.mark.parametrize("fast",[False,True])
@pytest.mark.parametrize("where",["find_valid_center","scene_window_quality"])
def test_scene_selection_propagates_deadline(monkeypatch,fast,where):
    monkeypatch.setattr(fetch,"_DEADLINE",None)
    monkeypatch.setattr(fetch,"find_valid_center",lambda *a,**k:(500000,3200000,0))
    monkeypatch.setattr(fetch,"scene_window_quality",lambda *a,**k:{"valid_pct":100,"cloud_pct":0,"water_pct":0})
    calls=[]
    def expired(*a,**k):
        calls.append(1)
        raise TimeoutError("budget expired")
    monkeypatch.setattr(fetch,where,expired)
    picker=fetch._pick_scene_stac if fast else fetch._pick_scene
    with pytest.raises(TimeoutError):
        picker([candidate(),candidate("second")],116.3,29.15,"2020-06-01",35,half_km=.32)
    assert len(calls)==1


def test_zero_cloud_is_ranked_ahead_of_cloudy_candidate(monkeypatch):
    monkeypatch.setattr(fetch,"_DEADLINE",None)
    checked=[]
    def center(item,*args):
        checked.append(item["id"])
        return (500000,3200000,0)
    monkeypatch.setattr(fetch,"find_valid_center",center)
    monkeypatch.setattr(fetch,"scene_window_quality",lambda *a,**k:{"valid_pct":100,"cloud_pct":0,"water_pct":0})
    result=fetch._pick_scene_stac([candidate("cloudy",10),candidate("clear",0)],116.3,29.15,"2020-06-01",35,verify_k=1)
    assert checked==["clear"]
    assert result[0][0]["id"]=="clear"
