"""Guard the post-hoc integration's sample and accounting boundaries."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from integrate_results import comparison_pairs, cost_for_predictions, failure_kind, network_subset_ids, ordered_predictions, specifications


def test_pair_plan_has_all_families_and_no_duplicate_pairs():
    specs=specifications()
    assert len(specs)==15 and len({s['method'] for s in specs})==15
    pairs=comparison_pairs()
    assert len(pairs)==42 and len(set(pairs))==42
    assert sum(b=='crf' for a,b in pairs)==12


def test_prediction_ids_must_match_and_reorder_only():
    gold=[{'id':'a'},{'id':'b'}]
    assert [p['id'] for p in ordered_predictions(gold,[{'id':'b'},{'id':'a'}])]==['a','b']
    for rows in [[{'id':'a'}],[{'id':'a'},{'id':'a'}],[{'id':'a'},{'id':'c'}]]:
        with pytest.raises(ValueError):ordered_predictions(gold,rows)


def test_shared_network_subset_keeps_refusals_and_format_errors():
    gold=[{'id':i} for i in 'abcdef']
    first=[{'id':'a','status':'api_error','error':{'exception':'URLError'}},
           {'id':'b','status':'format_error'},
           {'id':'c','status':'api_error','error':{'http_status':400,'provider_error_code':'1301'}},
           {'id':'d','status':'api_error','error':{'http_status':400,'exception':'URLError'}},
           {'id':'e','status':'ok','error':{'exception':'URLError'}},
           {'id':'f','status':'ok'}]
    second=[dict(p,status='ok') for p in first]
    second[-1]={'id':'f','status':'api_error','error':{'exception':'ConnectionRefusedError'}}
    kept,removed=network_subset_ids(gold,{'one':first,'two':second})
    assert kept==list('bcde') and removed==list('af')
    assert failure_kind(first[2])=='provider_content_refusal'
    assert failure_kind(first[4])==''  # Historical retry error on a successful result is not a failure.


def test_budget_includes_previous_unknown_attempts_once_per_cache_key():
    predictions=[{'cache_key':'x'},{'cache_key':'x'},{'cache_key':'y'}]
    attempts=[{'key':'x','status':'unknown','charged_upper_cny':2},
              {'key':'x','status':'response','charged_upper_cny':.2,
               'usage':{'prompt_tokens':10,'completion_tokens':2,'prompt_cache_hit_tokens':5}},
              {'key':'y','status':'response','charged_upper_cny':.3,
               'usage':{'prompt_tokens':20,'completion_tokens':3,'prompt_tokens_details':{'cached_tokens':10}}},
              {'key':'unrelated','status':'response','charged_upper_cny':99}]
    result=cost_for_predictions(predictions,attempts)
    assert result['attempts_including_retries']==3
    assert result['known_usage_estimate_cny']==pytest.approx(.5)
    assert result['unknown_reservations_cny']==2
    assert result['gross_ledger_upper_cny']==pytest.approx(2.5)
    assert result['cached_input_tokens']==15
    assert result['cache_hit_fraction']==.5
