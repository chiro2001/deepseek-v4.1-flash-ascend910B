"""Strict formal-model comparison and compact failure diagnostics."""
import math
import numpy as np


def compare_requests(reference, actual, label, prompt_len=2048):
    # Validate the original values; narrowing first could turn 65536 into a
    # seemingly valid expert ID and hide corrupted route exports.
    a = np.asarray(reference['routes'])
    b = np.asarray(actual['routes'])
    tokens_equal = reference['token_ids'] == actual['token_ids']
    def valid(routes, row):
        return (np.issubdtype(routes.dtype, np.integer)
                and routes.shape == (prompt_len + len(row['token_ids']) - 1, 40, 6)
                and bool(np.all((routes >= 0) & (routes < 384)))
                and bool(np.all(np.diff(np.sort(routes, axis=-1), axis=-1) > 0)))
    result = {'label':label, 'tokens_equal':tokens_equal,
              'reference_route_shape':list(a.shape), 'actual_route_shape':list(b.shape),
              'reference_routes_valid':valid(a,reference), 'actual_routes_valid':valid(b,actual),
              'routes_equal':bool(np.array_equal(a,b))}
    if a.shape == b.shape and a.ndim == 3:
        diff = np.any(a != b, axis=-1)
        sets = np.any(np.sort(a,axis=-1) != np.sort(b,axis=-1),axis=-1)
        result.update(prefill_differing=int(diff[:prompt_len].sum()),
                      decode_differing=int(diff[prompt_len:].sum()),
                      route_set_differing=int(sets.sum()))
        positions = np.argwhere(diff)
        result['first_route_difference'] = None
        if len(positions):
            token,layer = positions[0]
            result['first_route_difference'] = {'token':int(token),'layer':int(layer),
                'reference':a[token,layer].tolist(),'actual':b[token,layer].tolist()}
    ref_probs = reference['logprobs']; actual_probs = actual['logprobs']
    lengths_valid = (len(ref_probs) == len(reference['token_ids'])
                     and len(actual_probs) == len(actual['token_ids'])
                     and len(ref_probs) == len(actual_probs) and bool(ref_probs))
    keys_equal = lengths_valid and all(set(r)==set(s) for r,s in zip(ref_probs,actual_probs))
    common_deltas = [abs(v-s[k]) for r,s in zip(ref_probs,actual_probs) for k,v in r.items() if k in s]
    common_max = max(common_deltas) if common_deltas else None
    deltas_finite = bool(common_deltas) and all(math.isfinite(v) for v in common_deltas)
    result.update(logprob_lengths_valid=lengths_valid, logprob_keys_equal=keys_equal,
                  max_common_logprob_delta=common_max,
                  max_logprob_delta=common_max if keys_equal else None,
                  logprob_deltas_finite=deltas_finite)
    result['passed'] = (tokens_equal and result['reference_routes_valid']
                        and result['actual_routes_valid'] and result['routes_equal']
                        and keys_equal and deltas_finite and common_max is not None and common_max < 1e-3)
    return result
