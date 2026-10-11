"""Source-only rewrite of the merged V4.1 projection's independent KV tail.

This changes submission order, not the original Q/KV numerical operations.
It does not import an accelerator package or claim model/performance validation.
"""
import ast
import hashlib
import textwrap


def early_kv_tail(source):
    source = textwrap.dedent(source)
    assert source.count('q_a, formal_kv = formal_qkv_project(') == 1
    assert source.count('kv = formal_kv') == 1
    begin = source.index('    # Part 2:')
    q_begin = source.index('    qr = attn.q_norm(q_a)', begin)
    q_end = source.index('    # Part 3:', q_begin)
    tail_begin = source.index('        kv = attn.kv_norm(kv)', q_end)
    final_q = source.index('    q = wq_b.matmul(', tail_begin)
    # Keep both original branch bodies byte-for-byte, including RoPE and SWA.
    q_body = source[q_begin:q_end].rstrip() + '\n'
    kv_body = source[tail_begin:final_q]
    assert 'scatter_cache_sk(' in kv_body
    replacement = (
        '    # Merged output is ready: submit the original KV tail before Q norm.\n'
        '    assert share_quant, "early merged KV tail requires shared native quantization"\n'
        '    kv_tail_start = main_stream.record_event()\n'
        '    with npu_stream_switch(aux_stream, enabled=True):\n'
        '        aux_stream.wait_event(kv_tail_start)\n'
        '        kv = formal_kv\n'
        + kv_body + q_body
    )
    updated = source[:begin] + replacement + source[final_q:]
    tree = ast.parse(updated)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert updated.count('formal_qkv_project(') == 1
    assert updated.count('attn.q_norm(q_a)') == updated.count('attn.kv_norm(kv)') == 1
    assert updated.count('scatter_cache_sk(') == 1
    assert updated.count('inplace_partial_rotary_mul(') == 2
    assert 'kv_matmul_done' not in updated and 'part3_start' not in updated
    # The final join and Q RoPE still follow both independent branch bodies.
    assert source[final_q:] == updated[updated.index('    q = wq_b.matmul('):]
    assert q_body in updated and kv_body in updated
    evidence = {
        'variant': 'early-kv-tail-v2',
        'preserved_q_body_sha256': hashlib.sha256(q_body.encode()).hexdigest(),
        'preserved_kv_tail_sha256': hashlib.sha256(kv_body.encode()).hexdigest(),
        'preserved_q_matmul_join_rope_sha256': hashlib.sha256(source[final_q:].encode()).hexdigest(),
        'rewrite_sha256': hashlib.sha256(updated.encode()).hexdigest(),
        'scope': 'Frozen numerical bodies and final join; device/model gates remain required',
    }
    return updated, evidence
