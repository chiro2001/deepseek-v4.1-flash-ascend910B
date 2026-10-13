"""Preserve actual proposal phases so prefill eager is distinct from decode fallback."""
from .topology import integer


def proposal_counts(proposals):
    if not isinstance(proposals, list):
        raise ValueError("Actual proposal list is required")
    graph = eager = 0
    for proposal in proposals:
        if not isinstance(proposal, dict) or type(proposal.get("runtime_graph")) is not bool:
            raise ValueError("Every proposal must carry actual boolean graph submission evidence")
        integer(proposal.get("k"), "proposal k", minimum=1)
        if proposal["runtime_graph"]:
            graph += 1
        else:
            eager += 1
    return {"graph_calls": graph, "eager_calls": eager, "proposal_calls": graph + eager}


def from_counter_request(request):
    initial = proposal_counts(request.get("initial_draft"))
    rows = request.get("rounds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Actual decode rounds are required for draft phase proof")
    rounds = []
    for index, row in enumerate(rows, 1):
        if row.get("phase") != "decode" or row.get("round") != index:
            raise ValueError("Actual decode proposal rounds are out of sequence")
        counts = proposal_counts(row.get("draft_calls"))
        if counts["proposal_calls"] == 0:
            raise ValueError("A real decode round has no actual proposal")
        rounds.append({"round": index, **counts})
    return {"schema": "actual_draft_phase_counts_v1", "initial_prefill": initial,
            "decode_rounds": rounds, "scope": "actual counter initial_draft and each decode round draft_calls; no estimated phases"}


def validate_phase_counts(proof, *, rounds, graph_calls, eager_calls):
    if not isinstance(proof, dict) or proof.get("schema") != "actual_draft_phase_counts_v1":
        raise ValueError("Actual per-phase draft graph proof is required")
    initial = proof.get("initial_prefill")
    rows = proof.get("decode_rounds")
    if not isinstance(initial, dict) or not isinstance(rows, list) or len(rows) != rounds:
        raise ValueError("Draft phase proof differs from actual verifier rounds")
    def counts(row):
        graph = integer(row.get("graph_calls"), "graph calls")
        eager = integer(row.get("eager_calls"), "eager calls")
        calls = integer(row.get("proposal_calls"), "proposal calls")
        if calls != graph + eager:
            raise ValueError("Proposal counts disagree within their actual phase")
        return graph, eager
    initial_graph, initial_eager = counts(initial)
    decode_graph = decode_eager = 0
    for index, row in enumerate(rows, 1):
        if type(row.get("round")) is not int or row["round"] != index:
            raise ValueError("Draft phase rounds are not the exact verifier sequence")
        graph, eager = counts(row)
        if graph <= 0 or eager != 0:
            raise ValueError("Every actual decode round must dispatch real draft graphs without eager fallback")
        decode_graph += graph
        decode_eager += eager
    if graph_calls != initial_graph + decode_graph or eager_calls != initial_eager + decode_eager:
        raise ValueError("Actual draft phase counts differ from cumulative graph/eager deltas")
    return {"initial_graph_calls": initial_graph, "initial_eager_calls": initial_eager,
            "decode_graph_calls": decode_graph, "decode_eager_calls": decode_eager}
