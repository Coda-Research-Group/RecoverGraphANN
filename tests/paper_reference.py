"""Every number the camera-ready prints, transcribed once.

The tests that check these live in `test_paper_numbers_trace.py` (Table 1) and
`test_figure3_numbers_trace.py` (Figure 3 and §Results). Keeping the transcription in one
place means there is exactly one thing to update when the paper changes, and it is obvious
what that thing is.

Nothing here is computed. If a value below disagrees with the artifact, either the paper or
the artifact is wrong, and the failing test says which value.
"""

from __future__ import annotations

__all__ = [
    'DOUBLE_ROUNDED',
    'GRAPH_INDEXES',
    'RAW_BAND',
    'RAW_HNSW_TOP',
    'ROARGRAPH_QUERY_AGNOSTIC',
    'ROARGRAPH_TRANSFORMED',
    'TABLE_1',
    'TOLERANCE',
]

#: Half a unit in the last printed digit.
TOLERANCE = 0.05

# --- Table 1 (`tab:degree`) ------------------------------------------------------------------
# hnswlib, efConstruction=500, efSearch=1000, k=10, single-threaded.
# (dataset, M, transformed) -> (acc, deg, recall %)

TABLE_1 = {
    ('yi-128-ip', 4, False): (1.2, 1.7, 23.8),
    ('yi-128-ip', 16, False): (1.4, 1.9, 35.1),
    ('yi-128-ip', 48, False): (1.5, 2.2, 43.0),
    ('yi-128-ip', 96, False): (1.6, 2.5, 45.4),
    ('yi-128-ip', 4, True): (3.9, 7.3, 81.5),
    ('yi-128-ip', 16, True): (11.4, 19.6, 96.0),
    ('yi-128-ip', 48, True): (13.4, 27.1, 96.7),
    ('yi-128-ip', 96, True): (13.5, 27.2, 96.9),
    ('llama-128-ip', 4, False): (1.1, 1.5, 6.8),
    ('llama-128-ip', 16, False): (1.1, 1.3, 15.8),
    ('llama-128-ip', 48, False): (1.1, 1.4, 21.1),
    ('llama-128-ip', 96, False): (1.2, 1.5, 22.0),
    ('llama-128-ip', 4, True): (4.0, 7.5, 70.9),
    ('llama-128-ip', 16, True): (14.5, 23.6, 94.4),
    ('llama-128-ip', 48, True): (22.0, 42.6, 95.8),
    ('llama-128-ip', 96, True): (22.4, 45.2, 95.8),
}

#: Two cells were rounded twice — full value to two decimals, then to one — which carries them
#: one digit too high. Both are cosmetic and neither changes anything the paper argues, but
#: they are recorded rather than absorbed into a tolerance, so that correcting the table makes
#: the suite say so.
#:
#:   (dataset, M, transformed, column) -> (printed, correct one-decimal value)
DOUBLE_ROUNDED = {
    ('llama-128-ip', 16, True, 'acc'): (14.5, 14.4),  # logged 14.445700977564922
    ('llama-128-ip', 96, False, 'deg'): (1.5, 1.4),  # logged 1.4488850658373587
}

# --- Figure 3 (`fig:recall_qps`) and §Results ------------------------------------------------
# All at the highest search effort in the sweep (efSearch / search_L / L_pq = 1000).

#: The four indexes, all run query-agnostically. RoarGraph is query-aware by design and is
#: adapted to this setting; its native mode is out of scope — VIBE benchmarks that.
GRAPH_INDEXES = ('hnsw-hnswlib', 'flatnav', 'nsg-faiss')

#: ¶3: "At the highest recall raw HNSW reaches 43.0% on yi-128-ip and 21.1% on llama-128-ip".
RAW_HNSW_TOP = {'yi-128-ip': 43.0, 'llama-128-ip': 21.1}

#: ¶3: "In raw space, the query-agnostic graph indexes track each other closely: 38.0%-43.0%
#: on yi-128-ip and 18.5%-28.7% on llama-128-ip".
RAW_BAND = {'yi-128-ip': (38.0, 43.0), 'llama-128-ip': (18.5, 28.7)}

#: ¶5: "withholding that sample ... collapses it to 36.8% and 31.7%".
ROARGRAPH_QUERY_AGNOSTIC = {'yi-128-ip': 36.8, 'llama-128-ip': 31.7}

#: ¶5: "Applying the transformation to RoarGraph without the sample lifts it only to 64.5% and
#: 74.4%".
ROARGRAPH_TRANSFORMED = {'yi-128-ip': 64.5, 'llama-128-ip': 74.4}
