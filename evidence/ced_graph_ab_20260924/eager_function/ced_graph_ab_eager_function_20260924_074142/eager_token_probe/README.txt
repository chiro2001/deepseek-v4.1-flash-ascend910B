Eager 22-token token-sequence baseline. Exactly two generation requests were sent; both used the archived same prompt and temperature=0, stream=false. API accepted logprobs=true, top_logprobs=5, so no no-logprobs fallback was sent.
max_tokens=1: request SHA bd6276adca2e71471858eb7402972561c1d543f39db67f76c3a706a38e6e2ff5; output Z; prompt/completion/total=22/1/23; finish=length; wall=1.227791s; the emitted token is Z with logprob -1.1920928244535389e-7.
max_tokens=2: request SHA 403e6cb2b9f26f1cd6adc8fc7ac21af1bd265769fa42fab2f6f2ccfce021378a; output ZQ; prompt/completion/total=22/2/24; finish=length; wall=0.583644s; token sequence Z then Q, Q logprob 0.0.
Both D replay logs cover positions 0..20 before the uncached last token, then replay chunk 21 on all eight D workers.
No additional generation requests were sent for this eager token probe.
