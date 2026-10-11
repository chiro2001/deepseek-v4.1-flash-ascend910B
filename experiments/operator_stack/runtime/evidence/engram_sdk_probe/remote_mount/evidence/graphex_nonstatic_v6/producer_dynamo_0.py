


def forward(self, L_a_ : torch.Tensor, L_w_ : torch.Tensor, L_c_ : torch.Tensor):
    l_a_ = L_a_
    l_w_ = L_w_
    l_c_ = L_c_
    linear = torch._C._nn.linear(l_a_, l_w_);  l_a_ = l_w_ = None
    get_external_object_by_index = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(0);  get_external_object_by_index = None
    publish_default = torch.ops.engram_sdk_probe.publish.default(linear, l_c_, 281471497624656);  linear = l_c_ = publish_default = None
    return ()
    