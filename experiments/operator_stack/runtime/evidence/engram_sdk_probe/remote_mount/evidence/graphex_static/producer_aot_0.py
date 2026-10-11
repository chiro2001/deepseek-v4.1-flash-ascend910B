


def forward(self, arg0_1, arg1_1, arg2_1):
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    matmul = torch.ops.aten.matmul.default(arg0_1, t);  arg0_1 = t = None
    auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.engram_sdk_probe.publish.default, value = matmul, event_id = 281471313784656, _cache_base_index = 0, _all_bases = [arg2_1]);  matmul = arg2_1 = None
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    return (getitem_1,)
    