


def forward(self, arg0_1, arg1_1, arg2_1):
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    matmul = torch.ops.aten.matmul.default(arg0_1, t);  arg0_1 = t = None
    auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.engram_sdk_probe.publish.default, value = matmul, cache = arg2_1, event_id = 281471497624656);  matmul = arg2_1 = None
    getitem_1 = auto_functionalized[1];  auto_functionalized = None
    return (getitem_1,)
    