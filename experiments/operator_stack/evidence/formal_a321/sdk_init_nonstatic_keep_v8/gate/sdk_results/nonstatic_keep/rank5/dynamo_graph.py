


def forward(self, L_actual_0_ : torch.Tensor, L_actual_1_ : torch.Tensor, L_actual_7_ : torch.Tensor, L_actual_2_ : torch.Tensor, L_actual_8_ : torch.Tensor, L_actual_3_ : torch.Tensor, L_actual_4_ : torch.Tensor, L_actual_9_ : torch.Tensor, L_actual_10_ : torch.Tensor, L_actual_11_ : torch.Tensor, L_actual_12_ : torch.Tensor, L_actual_6_ : torch.Tensor, L_actual_5_ : torch.Tensor, L_actual_13_ : torch.Tensor, L_actual_14_ : torch.Tensor, L_actual_15_ : torch.Tensor, L_actual_16_ : torch.Tensor, L_actual_17_ : torch.Tensor, L_actual_18_ : torch.Tensor):
    l_actual_0_ = L_actual_0_
    l_actual_1_ = L_actual_1_
    l_actual_7_ = L_actual_7_
    l_actual_2_ = L_actual_2_
    l_actual_8_ = L_actual_8_
    l_actual_3_ = L_actual_3_
    l_actual_4_ = L_actual_4_
    l_actual_9_ = L_actual_9_
    l_actual_10_ = L_actual_10_
    l_actual_11_ = L_actual_11_
    l_actual_12_ = L_actual_12_
    l_actual_6_ = L_actual_6_
    l_actual_5_ = L_actual_5_
    l_actual_13_ = L_actual_13_
    l_actual_14_ = L_actual_14_
    l_actual_15_ = L_actual_15_
    l_actual_16_ = L_actual_16_
    l_actual_17_ = L_actual_17_
    l_actual_18_ = L_actual_18_
    value = torch._C._nn.linear(l_actual_0_, l_actual_1_);  l_actual_0_ = l_actual_1_ = None
    zero_ = l_actual_7_.zero_();  zero_ = None
    getitem = l_actual_7_[slice(None, 6, None)];  l_actual_7_ = None
    copy_ = getitem.copy_(value);  getitem = copy_ = None
    fill_ = l_actual_8_.fill_(-1);  fill_ = None
    getitem_1 = l_actual_8_[slice(None, 6, None)];  l_actual_8_ = None
    copy__1 = getitem_1.copy_(l_actual_2_);  getitem_1 = copy__1 = None
    zero__1 = l_actual_9_.zero_();  zero__1 = None
    getitem_2 = l_actual_9_[slice(None, 6, None)];  l_actual_9_ = None
    copy__2 = getitem_2.copy_(l_actual_3_);  getitem_2 = copy__2 = None
    fill__1 = l_actual_12_.fill_(6);  l_actual_12_ = fill__1 = None
    add_ = l_actual_11_.add_(1);  l_actual_11_ = add_ = None
    copy__3 = l_actual_10_.copy_(l_actual_4_);  l_actual_10_ = copy__3 = None
    index_copy_ = l_actual_6_.index_copy_(0, l_actual_5_, value);  index_copy_ = None
    consumed = l_actual_6_.index_select(0, l_actual_5_);  l_actual_6_ = l_actual_5_ = None
    zero__2 = l_actual_13_.zero_();  zero__2 = None
    getitem_3 = l_actual_13_[slice(None, 6, None)];  l_actual_13_ = None
    copy__4 = getitem_3.copy_(consumed);  getitem_3 = consumed = copy__4 = None
    fill__2 = l_actual_14_.fill_(-1);  fill__2 = None
    getitem_4 = l_actual_14_[slice(None, 6, None)];  l_actual_14_ = None
    copy__5 = getitem_4.copy_(l_actual_2_);  getitem_4 = l_actual_2_ = copy__5 = None
    zero__3 = l_actual_15_.zero_();  zero__3 = None
    getitem_5 = l_actual_15_[slice(None, 6, None)];  l_actual_15_ = None
    copy__6 = getitem_5.copy_(l_actual_3_);  getitem_5 = l_actual_3_ = copy__6 = None
    fill__3 = l_actual_18_.fill_(6);  l_actual_18_ = fill__3 = None
    add__1 = l_actual_17_.add_(1);  l_actual_17_ = add__1 = None
    copy__7 = l_actual_16_.copy_(l_actual_4_);  l_actual_16_ = l_actual_4_ = copy__7 = None
    mul = value * 2;  value = None
    return (mul,)
    