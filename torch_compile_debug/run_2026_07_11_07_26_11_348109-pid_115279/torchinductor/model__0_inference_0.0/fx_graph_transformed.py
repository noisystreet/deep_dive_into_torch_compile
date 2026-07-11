class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10]"):
        # No stacktrace found for following nodes
        sin: "f32[10]" = torch.ops.aten.sin.default(arg0_1)
        cos: "f32[10]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
        add: "f32[10]" = torch.ops.aten.add.Tensor(sin, cos);  sin = cos = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None
        return (sum_1,)
