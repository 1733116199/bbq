import os
import numpy as np
import torch
import random
import cutlass

alpha = 1
beta = 0

plan = cutlass.Gemm(
    element_A=cutlass.DataType.s4, 
    layout_A=cutlass.LayoutType.RowMajor, 
    element_B=cutlass.DataType.s4, 
    layout_B=cutlass.LayoutType.ColumnMajor,
    element_C=cutlass.DataType.s32, 
    layout_C=cutlass.LayoutType.RowMajor,
    element_D=cutlass.DataType.s32,
    element_accumulator=cutlass.DataType.s32,
    alpha=alpha,
    beta=beta,
)


print(plan.op_class)
print(plan.cc)
print(plan.activation)

import cutlass.emit
op = plan.construct(alignment_A=32, alignment_B=32, alignment_C=8)
grouped_gemm = cutlass.emit.pytorch(op, name='i4mm', cc=plan.cc, sourcedir='.', jit=False)