
.macro ld1h_data_2, reg1, reg2, pred, addr
ld1h \reg1, \pred/Z, [\addr]
ld1h \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro clamp_float reg, min_val, max_val, pred
FMIN \reg, \pred/M, \reg, \min_val
FMAX \reg, \pred/M, \reg, \max_val
.endm
