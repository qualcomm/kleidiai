.macro load_data, reg1, reg2, reg3, reg4, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
ld1w \reg3, \pred/Z, [\addr, #2, MUL VL]
ld1w \reg4, \pred/Z, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro load_data_2, reg1, reg2, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro store_data, reg1, reg2, reg3, reg4, pred, addr
st1w \reg1, \pred, [\addr]
st1w \reg2, \pred, [\addr, #1, MUL VL]
st1w \reg3, \pred, [\addr, #2, MUL VL]
st1w \reg4, \pred, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro move_vector_tile, tile, reg1, reg2, reg3, reg4, pred, addr
mova \tile[\addr, 0], \pred/M, \reg1
mova \tile[\addr, 1], \pred/M, \reg2
mova \tile[\addr, 2], \pred/M, \reg3
mova \tile[\addr, 3], \pred/M, \reg4
.endm

.macro move_tile_vector, tile, reg1, reg2, reg3, reg4, pred, addr
mova \reg1, \pred/M, \tile[\addr, 0] 
mova \reg2, \pred/M, \tile[\addr, 1] 
mova \reg3, \pred/M, \tile[\addr, 2] 
mova \reg4, \pred/M, \tile[\addr, 3] 
.endm

.macro clamp_float reg, max_val, min_val, pred
FMIN \reg, \pred/M, \reg, \max_val
FMAX \reg, \pred/M, \reg, \min_val
.endm




.macro load_data_pred_index_counter_2, reg1, reg2, predr, predtype, start_idx, end_idx, addr, word_length
cntw \word_length
whilelt \predtype, \start_idx, \end_idx
ld1w \reg1, \predr/Z, [\addr]
add \start_idx, \start_idx, \word_length
whilelt \predtype, \start_idx, \end_idx
ld1w \reg2, \predr/Z, [\addr, #1, MUL VL]
neg \word_length, \word_length
add \start_idx, \start_idx, \word_length
neg \word_length, \word_length
.endm

.macro store_data_pred_index_counter_2, reg1, reg2, predr, predtype, start_idx, end_idx, addr, word_length
cntw \word_length
whilelt \predtype, \start_idx, \end_idx
st1w \reg1, \predr, [\addr]
add \start_idx, \start_idx, \word_length
whilelt \predtype, \start_idx, \end_idx
st1w \reg2, \predr, [\addr, #1, MUL VL]
neg \word_length, \word_length
add \start_idx, \start_idx, \word_length
neg \word_length, \word_length
.endm
