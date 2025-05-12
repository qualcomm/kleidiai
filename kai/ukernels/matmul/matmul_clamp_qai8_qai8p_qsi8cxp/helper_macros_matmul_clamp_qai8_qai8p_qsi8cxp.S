
.macro frintn_convert, reg1, reg2, reg3, reg4, pred
frintn \reg1, \pred/M, \reg1
frintn \reg2, \pred/M, \reg2
frintn \reg3, \pred/M, \reg3
frintn \reg4, \pred/M, \reg4
.endm

.macro fcvtzs_convert, reg1, reg2, reg3, reg4, pred
fcvtzs \reg1, \pred/M, \reg1
fcvtzs \reg2, \pred/M, \reg2
fcvtzs \reg3, \pred/M, \reg3
fcvtzs \reg4, \pred/M, \reg4
.endm

.macro add____convert, reg1, reg2, reg3, reg4, reg5
add \reg1, \reg1, \reg5
add \reg2, \reg2, \reg5
add \reg3, \reg3, \reg5
add \reg4, \reg4, \reg5
.endm
 
.macro sclamp_convert2, reg1, reg2, reg3, reg4, min_val, max_val, pred
SMIN \reg1, \pred/M, \reg1, \min_val
SMIN \reg2, \pred/M, \reg2, \min_val
SMIN \reg3, \pred/M, \reg3, \min_val
SMIN \reg4, \pred/M, \reg4, \min_val
SMAX \reg1, \pred/M, \reg1, \max_val
SMAX \reg2, \pred/M, \reg2, \max_val
SMAX \reg3, \pred/M, \reg3, \max_val
SMAX \reg4, \pred/M, \reg4, \max_val
.endm




.macro sclamp_convert, reg1, reg2, reg3, reg4, min_val, max_val, pred
SCLAMP \reg1,  \min_val, \max_val
SCLAMP \reg2,  \min_val, \max_val
SCLAMP \reg3,  \min_val, \max_val
SCLAMP \reg4,  \min_val, \max_val
.endm


.macro scvtf_convert, reg1, reg2, reg3, reg4, pred
scvtf \reg1, \pred/M, \reg1
scvtf \reg2, \pred/M, \reg2
scvtf \reg3, \pred/M, \reg3
scvtf \reg4, \pred/M, \reg4
.endm

.macro move_tile_vector, tile, reg1, reg2, reg3, reg4, pred, addr
mova \reg1, \pred/M,  \tile[\addr, 0]
mova \reg2, \pred/M,  \tile[\addr, 1]
mova \reg3, \pred/M,  \tile[\addr, 2]
mova \reg4, \pred/M,  \tile[\addr, 3]
.endm

.macro load_data_2, reg1, reg2, pred, addr
ld1b \reg1, \pred/Z, [\addr]
ld1b \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
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
