//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//



#if defined(_MSC_VER)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme1_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme1_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme1_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme1_mla)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    SMSTART 
    mov x8, #0x0
    ldr x5, [x0, #0x18]
    cntw x6, ALL, MUL #4
    ptrue p1.b
    ldr x7, [x0, #0x20]
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x17, x5, x6
    ldr x20, [x0, #0x28]
    sub x17, x17, #0x1
    ldr x16, [x0, #0x8]
    udiv x17, x17, x6
    ldr x15, [x0, #0x30]
    mov x14, x21
    add x21, x17, #0x3
    mov x13, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x6
    mul x21, x21, x7
    lsl x21, x21, #0x2
KAI_ASM_LABEL(label_1)  // RHS size check loop
    cmp x21, #0x200, LSL #12
    blt label_2
    tbnz x21, #0, label_3
    lsr x21, x21, #0x1
    lsl x22, x22, #0x1
    b label_1
KAI_ASM_LABEL(label_2)  // RHS do prefetch
    lsl x20, x21, #0x26
    sub x22, x22, #0x1
    lsl x22, x22, #0x16
    orr x21, x21, x20
    orr x21, x21, x22
    rprfm #2, x21, [x14]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    mov x12, x7
    cntw x20, ALL, MUL #2
    lsl x12, x12, #0x2
    add x12, x12, #0x4
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_22
    cmp x17, #0x2
    bgt label_16
    beq label_10
    cntw x20, ALL, MUL #2
    add x22, x14, x12
    ld1w z8.s, p1/Z, [x14]
    ld1w z9.s, p1/Z, [x14, #1, MUL VL]
    cmp x5, x20
    mov x11, x7
    csel x22, x22, x14, GT
    mov x21, x5
    ld1w z10.s, p1/Z, [x22]
    ld1w z11.s, p1/Z, [x22, #1, MUL VL]
    mov x10, x16
    lsl x20, x7, #0x2
    mov x3, xzr
    whilelt p4.s, x3, x21    
    incw x3
    whilelt p5.s, x3, x21    
    incw x3
    whilelt p6.s, x3, x21    
    incw x3
    whilelt p7.s, x3, x21  
    cmp x11, #0x4
    rprfm pldkeep, x20, [x10]
    addvl x14, x14, #2
    addvl x22, x22, #2
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x11
    ld1rqw { z3.s }, p0/Z, [x10]
    sub x11, x11, #0x4
    add x10, x10, #0x10
    cmp x11, #0x4
    ldnt1w z4.s, p1/Z, [x14]
    ldnt1w z5.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z6.s, p1/Z, [x22]
    ldnt1w z7.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmla z8.s, z4.s, z3.s[0]
    fmla z9.s, z5.s, z3.s[0]
    fmla z10.s, z6.s, z3.s[0]
    fmla z11.s, z7.s, z3.s[0]
    ldnt1w z4.s, p1/Z, [x14]
    ldnt1w z5.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z6.s, p1/Z, [x22]
    ldnt1w z7.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmla z8.s, z4.s, z3.s[1]
    fmla z9.s, z5.s, z3.s[1]
    fmla z10.s, z6.s, z3.s[1]
    fmla z11.s, z7.s, z3.s[1]
    ldnt1w z4.s, p1/Z, [x14]
    ldnt1w z5.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z6.s, p1/Z, [x22]
    ldnt1w z7.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmla z8.s, z4.s, z3.s[2]
    fmla z9.s, z5.s, z3.s[2]
    fmla z10.s, z6.s, z3.s[2]
    fmla z11.s, z7.s, z3.s[2]
    ldnt1w z4.s, p1/Z, [x14]
    ldnt1w z5.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z6.s, p1/Z, [x22]
    ldnt1w z7.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmla z8.s, z4.s, z3.s[3]
    fmla z9.s, z5.s, z3.s[3]
    fmla z10.s, z6.s, z3.s[3]
    fmla z11.s, z7.s, z3.s[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x11
    subs x11, x11, #0x1
    ld1rqw { z3.s }, p0/Z, [x10]
    ldnt1w z4.s, p1/Z, [x14]
    ldnt1w z5.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z6.s, p1/Z, [x22]
    ldnt1w z7.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmla z8.s, z4.s, z3.s[0]
    fmla z9.s, z5.s, z3.s[0]
    fmla z10.s, z6.s, z3.s[0]
    fmla z11.s, z7.s, z3.s[0]
    ble label_7
    subs x11, x11, #0x1
    ldnt1w z4.s, p1/Z, [x14]
    ldnt1w z5.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z6.s, p1/Z, [x22]
    ldnt1w z7.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmla z8.s, z4.s, z3.s[1]
    fmla z9.s, z5.s, z3.s[1]
    fmla z10.s, z6.s, z3.s[1]
    fmla z11.s, z7.s, z3.s[1]
    ble label_7
    subs x11, x11, #0x1
    ldnt1w z4.s, p1/Z, [x14]
    ldnt1w z5.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z6.s, p1/Z, [x22]
    ldnt1w z7.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmla z8.s, z4.s, z3.s[2]
    fmla z9.s, z5.s, z3.s[2]
    fmla z10.s, z6.s, z3.s[2]
    fmla z11.s, z7.s, z3.s[2]
    ble label_7
    ldnt1w z4.s, p1/Z, [x14]
    ldnt1w z5.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z6.s, p1/Z, [x22]
    ldnt1w z7.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmla z8.s, z4.s, z3.s[3]
    fmla z9.s, z5.s, z3.s[3]
    fmla z10.s, z6.s, z3.s[3]
    fmla z11.s, z7.s, z3.s[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x15, #1, label_8
    add x21, x0, #0x4
    add x20, x0, #0x0
    ld1rw { z21.s }, p1/Z, [x21]
    ld1rw { z29.s }, p1/Z, [x20]
    FMIN z8.s, p1/M, z8.s, z29.s
    FMIN z9.s, p1/M, z9.s, z29.s
    FMIN z10.s, p1/M, z10.s, z29.s
    FMIN z11.s, p1/M, z11.s, z29.s
    FMAX z8.s, p1/M, z8.s, z21.s
    FMAX z9.s, p1/M, z9.s, z21.s
    FMAX z10.s, p1/M, z10.s, z21.s
    FMAX z11.s, p1/M, z11.s, z21.s
    st1w z8.s, p4, [x13]
    st1w z9.s, p5, [x13, #1, MUL VL]
    st1w z10.s, p6, [x13, #2, MUL VL]
    st1w z11.s, p7, [x13, #3, MUL VL]
    addvl x13, x13, #4
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    st1w z8.s, p4, [x13]
    st1w z9.s, p5, [x13, #1, MUL VL]
    st1w z10.s, p6, [x13, #2, MUL VL]
    st1w z11.s, p7, [x13, #3, MUL VL]
    addvl x13, x13, #4
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    add x24, x14, x12, LSL #1
    cntw x20, ALL, MUL #6
    ld1w z4.s, p1/Z, [x14]
    ld1w z5.s, p1/Z, [x14, #1, MUL VL]
    add x23, x24, x12                            
    cmp x5, x20                                  
    ld1w z28.s, p1/Z, [x24]
    ld1w z29.s, p1/Z, [x24, #1, MUL VL]
    add x22, x14, x12                           
    csel x23, x23, x14, GT                      
    ld1w z6.s, p1/Z, [x22]
    ld1w z7.s, p1/Z, [x22, #1, MUL VL]
    mov x11, x7                                  
    sub x21, x5, x6                              
    ld1w z30.s, p1/Z, [x23]
    ld1w z31.s, p1/Z, [x23, #1, MUL VL]
    mov x10, x16
    lsl x20, x7, #0x2
    mov x3, xzr
    whilelt p4.s, x3, x21    
    incw x3
    whilelt p5.s, x3, x21    
    incw x3
    whilelt p6.s, x3, x21    
    incw x3
    whilelt p7.s, x3, x21 
    cmp x11, #0x4
    rprfm pldkeep, x20, [x10]
    addvl x14, x14, #2
    addvl x22, x22, #2
    addvl x24, x24, #2
    addvl x23, x23, #2
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x11
    ld1rqw { z3.s }, p0/Z, [x10]
    sub x11, x11, #0x4
    add x10, x10, #0x10
    cmp x11, #0x4
    ldnt1w z12.s, p1/Z, [x14]
    ldnt1w z13.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z14.s, p1/Z, [x22]
    ldnt1w z15.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z20.s, p1/Z, [x24]
    ldnt1w z21.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmla z4.s, z12.s, z3.s[0]
    fmla z5.s, z13.s, z3.s[0]
    fmla z6.s, z14.s, z3.s[0]
    fmla z7.s, z15.s, z3.s[0]
    fmla z28.s, z20.s, z3.s[0]
    fmla z29.s, z21.s, z3.s[0]
    fmla z30.s, z22.s, z3.s[0]
    fmla z31.s, z23.s, z3.s[0]
    ldnt1w z12.s, p1/Z, [x14]
    ldnt1w z13.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z14.s, p1/Z, [x22]
    ldnt1w z15.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z20.s, p1/Z, [x24]
    ldnt1w z21.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmla z4.s, z12.s, z3.s[1]
    fmla z5.s, z13.s, z3.s[1]
    fmla z6.s, z14.s, z3.s[1]
    fmla z7.s, z15.s, z3.s[1]
    fmla z28.s, z20.s, z3.s[1]
    fmla z29.s, z21.s, z3.s[1]
    fmla z30.s, z22.s, z3.s[1]
    fmla z31.s, z23.s, z3.s[1]
    ldnt1w z12.s, p1/Z, [x14]
    ldnt1w z13.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z14.s, p1/Z, [x22]
    ldnt1w z15.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z20.s, p1/Z, [x24]
    ldnt1w z21.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmla z4.s, z12.s, z3.s[2]
    fmla z5.s, z13.s, z3.s[2]
    fmla z6.s, z14.s, z3.s[2]
    fmla z7.s, z15.s, z3.s[2]
    fmla z28.s, z20.s, z3.s[2]
    fmla z29.s, z21.s, z3.s[2]
    fmla z30.s, z22.s, z3.s[2]
    fmla z31.s, z23.s, z3.s[2]
    ldnt1w z12.s, p1/Z, [x14]
    ldnt1w z13.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z14.s, p1/Z, [x22]
    ldnt1w z15.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z20.s, p1/Z, [x24]
    ldnt1w z21.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmla z4.s, z12.s, z3.s[3]
    fmla z5.s, z13.s, z3.s[3]
    fmla z6.s, z14.s, z3.s[3]
    fmla z7.s, z15.s, z3.s[3]
    fmla z28.s, z20.s, z3.s[3]
    fmla z29.s, z21.s, z3.s[3]
    fmla z30.s, z22.s, z3.s[3]
    fmla z31.s, z23.s, z3.s[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x11
    subs x11, x11, #0x1
    ld1rqw { z3.s }, p0/Z, [x10]
    ldnt1w z12.s, p1/Z, [x14]
    ldnt1w z13.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z14.s, p1/Z, [x22]
    ldnt1w z15.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z20.s, p1/Z, [x24]
    ldnt1w z21.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmla z4.s, z12.s, z3.s[0]
    fmla z5.s, z13.s, z3.s[0]
    fmla z6.s, z14.s, z3.s[0]
    fmla z7.s, z15.s, z3.s[0]
    fmla z28.s, z20.s, z3.s[0]
    fmla z29.s, z21.s, z3.s[0]
    fmla z30.s, z22.s, z3.s[0]
    fmla z31.s, z23.s, z3.s[0]
    ble label_13
    subs x11, x11, #0x1
    ldnt1w z12.s, p1/Z, [x14]
    ldnt1w z13.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z14.s, p1/Z, [x22]
    ldnt1w z15.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z20.s, p1/Z, [x24]
    ldnt1w z21.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmla z4.s, z12.s, z3.s[1]
    fmla z5.s, z13.s, z3.s[1]
    fmla z6.s, z14.s, z3.s[1]
    fmla z7.s, z15.s, z3.s[1]
    fmla z28.s, z20.s, z3.s[1]
    fmla z29.s, z21.s, z3.s[1]
    fmla z30.s, z22.s, z3.s[1]
    fmla z31.s, z23.s, z3.s[1]
    ble label_13
    subs x11, x11, #0x1
    ldnt1w z12.s, p1/Z, [x14]
    ldnt1w z13.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z14.s, p1/Z, [x22]
    ldnt1w z15.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z20.s, p1/Z, [x24]
    ldnt1w z21.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmla z4.s, z12.s, z3.s[2]
    fmla z5.s, z13.s, z3.s[2]
    fmla z6.s, z14.s, z3.s[2]
    fmla z7.s, z15.s, z3.s[2]
    fmla z28.s, z20.s, z3.s[2]
    fmla z29.s, z21.s, z3.s[2]
    fmla z30.s, z22.s, z3.s[2]
    fmla z31.s, z23.s, z3.s[2]
    ble label_13
    ldnt1w z12.s, p1/Z, [x14]
    ldnt1w z13.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z14.s, p1/Z, [x22]
    ldnt1w z15.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z20.s, p1/Z, [x24]
    ldnt1w z21.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmla z4.s, z12.s, z3.s[3]
    fmla z5.s, z13.s, z3.s[3]
    fmla z6.s, z14.s, z3.s[3]
    fmla z7.s, z15.s, z3.s[3]
    fmla z28.s, z20.s, z3.s[3]
    fmla z29.s, z21.s, z3.s[3]
    fmla z30.s, z22.s, z3.s[3]
    fmla z31.s, z23.s, z3.s[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x15, #1, label_14
    add x21, x0, #0x4
    add x20, x0, #0x0
    ld1rw { z17.s }, p1/Z, [x21]
    ld1rw { z9.s }, p1/Z,  [x20]
    FMIN z4.s, p1/M, z4.s, z9.s
    FMIN z5.s, p1/M, z5.s, z9.s
    FMIN z6.s, p1/M, z6.s, z9.s
    FMIN z7.s, p1/M, z7.s, z9.s
    FMAX z4.s, p1/M, z4.s, z17.s
    FMAX z5.s, p1/M, z5.s, z17.s
    FMAX z6.s, p1/M, z6.s, z17.s
    FMAX z7.s, p1/M, z7.s, z17.s
    FMIN z28.s, p1/M, z28.s, z9.s
    FMIN z29.s, p1/M, z29.s, z9.s
    FMIN z30.s, p1/M, z30.s, z9.s
    FMIN z31.s, p1/M, z31.s, z9.s
    FMAX z28.s, p1/M, z28.s, z17.s
    FMAX z29.s, p1/M, z29.s, z17.s
    FMAX z30.s, p1/M, z30.s, z17.s
    FMAX z31.s, p1/M, z31.s, z17.s
    st1w z4.s, p1, [x13]
    st1w z5.s, p1, [x13, #1, MUL VL]
    st1w z6.s, p1, [x13, #2, MUL VL]
    st1w z7.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z28.s, p4, [x13]
    st1w z29.s, p5, [x13, #1, MUL VL]
    st1w z30.s, p6, [x13, #2, MUL VL]
    st1w z31.s, p7, [x13, #3, MUL VL]
    addvl x13, x13, #4
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    st1w z4.s, p1, [x13]
    st1w z5.s, p1, [x13, #1, MUL VL]
    st1w z6.s, p1, [x13, #2, MUL VL]
    st1w z7.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z28.s, p4, [x13]
    st1w z29.s, p5, [x13, #1, MUL VL]
    st1w z30.s, p6, [x13, #2, MUL VL]
    st1w z31.s, p7, [x13, #3, MUL VL]
    addvl x13, x13, #4
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    add x26, x14, x12, LSL #2
    cntw x20, ALL, MUL #10
    ld1w z8.s, p1/Z, [x14]
    ld1w z9.s, p1/Z, [x14, #1, MUL VL]
    add x25, x14, x12, LSL #1                          
    add x24, x26, x12                                  
    ld1w z16.s, p1/Z, [x26]
    ld1w z17.s, p1/Z, [x26, #1, MUL VL]
    cmp x5, x20                                        
    add x23, x14, x12                                  
    ld1w z12.s, p1/Z, [x25]
    ld1w z13.s, p1/Z, [x25, #1, MUL VL]
    add x22, x25, x12                                  
    csel x24, x24, x14, GT                             
    ld1w z10.s, p1/Z, [x23]
    ld1w z11.s, p1/Z, [x23, #1, MUL VL]
    mov x20, #0x2                                      
    ld1w z14.s, p1/Z, [x22]
    ld1w z15.s, p1/Z, [x22, #1, MUL VL]
    mov x11, x7                                        
    ld1w z18.s, p1/Z, [x24]
    ld1w z19.s, p1/Z, [x24, #1, MUL VL]
    msub x21, x6, x20, x5
    mov x10, x16
    lsl x20, x7, #0x2
    mov x3, xzr
    whilelt p4.s, x3, x21    
    incw x3
    whilelt p5.s, x3, x21    
    incw x3
    whilelt p6.s, x3, x21    
    incw x3
    whilelt p7.s, x3, x21 
    cmp x11, #0x4
    rprfm pldkeep, x20, [x10]
    addvl x14, x14, #2
    addvl x23, x23, #2
    addvl x25, x25, #2
    addvl x22, x22, #2
    addvl x26, x26, #2
    addvl x24, x24, #2
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x11
    ld1rqw { z3.s }, p0/Z, [x10]
    sub x11, x11, #0x4
    add x10, x10, #0x10
    cmp x11, #0x4
    ldnt1w z20.s, p1/Z, [x14]
    ldnt1w z21.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z24.s, p1/Z, [x25]
    ldnt1w z25.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z26.s, p1/Z, [x22]
    ldnt1w z27.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z28.s, p1/Z, [x26]
    ldnt1w z29.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z20.s, z3.s[0]
    fmla z9.s, z21.s, z3.s[0]
    fmla z10.s, z22.s, z3.s[0]
    fmla z11.s, z23.s, z3.s[0]
    fmla z12.s, z24.s, z3.s[0]
    fmla z13.s, z25.s, z3.s[0]
    fmla z14.s, z26.s, z3.s[0]
    fmla z15.s, z27.s, z3.s[0]
    fmla z16.s, z28.s, z3.s[0]
    fmla z17.s, z29.s, z3.s[0]
    fmla z18.s, z30.s, z3.s[0]
    fmla z19.s, z31.s, z3.s[0]
    ldnt1w z20.s, p1/Z, [x14]
    ldnt1w z21.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z24.s, p1/Z, [x25]
    ldnt1w z25.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z26.s, p1/Z, [x22]
    ldnt1w z27.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z28.s, p1/Z, [x26]
    ldnt1w z29.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z20.s, z3.s[1]
    fmla z9.s, z21.s, z3.s[1]
    fmla z10.s, z22.s, z3.s[1]
    fmla z11.s, z23.s, z3.s[1]
    fmla z12.s, z24.s, z3.s[1]
    fmla z13.s, z25.s, z3.s[1]
    fmla z14.s, z26.s, z3.s[1]
    fmla z15.s, z27.s, z3.s[1]
    fmla z16.s, z28.s, z3.s[1]
    fmla z17.s, z29.s, z3.s[1]
    fmla z18.s, z30.s, z3.s[1]
    fmla z19.s, z31.s, z3.s[1]
    ldnt1w z20.s, p1/Z, [x14]
    ldnt1w z21.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z24.s, p1/Z, [x25]
    ldnt1w z25.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z26.s, p1/Z, [x22]
    ldnt1w z27.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z28.s, p1/Z, [x26]
    ldnt1w z29.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z20.s, z3.s[2]
    fmla z9.s, z21.s, z3.s[2]
    fmla z10.s, z22.s, z3.s[2]
    fmla z11.s, z23.s, z3.s[2]
    fmla z12.s, z24.s, z3.s[2]
    fmla z13.s, z25.s, z3.s[2]
    fmla z14.s, z26.s, z3.s[2]
    fmla z15.s, z27.s, z3.s[2]
    fmla z16.s, z28.s, z3.s[2]
    fmla z17.s, z29.s, z3.s[2]
    fmla z18.s, z30.s, z3.s[2]
    fmla z19.s, z31.s, z3.s[2]
    ldnt1w z20.s, p1/Z, [x14]
    ldnt1w z21.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z24.s, p1/Z, [x25]
    ldnt1w z25.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z26.s, p1/Z, [x22]
    ldnt1w z27.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z28.s, p1/Z, [x26]
    ldnt1w z29.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z20.s, z3.s[3]
    fmla z9.s, z21.s, z3.s[3]
    fmla z10.s, z22.s, z3.s[3]
    fmla z11.s, z23.s, z3.s[3]
    fmla z12.s, z24.s, z3.s[3]
    fmla z13.s, z25.s, z3.s[3]
    fmla z14.s, z26.s, z3.s[3]
    fmla z15.s, z27.s, z3.s[3]
    fmla z16.s, z28.s, z3.s[3]
    fmla z17.s, z29.s, z3.s[3]
    fmla z18.s, z30.s, z3.s[3]
    fmla z19.s, z31.s, z3.s[3]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x11
    subs x11, x11, #0x1
    ld1rqw { z3.s }, p0/Z, [x10]
    ldnt1w z20.s, p1/Z, [x14]
    ldnt1w z21.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z24.s, p1/Z, [x25]
    ldnt1w z25.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z26.s, p1/Z, [x22]
    ldnt1w z27.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z28.s, p1/Z, [x26]
    ldnt1w z29.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z20.s, z3.s[0]
    fmla z9.s, z21.s, z3.s[0]
    fmla z10.s, z22.s, z3.s[0]
    fmla z11.s, z23.s, z3.s[0]
    fmla z12.s, z24.s, z3.s[0]
    fmla z13.s, z25.s, z3.s[0]
    fmla z14.s, z26.s, z3.s[0]
    fmla z15.s, z27.s, z3.s[0]
    fmla z16.s, z28.s, z3.s[0]
    fmla z17.s, z29.s, z3.s[0]
    fmla z18.s, z30.s, z3.s[0]
    fmla z19.s, z31.s, z3.s[0]
    ble label_19
    subs x11, x11, #0x1
    ldnt1w z20.s, p1/Z, [x14]
    ldnt1w z21.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z24.s, p1/Z, [x25]
    ldnt1w z25.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z26.s, p1/Z, [x22]
    ldnt1w z27.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z28.s, p1/Z, [x26]
    ldnt1w z29.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z20.s, z3.s[1]
    fmla z9.s, z21.s, z3.s[1]
    fmla z10.s, z22.s, z3.s[1]
    fmla z11.s, z23.s, z3.s[1]
    fmla z12.s, z24.s, z3.s[1]
    fmla z13.s, z25.s, z3.s[1]
    fmla z14.s, z26.s, z3.s[1]
    fmla z15.s, z27.s, z3.s[1]
    fmla z16.s, z28.s, z3.s[1]
    fmla z17.s, z29.s, z3.s[1]
    fmla z18.s, z30.s, z3.s[1]
    fmla z19.s, z31.s, z3.s[1]
    ble label_19
    subs x11, x11, #0x1
    ldnt1w z20.s, p1/Z, [x14]
    ldnt1w z21.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z24.s, p1/Z, [x25]
    ldnt1w z25.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z26.s, p1/Z, [x22]
    ldnt1w z27.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z28.s, p1/Z, [x26]
    ldnt1w z29.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z20.s, z3.s[2]
    fmla z9.s, z21.s, z3.s[2]
    fmla z10.s, z22.s, z3.s[2]
    fmla z11.s, z23.s, z3.s[2]
    fmla z12.s, z24.s, z3.s[2]
    fmla z13.s, z25.s, z3.s[2]
    fmla z14.s, z26.s, z3.s[2]
    fmla z15.s, z27.s, z3.s[2]
    fmla z16.s, z28.s, z3.s[2]
    fmla z17.s, z29.s, z3.s[2]
    fmla z18.s, z30.s, z3.s[2]
    fmla z19.s, z31.s, z3.s[2]
    ble label_19
    ldnt1w z20.s, p1/Z, [x14]
    ldnt1w z21.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z22.s, p1/Z, [x23]
    ldnt1w z23.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z24.s, p1/Z, [x25]
    ldnt1w z25.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z26.s, p1/Z, [x22]
    ldnt1w z27.s, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1w z28.s, p1/Z, [x26]
    ldnt1w z29.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z20.s, z3.s[3]
    fmla z9.s, z21.s, z3.s[3]
    fmla z10.s, z22.s, z3.s[3]
    fmla z11.s, z23.s, z3.s[3]
    fmla z12.s, z24.s, z3.s[3]
    fmla z13.s, z25.s, z3.s[3]
    fmla z14.s, z26.s, z3.s[3]
    fmla z15.s, z27.s, z3.s[3]
    fmla z16.s, z28.s, z3.s[3]
    fmla z17.s, z29.s, z3.s[3]
    fmla z18.s, z30.s, z3.s[3]
    fmla z19.s, z31.s, z3.s[3]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x15, #1, label_20
    add x21, x0, #0x4
    add x20, x0, #0x0
    ld1rw { z1.s }, p1/Z, [x21]
    ld1rw { z0.s }, p1/Z, [x20]
    FMIN z8.s, p1/M, z8.s, z0.s
    FMIN z9.s, p1/M, z9.s, z0.s
    FMIN z10.s, p1/M, z10.s, z0.s
    FMIN z11.s, p1/M, z11.s, z0.s
    FMAX z8.s, p1/M, z8.s, z1.s
    FMAX z9.s, p1/M, z9.s, z1.s
    FMAX z10.s, p1/M, z10.s, z1.s
    FMAX z11.s, p1/M, z11.s, z1.s
    FMIN z12.s, p1/M, z12.s, z0.s
    FMIN z13.s, p1/M, z13.s, z0.s
    FMIN z14.s, p1/M, z14.s, z0.s
    FMIN z15.s, p1/M, z15.s, z0.s
    FMAX z12.s, p1/M, z12.s, z1.s
    FMAX z13.s, p1/M, z13.s, z1.s
    FMAX z14.s, p1/M, z14.s, z1.s
    FMAX z15.s, p1/M, z15.s, z1.s
    FMIN z16.s, p1/M, z16.s, z0.s
    FMIN z17.s, p1/M, z17.s, z0.s
    FMIN z18.s, p1/M, z18.s, z0.s
    FMIN z19.s, p1/M, z19.s, z0.s
    FMAX z16.s, p1/M, z16.s, z1.s
    FMAX z17.s, p1/M, z17.s, z1.s
    FMAX z18.s, p1/M, z18.s, z1.s
    FMAX z19.s, p1/M, z19.s, z1.s
    st1w z8.s, p1, [x13]
    st1w z9.s, p1, [x13, #1, MUL VL]
    st1w z10.s, p1, [x13, #2, MUL VL]
    st1w z11.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z12.s, p1, [x13]
    st1w z13.s, p1, [x13, #1, MUL VL]
    st1w z14.s, p1, [x13, #2, MUL VL]
    st1w z15.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z16.s, p4, [x13]
    st1w z17.s, p5, [x13, #1, MUL VL]
    st1w z18.s, p6, [x13, #2, MUL VL]
    st1w z19.s, p7, [x13, #3, MUL VL]
    addvl x13, x13, #4
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    st1w z8.s, p1, [x13]
    st1w z9.s, p1, [x13, #1, MUL VL]
    st1w z10.s, p1, [x13, #2, MUL VL]
    st1w z11.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z12.s, p1, [x13]
    st1w z13.s, p1, [x13, #1, MUL VL]
    st1w z14.s, p1, [x13, #2, MUL VL]
    st1w z15.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z16.s, p4, [x13]
    st1w z17.s, p5, [x13, #1, MUL VL]
    st1w z18.s, p6, [x13, #2, MUL VL]
    st1w z19.s, p7, [x13, #3, MUL VL]
    addvl x13, x13, #4
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    add x9, x14, x12, LSL #2
    cntw x20, ALL, MUL #14
    ld1w z8.s, p1/Z, [x14]
    ld1w z9.s, p1/Z, [x14, #1, MUL VL]
    add x28, x9, x12, LSL #1
    add x27, x14, x12, LSL #1
    ld1w z16.s, p1/Z, [x9]
    ld1w z17.s, p1/Z, [x9, #1, MUL VL]
    add x26, x28, x12
    cmp x5, x20
    ld1w z12.s, p1/Z, [x27]
    ld1w z13.s, p1/Z, [x27, #1, MUL VL]
    add x25, x14, x12
    add x24, x27, x12
    ld1w z20.s, p1/Z, [x28]
    ld1w z21.s, p1/Z, [x28, #1, MUL VL]
    add x23, x9, x12
    csel x26, x26, x14, GT
    ld1w z10.s, p1/Z, [x25]
    ld1w z11.s, p1/Z, [x25, #1, MUL VL]
    mov x20, #0x3
    ld1w z14.s, p1/Z, [x24]
    ld1w z15.s, p1/Z, [x24, #1, MUL VL]
    mov x11, x7
    ld1w z18.s, p1/Z, [x23]
    ld1w z19.s, p1/Z, [x23, #1, MUL VL]
    msub x21, x6, x20, x5
    mov x10, x16
    ld1w z22.s, p1/Z, [x26]
    ld1w z23.s, p1/Z, [x26, #1, MUL VL]
    lsl x20, x7, #0x2
    mov x3, xzr
    whilelt p4.s, x3, x21    
    incw x3
    whilelt p5.s, x3, x21    
    incw x3
    whilelt p6.s, x3, x21    
    incw x3
    whilelt p7.s, x3, x21 
    cmp x11, #0x4
    rprfm pldkeep, x20, [x10]
    add x22, x14, x12, LSL #3
    addvl x14, x14, #2
    addvl x25, x25, #2
    addvl x27, x27, #2
    addvl x24, x24, #2
    addvl x9, x9, #2
    addvl x23, x23, #2
    addvl x28, x28, #2
    addvl x26, x26, #2
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x11
    ld1rqw { z3.s }, p0/Z, [x10]
    sub x11, x11, #0x4
    add x10, x10, #0x10
    cmp x11, #0x4
    ldnt1w z24.s, p1/Z, [x14]
    ldnt1w z25.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z26.s, p1/Z, [x25]
    ldnt1w z27.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z28.s, p1/Z, [x27]
    ldnt1w z29.s, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z24.s, z3.s[0]
    fmla z9.s, z25.s, z3.s[0]
    fmla z10.s, z26.s, z3.s[0]
    fmla z11.s, z27.s, z3.s[0]
    fmla z12.s, z28.s, z3.s[0]
    fmla z13.s, z29.s, z3.s[0]
    fmla z14.s, z30.s, z3.s[0]
    fmla z15.s, z31.s, z3.s[0]
    ldnt1w z24.s, p1/Z, [x9]
    ldnt1w z25.s, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w z26.s, p1/Z, [x23]
    ldnt1w z27.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z28.s, p1/Z, [x28]
    ldnt1w z29.s, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1w z30.s, p1/Z, [x26]
    ldnt1w z31.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmla z16.s, z24.s, z3.s[0]
    fmla z17.s, z25.s, z3.s[0]
    fmla z18.s, z26.s, z3.s[0]
    fmla z19.s, z27.s, z3.s[0]
    fmla z20.s, z28.s, z3.s[0]
    fmla z21.s, z29.s, z3.s[0]
    fmla z22.s, z30.s, z3.s[0]
    fmla z23.s, z31.s, z3.s[0]
    ldnt1w z24.s, p1/Z, [x14]
    ldnt1w z25.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z26.s, p1/Z, [x25]
    ldnt1w z27.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z28.s, p1/Z, [x27]
    ldnt1w z29.s, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z24.s, z3.s[1]
    fmla z9.s, z25.s, z3.s[1]
    fmla z10.s, z26.s, z3.s[1]
    fmla z11.s, z27.s, z3.s[1]
    fmla z12.s, z28.s, z3.s[1]
    fmla z13.s, z29.s, z3.s[1]
    fmla z14.s, z30.s, z3.s[1]
    fmla z15.s, z31.s, z3.s[1]
    ldnt1w z24.s, p1/Z, [x9]
    ldnt1w z25.s, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w z26.s, p1/Z, [x23]
    ldnt1w z27.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z28.s, p1/Z, [x28]
    ldnt1w z29.s, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1w z30.s, p1/Z, [x26]
    ldnt1w z31.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmla z16.s, z24.s, z3.s[1]
    fmla z17.s, z25.s, z3.s[1]
    fmla z18.s, z26.s, z3.s[1]
    fmla z19.s, z27.s, z3.s[1]
    fmla z20.s, z28.s, z3.s[1]
    fmla z21.s, z29.s, z3.s[1]
    fmla z22.s, z30.s, z3.s[1]
    fmla z23.s, z31.s, z3.s[1]
    ldnt1w z24.s, p1/Z, [x14]
    ldnt1w z25.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z26.s, p1/Z, [x25]
    ldnt1w z27.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z28.s, p1/Z, [x27]
    ldnt1w z29.s, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z24.s, z3.s[2]
    fmla z9.s, z25.s, z3.s[2]
    fmla z10.s, z26.s, z3.s[2]
    fmla z11.s, z27.s, z3.s[2]
    fmla z12.s, z28.s, z3.s[2]
    fmla z13.s, z29.s, z3.s[2]
    fmla z14.s, z30.s, z3.s[2]
    fmla z15.s, z31.s, z3.s[2]
    ldnt1w z24.s, p1/Z, [x9]
    ldnt1w z25.s, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w z26.s, p1/Z, [x23]
    ldnt1w z27.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z28.s, p1/Z, [x28]
    ldnt1w z29.s, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1w z30.s, p1/Z, [x26]
    ldnt1w z31.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmla z16.s, z24.s, z3.s[2]
    fmla z17.s, z25.s, z3.s[2]
    fmla z18.s, z26.s, z3.s[2]
    fmla z19.s, z27.s, z3.s[2]
    fmla z20.s, z28.s, z3.s[2]
    fmla z21.s, z29.s, z3.s[2]
    fmla z22.s, z30.s, z3.s[2]
    fmla z23.s, z31.s, z3.s[2]
    ldnt1w z24.s, p1/Z, [x14]
    ldnt1w z25.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z26.s, p1/Z, [x25]
    ldnt1w z27.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z28.s, p1/Z, [x27]
    ldnt1w z29.s, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z24.s, z3.s[3]
    fmla z9.s, z25.s, z3.s[3]
    fmla z10.s, z26.s, z3.s[3]
    fmla z11.s, z27.s, z3.s[3]
    fmla z12.s, z28.s, z3.s[3]
    fmla z13.s, z29.s, z3.s[3]
    fmla z14.s, z30.s, z3.s[3]
    fmla z15.s, z31.s, z3.s[3]
    ldnt1w z24.s, p1/Z, [x9]
    ldnt1w z25.s, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w z26.s, p1/Z, [x23]
    ldnt1w z27.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z28.s, p1/Z, [x28]
    ldnt1w z29.s, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1w z30.s, p1/Z, [x26]
    ldnt1w z31.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmla z16.s, z24.s, z3.s[3]
    fmla z17.s, z25.s, z3.s[3]
    fmla z18.s, z26.s, z3.s[3]
    fmla z19.s, z27.s, z3.s[3]
    fmla z20.s, z28.s, z3.s[3]
    fmla z21.s, z29.s, z3.s[3]
    fmla z22.s, z30.s, z3.s[3]
    fmla z23.s, z31.s, z3.s[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x11
    subs x11, x11, #0x1
    ld1rqw { z3.s }, p0/Z, [x10]
    ldnt1w z24.s, p1/Z, [x14]
    ldnt1w z25.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z26.s, p1/Z, [x25]
    ldnt1w z27.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z28.s, p1/Z, [x27]
    ldnt1w z29.s, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z24.s, z3.s[0]
    fmla z9.s, z25.s, z3.s[0]
    fmla z10.s, z26.s, z3.s[0]
    fmla z11.s, z27.s, z3.s[0]
    fmla z12.s, z28.s, z3.s[0]
    fmla z13.s, z29.s, z3.s[0]
    fmla z14.s, z30.s, z3.s[0]
    fmla z15.s, z31.s, z3.s[0]
    ldnt1w z24.s, p1/Z, [x9]
    ldnt1w z25.s, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w z26.s, p1/Z, [x23]
    ldnt1w z27.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z28.s, p1/Z, [x28]
    ldnt1w z29.s, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1w z30.s, p1/Z, [x26]
    ldnt1w z31.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmla z16.s, z24.s, z3.s[0]
    fmla z17.s, z25.s, z3.s[0]
    fmla z18.s, z26.s, z3.s[0]
    fmla z19.s, z27.s, z3.s[0]
    fmla z20.s, z28.s, z3.s[0]
    fmla z21.s, z29.s, z3.s[0]
    fmla z22.s, z30.s, z3.s[0]
    fmla z23.s, z31.s, z3.s[0]
    ble label_25
    subs x11, x11, #0x1
    ldnt1w z24.s, p1/Z, [x14]
    ldnt1w z25.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z26.s, p1/Z, [x25]
    ldnt1w z27.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z28.s, p1/Z, [x27]
    ldnt1w z29.s, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z24.s, z3.s[1]
    fmla z9.s, z25.s, z3.s[1]
    fmla z10.s, z26.s, z3.s[1]
    fmla z11.s, z27.s, z3.s[1]
    fmla z12.s, z28.s, z3.s[1]
    fmla z13.s, z29.s, z3.s[1]
    fmla z14.s, z30.s, z3.s[1]
    fmla z15.s, z31.s, z3.s[1]
    ldnt1w z24.s, p1/Z, [x9]
    ldnt1w z25.s, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w z26.s, p1/Z, [x23]
    ldnt1w z27.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z28.s, p1/Z, [x28]
    ldnt1w z29.s, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1w z30.s, p1/Z, [x26]
    ldnt1w z31.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmla z16.s, z24.s, z3.s[1]
    fmla z17.s, z25.s, z3.s[1]
    fmla z18.s, z26.s, z3.s[1]
    fmla z19.s, z27.s, z3.s[1]
    fmla z20.s, z28.s, z3.s[1]
    fmla z21.s, z29.s, z3.s[1]
    fmla z22.s, z30.s, z3.s[1]
    fmla z23.s, z31.s, z3.s[1]
    ble label_25
    subs x11, x11, #0x1
    ldnt1w z24.s, p1/Z, [x14]
    ldnt1w z25.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z26.s, p1/Z, [x25]
    ldnt1w z27.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z28.s, p1/Z, [x27]
    ldnt1w z29.s, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z24.s, z3.s[2]
    fmla z9.s, z25.s, z3.s[2]
    fmla z10.s, z26.s, z3.s[2]
    fmla z11.s, z27.s, z3.s[2]
    fmla z12.s, z28.s, z3.s[2]
    fmla z13.s, z29.s, z3.s[2]
    fmla z14.s, z30.s, z3.s[2]
    fmla z15.s, z31.s, z3.s[2]
    ldnt1w z24.s, p1/Z, [x9]
    ldnt1w z25.s, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w z26.s, p1/Z, [x23]
    ldnt1w z27.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z28.s, p1/Z, [x28]
    ldnt1w z29.s, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1w z30.s, p1/Z, [x26]
    ldnt1w z31.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmla z16.s, z24.s, z3.s[2]
    fmla z17.s, z25.s, z3.s[2]
    fmla z18.s, z26.s, z3.s[2]
    fmla z19.s, z27.s, z3.s[2]
    fmla z20.s, z28.s, z3.s[2]
    fmla z21.s, z29.s, z3.s[2]
    fmla z22.s, z30.s, z3.s[2]
    fmla z23.s, z31.s, z3.s[2]
    ble label_25
    ldnt1w z24.s, p1/Z, [x14]
    ldnt1w z25.s, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1w z26.s, p1/Z, [x25]
    ldnt1w z27.s, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1w z28.s, p1/Z, [x27]
    ldnt1w z29.s, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1w z30.s, p1/Z, [x24]
    ldnt1w z31.s, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmla z8.s, z24.s, z3.s[3]
    fmla z9.s, z25.s, z3.s[3]
    fmla z10.s, z26.s, z3.s[3]
    fmla z11.s, z27.s, z3.s[3]
    fmla z12.s, z28.s, z3.s[3]
    fmla z13.s, z29.s, z3.s[3]
    fmla z14.s, z30.s, z3.s[3]
    fmla z15.s, z31.s, z3.s[3]
    ldnt1w z24.s, p1/Z, [x9]
    ldnt1w z25.s, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w z26.s, p1/Z, [x23]
    ldnt1w z27.s, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1w z28.s, p1/Z, [x28]
    ldnt1w z29.s, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1w z30.s, p1/Z, [x26]
    ldnt1w z31.s, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmla z16.s, z24.s, z3.s[3]
    fmla z17.s, z25.s, z3.s[3]
    fmla z18.s, z26.s, z3.s[3]
    fmla z19.s, z27.s, z3.s[3]
    fmla z20.s, z28.s, z3.s[3]
    fmla z21.s, z29.s, z3.s[3]
    fmla z22.s, z30.s, z3.s[3]
    fmla z23.s, z31.s, z3.s[3]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x15, #1, label_26
    add x21, x0, #0x4
    add x20, x0, #0x0
    ld1rw { z1.s }, p1/Z, [x21]
    ld1rw { z0.s }, p1/Z, [x20]
    FMIN z8.s, p1/M, z8.s, z0.s
    FMIN z9.s, p1/M, z9.s, z0.s
    FMIN z10.s, p1/M, z10.s, z0.s
    FMIN z11.s, p1/M, z11.s, z0.s
    FMAX z8.s, p1/M, z8.s, z1.s
    FMAX z9.s, p1/M, z9.s, z1.s
    FMAX z10.s, p1/M, z10.s, z1.s
    FMAX z11.s, p1/M, z11.s, z1.s
    FMIN z12.s, p1/M, z12.s, z0.s
    FMIN z13.s, p1/M, z13.s, z0.s
    FMIN z14.s, p1/M, z14.s, z0.s
    FMIN z15.s, p1/M, z15.s, z0.s
    FMAX z12.s, p1/M, z12.s, z1.s
    FMAX z13.s, p1/M, z13.s, z1.s
    FMAX z14.s, p1/M, z14.s, z1.s
    FMAX z15.s, p1/M, z15.s, z1.s
    FMIN z16.s, p1/M, z16.s, z0.s
    FMIN z17.s, p1/M, z17.s, z0.s
    FMIN z18.s, p1/M, z18.s, z0.s
    FMIN z19.s, p1/M, z19.s, z0.s
    FMAX z16.s, p1/M, z16.s, z1.s
    FMAX z17.s, p1/M, z17.s, z1.s
    FMAX z18.s, p1/M, z18.s, z1.s
    FMAX z19.s, p1/M, z19.s, z1.s
    FMIN z20.s, p1/M, z20.s, z0.s
    FMIN z21.s, p1/M, z21.s, z0.s
    FMIN z22.s, p1/M, z22.s, z0.s
    FMIN z23.s, p1/M, z23.s, z0.s
    FMAX z20.s, p1/M, z20.s, z1.s
    FMAX z21.s, p1/M, z21.s, z1.s
    FMAX z22.s, p1/M, z22.s, z1.s
    FMAX z23.s, p1/M, z23.s, z1.s
    st1w z8.s, p1, [x13]
    st1w z9.s, p1, [x13, #1, MUL VL]
    st1w z10.s, p1, [x13, #2, MUL VL]
    st1w z11.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z12.s, p1, [x13]
    st1w z13.s, p1, [x13, #1, MUL VL]
    st1w z14.s, p1, [x13, #2, MUL VL]
    st1w z15.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z16.s, p1, [x13]
    st1w z17.s, p1, [x13, #1, MUL VL]
    st1w z18.s, p1, [x13, #2, MUL VL]
    st1w z19.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z20.s, p4, [x13]
    st1w z21.s, p5, [x13, #1, MUL VL]
    st1w z22.s, p6, [x13, #2, MUL VL]
    st1w z23.s, p7, [x13, #3, MUL VL]
    addvl x13, x13, #4
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    st1w z8.s, p1, [x13]
    st1w z9.s, p1, [x13, #1, MUL VL]
    st1w z10.s, p1, [x13, #2, MUL VL]
    st1w z11.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z12.s, p1, [x13]
    st1w z13.s, p1, [x13, #1, MUL VL]
    st1w z14.s, p1, [x13, #2, MUL VL]
    st1w z15.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z16.s, p1, [x13]
    st1w z17.s, p1, [x13, #1, MUL VL]
    st1w z18.s, p1, [x13, #2, MUL VL]
    st1w z19.s, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    st1w z20.s, p4, [x13]
    st1w z21.s, p5, [x13, #1, MUL VL]
    st1w z22.s, p6, [x13, #2, MUL VL]
    st1w z23.s, p7, [x13, #3, MUL VL]
    addvl x13, x13, #4
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x17, x17, #0x4
    mov x14, x22
    sub x5, x5, x6, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_28)  // Exit
    SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme1_mla)

    KAI_ASM_END
