# TCL File Generated by Component Editor 18.0
# Tue Feb 19 09:44:55 HKT 2019
# DO NOT MODIFY


# 
# compute "compute" v1.0
#  2019.02.19.09:44:55
# 
# 

# 
# request TCL package from ACDS 16.1
# 
package require -exact qsys 16.1


# 
# module compute
# 
set_module_property DESCRIPTION ""
set_module_property NAME compute
set_module_property VERSION 1.0
set_module_property INTERNAL false
set_module_property OPAQUE_ADDRESS_MAP true
set_module_property GROUP Chisel
set_module_property AUTHOR ""
set_module_property DISPLAY_NAME compute
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
set_module_property REPORT_TO_TALKBACK false
set_module_property ALLOW_GREYBOX_GENERATION false
set_module_property REPORT_HIERARCHY false


# 
# file sets
# 
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL Compute
set_fileset_property QUARTUS_SYNTH ENABLE_RELATIVE_INCLUDE_PATHS false
set_fileset_property QUARTUS_SYNTH ENABLE_FILE_OVERWRITE_MODE false
add_fileset_file compute.v VERILOG PATH compute.v TOP_LEVEL_FILE

add_fileset SIM_VERILOG SIM_VERILOG "" ""
set_fileset_property SIM_VERILOG ENABLE_RELATIVE_INCLUDE_PATHS false
set_fileset_property SIM_VERILOG ENABLE_FILE_OVERWRITE_MODE false
add_fileset_file compute.v VERILOG PATH compute.v


# 
# parameters
# 


# 
# display items
# 


# 
# connection point clock
# 
add_interface clock clock end
set_interface_property clock clockRate 0
set_interface_property clock ENABLED true
set_interface_property clock EXPORT_OF ""
set_interface_property clock PORT_NAME_MAP ""
set_interface_property clock CMSIS_SVD_VARIABLES ""
set_interface_property clock SVD_ADDRESS_GROUP ""

add_interface_port clock clock clk Input 1


# 
# connection point reset
# 
add_interface reset reset end
set_interface_property reset associatedClock clock
set_interface_property reset synchronousEdges DEASSERT
set_interface_property reset ENABLED true
set_interface_property reset EXPORT_OF ""
set_interface_property reset PORT_NAME_MAP ""
set_interface_property reset CMSIS_SVD_VARIABLES ""
set_interface_property reset SVD_ADDRESS_GROUP ""

add_interface_port reset reset reset Input 1


# 
# connection point biases
# 
add_interface biases avalon start
set_interface_property biases addressUnits SYMBOLS
set_interface_property biases associatedClock clock
set_interface_property biases associatedReset reset
set_interface_property biases bitsPerSymbol 8
set_interface_property biases burstOnBurstBoundariesOnly false
set_interface_property biases burstcountUnits WORDS
set_interface_property biases doStreamReads false
set_interface_property biases doStreamWrites false
set_interface_property biases holdTime 0
set_interface_property biases linewrapBursts false
set_interface_property biases maximumPendingReadTransactions 0
set_interface_property biases maximumPendingWriteTransactions 0
set_interface_property biases readLatency 0
set_interface_property biases readWaitTime 1
set_interface_property biases setupTime 0
set_interface_property biases timingUnits Cycles
set_interface_property biases writeWaitTime 0
set_interface_property biases ENABLED true
set_interface_property biases EXPORT_OF ""
set_interface_property biases PORT_NAME_MAP ""
set_interface_property biases CMSIS_SVD_VARIABLES ""
set_interface_property biases SVD_ADDRESS_GROUP ""

add_interface_port biases io_biases_readdata readdata Input 128
add_interface_port biases io_biases_waitrequest waitrequest Input 1
add_interface_port biases io_biases_read read Output 1
add_interface_port biases io_biases_address address Output 32


# 
# connection point done
# 
add_interface done avalon end
set_interface_property done addressUnits WORDS
set_interface_property done associatedClock clock
set_interface_property done associatedReset reset
set_interface_property done bitsPerSymbol 8
set_interface_property done burstOnBurstBoundariesOnly false
set_interface_property done burstcountUnits WORDS
set_interface_property done explicitAddressSpan 0
set_interface_property done holdTime 0
set_interface_property done linewrapBursts false
set_interface_property done maximumPendingReadTransactions 0
set_interface_property done maximumPendingWriteTransactions 0
set_interface_property done readLatency 0
set_interface_property done readWaitTime 1
set_interface_property done setupTime 0
set_interface_property done timingUnits Cycles
set_interface_property done writeWaitTime 0
set_interface_property done ENABLED true
set_interface_property done EXPORT_OF ""
set_interface_property done PORT_NAME_MAP ""
set_interface_property done CMSIS_SVD_VARIABLES ""
set_interface_property done SVD_ADDRESS_GROUP ""

add_interface_port done io_done_address address Input 1
add_interface_port done io_done_read read Input 1
add_interface_port done io_done_readdata readdata Output 8
add_interface_port done io_done_waitrequest waitrequest Output 1
add_interface_port done io_done_write write Input 1
add_interface_port done io_done_writedata writedata Input 8
set_interface_assignment done embeddedsw.configuration.isFlash 0
set_interface_assignment done embeddedsw.configuration.isMemoryDevice 0
set_interface_assignment done embeddedsw.configuration.isNonVolatileStorage 0
set_interface_assignment done embeddedsw.configuration.isPrintableDevice 0


# 
# connection point g2l_dep_queue
# 
add_interface g2l_dep_queue avalon_streaming start
set_interface_property g2l_dep_queue associatedClock clock
set_interface_property g2l_dep_queue associatedReset reset
set_interface_property g2l_dep_queue dataBitsPerSymbol 1
set_interface_property g2l_dep_queue errorDescriptor ""
set_interface_property g2l_dep_queue firstSymbolInHighOrderBits false
set_interface_property g2l_dep_queue maxChannel 0
set_interface_property g2l_dep_queue readyLatency 0
set_interface_property g2l_dep_queue ENABLED true
set_interface_property g2l_dep_queue EXPORT_OF ""
set_interface_property g2l_dep_queue PORT_NAME_MAP ""
set_interface_property g2l_dep_queue CMSIS_SVD_VARIABLES ""
set_interface_property g2l_dep_queue SVD_ADDRESS_GROUP ""

add_interface_port g2l_dep_queue io_g2l_dep_queue_data data Output 8
add_interface_port g2l_dep_queue io_g2l_dep_queue_ready ready Input 1
add_interface_port g2l_dep_queue io_g2l_dep_queue_valid valid Output 1


# 
# connection point g2s_dep_queue
# 
add_interface g2s_dep_queue avalon_streaming start
set_interface_property g2s_dep_queue associatedClock clock
set_interface_property g2s_dep_queue associatedReset reset
set_interface_property g2s_dep_queue dataBitsPerSymbol 1
set_interface_property g2s_dep_queue errorDescriptor ""
set_interface_property g2s_dep_queue firstSymbolInHighOrderBits false
set_interface_property g2s_dep_queue maxChannel 0
set_interface_property g2s_dep_queue readyLatency 0
set_interface_property g2s_dep_queue ENABLED true
set_interface_property g2s_dep_queue EXPORT_OF ""
set_interface_property g2s_dep_queue PORT_NAME_MAP ""
set_interface_property g2s_dep_queue CMSIS_SVD_VARIABLES ""
set_interface_property g2s_dep_queue SVD_ADDRESS_GROUP ""

add_interface_port g2s_dep_queue io_g2s_dep_queue_data data Output 8
add_interface_port g2s_dep_queue io_g2s_dep_queue_ready ready Input 1
add_interface_port g2s_dep_queue io_g2s_dep_queue_valid valid Output 1


# 
# connection point gemm_queue
# 
add_interface gemm_queue avalon_streaming end
set_interface_property gemm_queue associatedClock clock
set_interface_property gemm_queue associatedReset reset
set_interface_property gemm_queue dataBitsPerSymbol 128
set_interface_property gemm_queue errorDescriptor ""
set_interface_property gemm_queue firstSymbolInHighOrderBits false
set_interface_property gemm_queue maxChannel 0
set_interface_property gemm_queue readyLatency 0
set_interface_property gemm_queue ENABLED true
set_interface_property gemm_queue EXPORT_OF ""
set_interface_property gemm_queue PORT_NAME_MAP ""
set_interface_property gemm_queue CMSIS_SVD_VARIABLES ""
set_interface_property gemm_queue SVD_ADDRESS_GROUP ""

add_interface_port gemm_queue io_gemm_queue_data data Input 128
add_interface_port gemm_queue io_gemm_queue_ready ready Output 1
add_interface_port gemm_queue io_gemm_queue_valid valid Input 1


# 
# connection point inp_mem
# 
add_interface inp_mem avalon start
set_interface_property inp_mem addressUnits SYMBOLS
set_interface_property inp_mem associatedClock clock
set_interface_property inp_mem associatedReset reset
set_interface_property inp_mem bitsPerSymbol 8
set_interface_property inp_mem burstOnBurstBoundariesOnly false
set_interface_property inp_mem burstcountUnits WORDS
set_interface_property inp_mem doStreamReads false
set_interface_property inp_mem doStreamWrites false
set_interface_property inp_mem holdTime 0
set_interface_property inp_mem linewrapBursts false
set_interface_property inp_mem maximumPendingReadTransactions 0
set_interface_property inp_mem maximumPendingWriteTransactions 0
set_interface_property inp_mem readLatency 0
set_interface_property inp_mem readWaitTime 1
set_interface_property inp_mem setupTime 0
set_interface_property inp_mem timingUnits Cycles
set_interface_property inp_mem writeWaitTime 0
set_interface_property inp_mem ENABLED true
set_interface_property inp_mem EXPORT_OF ""
set_interface_property inp_mem PORT_NAME_MAP ""
set_interface_property inp_mem CMSIS_SVD_VARIABLES ""
set_interface_property inp_mem SVD_ADDRESS_GROUP ""

add_interface_port inp_mem io_inp_mem_address address Output 15
add_interface_port inp_mem io_inp_mem_read read Output 1
add_interface_port inp_mem io_inp_mem_readdata readdata Input 64
add_interface_port inp_mem io_inp_mem_write write Output 1
add_interface_port inp_mem io_inp_mem_writedata writedata Output 64
add_interface_port inp_mem io_inp_mem_waitrequest waitrequest Input 1


# 
# connection point wgt_mem
# 
add_interface wgt_mem avalon start
set_interface_property wgt_mem addressUnits SYMBOLS
set_interface_property wgt_mem associatedClock clock
set_interface_property wgt_mem associatedReset reset
set_interface_property wgt_mem bitsPerSymbol 8
set_interface_property wgt_mem burstOnBurstBoundariesOnly false
set_interface_property wgt_mem burstcountUnits WORDS
set_interface_property wgt_mem doStreamReads false
set_interface_property wgt_mem doStreamWrites false
set_interface_property wgt_mem holdTime 0
set_interface_property wgt_mem linewrapBursts false
set_interface_property wgt_mem maximumPendingReadTransactions 0
set_interface_property wgt_mem maximumPendingWriteTransactions 0
set_interface_property wgt_mem readLatency 0
set_interface_property wgt_mem readWaitTime 1
set_interface_property wgt_mem setupTime 0
set_interface_property wgt_mem timingUnits Cycles
set_interface_property wgt_mem writeWaitTime 0
set_interface_property wgt_mem ENABLED true
set_interface_property wgt_mem EXPORT_OF ""
set_interface_property wgt_mem PORT_NAME_MAP ""
set_interface_property wgt_mem CMSIS_SVD_VARIABLES ""
set_interface_property wgt_mem SVD_ADDRESS_GROUP ""

add_interface_port wgt_mem io_wgt_mem_address address Output 18
add_interface_port wgt_mem io_wgt_mem_read read Output 1
add_interface_port wgt_mem io_wgt_mem_readdata readdata Input 64
add_interface_port wgt_mem io_wgt_mem_write write Output 1
add_interface_port wgt_mem io_wgt_mem_writedata writedata Output 64
add_interface_port wgt_mem io_wgt_mem_waitrequest waitrequest Input 1


# 
# connection point uops
# 
add_interface uops avalon start
set_interface_property uops addressUnits SYMBOLS
set_interface_property uops associatedClock clock
set_interface_property uops associatedReset reset
set_interface_property uops bitsPerSymbol 8
set_interface_property uops burstOnBurstBoundariesOnly false
set_interface_property uops burstcountUnits WORDS
set_interface_property uops doStreamReads false
set_interface_property uops doStreamWrites false
set_interface_property uops holdTime 0
set_interface_property uops linewrapBursts false
set_interface_property uops maximumPendingReadTransactions 0
set_interface_property uops maximumPendingWriteTransactions 0
set_interface_property uops readLatency 0
set_interface_property uops readWaitTime 1
set_interface_property uops setupTime 0
set_interface_property uops timingUnits Cycles
set_interface_property uops writeWaitTime 0
set_interface_property uops ENABLED true
set_interface_property uops EXPORT_OF ""
set_interface_property uops PORT_NAME_MAP ""
set_interface_property uops CMSIS_SVD_VARIABLES ""
set_interface_property uops SVD_ADDRESS_GROUP ""

add_interface_port uops io_uops_readdata readdata Input 128
add_interface_port uops io_uops_waitrequest waitrequest Input 1
add_interface_port uops io_uops_read read Output 1
add_interface_port uops io_uops_address address Output 32


# 
# connection point s2g_dep_queue
# 
add_interface s2g_dep_queue avalon_streaming end
set_interface_property s2g_dep_queue associatedClock clock
set_interface_property s2g_dep_queue associatedReset reset
set_interface_property s2g_dep_queue dataBitsPerSymbol 1
set_interface_property s2g_dep_queue errorDescriptor ""
set_interface_property s2g_dep_queue firstSymbolInHighOrderBits false
set_interface_property s2g_dep_queue maxChannel 0
set_interface_property s2g_dep_queue readyLatency 0
set_interface_property s2g_dep_queue ENABLED true
set_interface_property s2g_dep_queue EXPORT_OF ""
set_interface_property s2g_dep_queue PORT_NAME_MAP ""
set_interface_property s2g_dep_queue CMSIS_SVD_VARIABLES ""
set_interface_property s2g_dep_queue SVD_ADDRESS_GROUP ""

add_interface_port s2g_dep_queue io_s2g_dep_queue_data data Input 8
add_interface_port s2g_dep_queue io_s2g_dep_queue_ready ready Output 1
add_interface_port s2g_dep_queue io_s2g_dep_queue_valid valid Input 1


# 
# connection point out_mem
# 
add_interface out_mem avalon start
set_interface_property out_mem addressUnits SYMBOLS
set_interface_property out_mem associatedClock clock
set_interface_property out_mem associatedReset reset
set_interface_property out_mem bitsPerSymbol 8
set_interface_property out_mem burstOnBurstBoundariesOnly false
set_interface_property out_mem burstcountUnits WORDS
set_interface_property out_mem doStreamReads false
set_interface_property out_mem doStreamWrites false
set_interface_property out_mem holdTime 0
set_interface_property out_mem linewrapBursts false
set_interface_property out_mem maximumPendingReadTransactions 0
set_interface_property out_mem maximumPendingWriteTransactions 0
set_interface_property out_mem readLatency 0
set_interface_property out_mem readWaitTime 1
set_interface_property out_mem setupTime 0
set_interface_property out_mem timingUnits Cycles
set_interface_property out_mem writeWaitTime 0
set_interface_property out_mem ENABLED true
set_interface_property out_mem EXPORT_OF ""
set_interface_property out_mem PORT_NAME_MAP ""
set_interface_property out_mem CMSIS_SVD_VARIABLES ""
set_interface_property out_mem SVD_ADDRESS_GROUP ""

add_interface_port out_mem io_out_mem_address address Output 17
add_interface_port out_mem io_out_mem_read read Output 1
add_interface_port out_mem io_out_mem_readdata readdata Input 128
add_interface_port out_mem io_out_mem_write write Output 1
add_interface_port out_mem io_out_mem_writedata writedata Output 128
add_interface_port out_mem io_out_mem_waitrequest waitrequest Input 1


# 
# connection point l2g_dep_queue
# 
add_interface l2g_dep_queue avalon_streaming end
set_interface_property l2g_dep_queue associatedClock clock
set_interface_property l2g_dep_queue associatedReset reset
set_interface_property l2g_dep_queue dataBitsPerSymbol 1
set_interface_property l2g_dep_queue errorDescriptor ""
set_interface_property l2g_dep_queue firstSymbolInHighOrderBits false
set_interface_property l2g_dep_queue maxChannel 0
set_interface_property l2g_dep_queue readyLatency 0
set_interface_property l2g_dep_queue ENABLED true
set_interface_property l2g_dep_queue EXPORT_OF ""
set_interface_property l2g_dep_queue PORT_NAME_MAP ""
set_interface_property l2g_dep_queue CMSIS_SVD_VARIABLES ""
set_interface_property l2g_dep_queue SVD_ADDRESS_GROUP ""

add_interface_port l2g_dep_queue io_l2g_dep_queue_data data Input 8
add_interface_port l2g_dep_queue io_l2g_dep_queue_ready ready Output 1
add_interface_port l2g_dep_queue io_l2g_dep_queue_valid valid Input 1

