# TCL File Generated by Component Editor 18.0
# Mon Dec 10 18:14:54 HKT 2018
# DO NOT MODIFY


# 
# stream_acp_adaptor "stream_acp_adaptor" v1.0
#  2018.12.10.18:14:54
# 
# 

# 
# request TCL package from ACDS 16.1
# 
package require -exact qsys 16.1


# 
# module stream_acp_adaptor
# 
set_module_property DESCRIPTION ""
set_module_property NAME stream_acp_adaptor
set_module_property VERSION 1.0
set_module_property INTERNAL false
set_module_property OPAQUE_ADDRESS_MAP true
set_module_property AUTHOR ""
set_module_property DISPLAY_NAME stream_acp_adaptor
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
set_module_property REPORT_TO_TALKBACK false
set_module_property ALLOW_GREYBOX_GENERATION false
set_module_property REPORT_HIERARCHY false


# 
# file sets
# 
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL stream_acp_adaptor
set_fileset_property QUARTUS_SYNTH ENABLE_RELATIVE_INCLUDE_PATHS false
set_fileset_property QUARTUS_SYNTH ENABLE_FILE_OVERWRITE_MODE false
add_fileset_file stream_acp_adaptor.v VERILOG PATH stream_acp_adaptor.v TOP_LEVEL_FILE

add_fileset SIM_VERILOG SIM_VERILOG "" ""
set_fileset_property SIM_VERILOG TOP_LEVEL stream_acp_adaptor
set_fileset_property SIM_VERILOG ENABLE_RELATIVE_INCLUDE_PATHS false
set_fileset_property SIM_VERILOG ENABLE_FILE_OVERWRITE_MODE true
add_fileset_file stream_acp_adaptor.v VERILOG PATH stream_acp_adaptor.v


# 
# parameters
# 


# 
# display items
# 


# 
# connection point avmm_desc
# 
add_interface avmm_desc avalon start
set_interface_property avmm_desc addressUnits SYMBOLS
set_interface_property avmm_desc associatedClock clock
set_interface_property avmm_desc associatedReset reset
set_interface_property avmm_desc bitsPerSymbol 8
set_interface_property avmm_desc burstOnBurstBoundariesOnly false
set_interface_property avmm_desc burstcountUnits WORDS
set_interface_property avmm_desc doStreamReads false
set_interface_property avmm_desc doStreamWrites false
set_interface_property avmm_desc holdTime 0
set_interface_property avmm_desc linewrapBursts false
set_interface_property avmm_desc maximumPendingReadTransactions 0
set_interface_property avmm_desc maximumPendingWriteTransactions 0
set_interface_property avmm_desc readLatency 0
set_interface_property avmm_desc readWaitTime 1
set_interface_property avmm_desc setupTime 0
set_interface_property avmm_desc timingUnits Cycles
set_interface_property avmm_desc writeWaitTime 0
set_interface_property avmm_desc ENABLED true
set_interface_property avmm_desc EXPORT_OF ""
set_interface_property avmm_desc PORT_NAME_MAP ""
set_interface_property avmm_desc CMSIS_SVD_VARIABLES ""
set_interface_property avmm_desc SVD_ADDRESS_GROUP ""

add_interface_port avmm_desc avmm_desc_address address Output 8
add_interface_port avmm_desc avmm_desc_waitrequest waitrequest Input 1
add_interface_port avmm_desc avmm_desc_write write Output 1
add_interface_port avmm_desc avmm_desc_writedata writedata Output 128
add_interface_port avmm_desc avmm_desc_read read Output 1
add_interface_port avmm_desc avmm_desc_readdata readdata Input 128


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

add_interface_port clock clock_clk clk Input 1


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
# connection point axi_slave
# 
add_interface axi_slave avalon end
set_interface_property axi_slave addressUnits WORDS
set_interface_property axi_slave associatedClock clock
set_interface_property axi_slave associatedReset reset
set_interface_property axi_slave bitsPerSymbol 8
set_interface_property axi_slave burstOnBurstBoundariesOnly false
set_interface_property axi_slave burstcountUnits WORDS
set_interface_property axi_slave explicitAddressSpan 0
set_interface_property axi_slave holdTime 0
set_interface_property axi_slave linewrapBursts false
set_interface_property axi_slave maximumPendingReadTransactions 0
set_interface_property axi_slave maximumPendingWriteTransactions 0
set_interface_property axi_slave readLatency 0
set_interface_property axi_slave readWaitTime 1
set_interface_property axi_slave setupTime 0
set_interface_property axi_slave timingUnits Cycles
set_interface_property axi_slave writeWaitTime 0
set_interface_property axi_slave ENABLED true
set_interface_property axi_slave EXPORT_OF ""
set_interface_property axi_slave PORT_NAME_MAP ""
set_interface_property axi_slave CMSIS_SVD_VARIABLES ""
set_interface_property axi_slave SVD_ADDRESS_GROUP ""

add_interface_port axi_slave axi_slave_address address Input 1
add_interface_port axi_slave axi_slave_read read Input 1
add_interface_port axi_slave axi_slave_readdata readdata Output 32
add_interface_port axi_slave axi_slave_write write Input 1
add_interface_port axi_slave axi_slave_waitrequest waitrequest Output 1
add_interface_port axi_slave axi_slave_writedata writedata Input 32
set_interface_assignment axi_slave embeddedsw.configuration.isFlash 0
set_interface_assignment axi_slave embeddedsw.configuration.isMemoryDevice 0
set_interface_assignment axi_slave embeddedsw.configuration.isNonVolatileStorage 0
set_interface_assignment axi_slave embeddedsw.configuration.isPrintableDevice 0


# 
# connection point cfg_master
# 
add_interface cfg_master avalon start
set_interface_property cfg_master addressUnits SYMBOLS
set_interface_property cfg_master associatedClock clock
set_interface_property cfg_master associatedReset reset
set_interface_property cfg_master bitsPerSymbol 8
set_interface_property cfg_master burstOnBurstBoundariesOnly false
set_interface_property cfg_master burstcountUnits WORDS
set_interface_property cfg_master doStreamReads false
set_interface_property cfg_master doStreamWrites false
set_interface_property cfg_master holdTime 0
set_interface_property cfg_master linewrapBursts false
set_interface_property cfg_master maximumPendingReadTransactions 0
set_interface_property cfg_master maximumPendingWriteTransactions 0
set_interface_property cfg_master readLatency 0
set_interface_property cfg_master readWaitTime 1
set_interface_property cfg_master setupTime 0
set_interface_property cfg_master timingUnits Cycles
set_interface_property cfg_master writeWaitTime 0
set_interface_property cfg_master ENABLED true
set_interface_property cfg_master EXPORT_OF ""
set_interface_property cfg_master PORT_NAME_MAP ""
set_interface_property cfg_master CMSIS_SVD_VARIABLES ""
set_interface_property cfg_master SVD_ADDRESS_GROUP ""

add_interface_port cfg_master cfg_master_address address Output 8
add_interface_port cfg_master cfg_master_read read Output 1
add_interface_port cfg_master cfg_master_readdata readdata Input 32
add_interface_port cfg_master cfg_master_write write Output 1
add_interface_port cfg_master cfg_master_writedata writedata Output 32
add_interface_port cfg_master cfg_master_waitrequest waitrequest Input 1


# 
# connection point avmm_csr
# 
add_interface avmm_csr avalon start
set_interface_property avmm_csr addressUnits SYMBOLS
set_interface_property avmm_csr associatedClock clock
set_interface_property avmm_csr associatedReset reset
set_interface_property avmm_csr bitsPerSymbol 8
set_interface_property avmm_csr burstOnBurstBoundariesOnly false
set_interface_property avmm_csr burstcountUnits WORDS
set_interface_property avmm_csr doStreamReads false
set_interface_property avmm_csr doStreamWrites false
set_interface_property avmm_csr holdTime 0
set_interface_property avmm_csr linewrapBursts false
set_interface_property avmm_csr maximumPendingReadTransactions 0
set_interface_property avmm_csr maximumPendingWriteTransactions 0
set_interface_property avmm_csr readLatency 0
set_interface_property avmm_csr readWaitTime 1
set_interface_property avmm_csr setupTime 0
set_interface_property avmm_csr timingUnits Cycles
set_interface_property avmm_csr writeWaitTime 0
set_interface_property avmm_csr ENABLED true
set_interface_property avmm_csr EXPORT_OF ""
set_interface_property avmm_csr PORT_NAME_MAP ""
set_interface_property avmm_csr CMSIS_SVD_VARIABLES ""
set_interface_property avmm_csr SVD_ADDRESS_GROUP ""

add_interface_port avmm_csr avmm_csr_address address Output 8
add_interface_port avmm_csr avmm_csr_waitrequest waitrequest Input 1
add_interface_port avmm_csr avmm_csr_read read Output 1
add_interface_port avmm_csr avmm_csr_readdata readdata Input 32
add_interface_port avmm_csr avmm_csr_write write Output 1
add_interface_port avmm_csr avmm_csr_writedata writedata Output 32


# 
# connection point data_length
# 
add_interface data_length conduit end
set_interface_property data_length associatedClock clock
set_interface_property data_length associatedReset reset
set_interface_property data_length ENABLED true
set_interface_property data_length EXPORT_OF ""
set_interface_property data_length PORT_NAME_MAP ""
set_interface_property data_length CMSIS_SVD_VARIABLES ""
set_interface_property data_length SVD_ADDRESS_GROUP ""

add_interface_port data_length data_length_data data Input 32

