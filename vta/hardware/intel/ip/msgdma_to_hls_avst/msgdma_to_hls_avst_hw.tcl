# TCL File Generated by Component Editor 18.0
# Thu Nov 15 18:20:03 HKT 2018
# DO NOT MODIFY


# 
# msgdma_to_hls_avst "msgdma_to_hls_avst" v1.0
#  2018.11.15.18:20:03
# 
# 

# 
# request TCL package from ACDS 16.1
# 
package require -exact qsys 16.1


# 
# module msgdma_to_hls_avst
# 
set_module_property DESCRIPTION ""
set_module_property NAME msgdma_to_hls_avst
set_module_property VERSION 1.0
set_module_property INTERNAL false
set_module_property OPAQUE_ADDRESS_MAP true
set_module_property AUTHOR ""
set_module_property DISPLAY_NAME msgdma_to_hls_avst
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
set_module_property REPORT_TO_TALKBACK false
set_module_property ALLOW_GREYBOX_GENERATION false
set_module_property REPORT_HIERARCHY false


# 
# file sets
# 
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL msgdma_to_hls_avst
set_fileset_property QUARTUS_SYNTH ENABLE_RELATIVE_INCLUDE_PATHS false
set_fileset_property QUARTUS_SYNTH ENABLE_FILE_OVERWRITE_MODE true
add_fileset_file msgdma_to_hls_avst.v VERILOG PATH msgdma_to_hls_avst.v TOP_LEVEL_FILE


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

add_interface_port reset reset_reset reset Input 1


# 
# connection point avst_sink
# 
add_interface avst_sink avalon_streaming end
set_interface_property avst_sink associatedClock clock
set_interface_property avst_sink associatedReset reset
set_interface_property avst_sink dataBitsPerSymbol 8
set_interface_property avst_sink errorDescriptor ""
set_interface_property avst_sink firstSymbolInHighOrderBits true
set_interface_property avst_sink maxChannel 0
set_interface_property avst_sink readyLatency 0
set_interface_property avst_sink ENABLED true
set_interface_property avst_sink EXPORT_OF ""
set_interface_property avst_sink PORT_NAME_MAP ""
set_interface_property avst_sink CMSIS_SVD_VARIABLES ""
set_interface_property avst_sink SVD_ADDRESS_GROUP ""

add_interface_port avst_sink avst_sink_data data Input 128
add_interface_port avst_sink avst_sink_ready ready Output 1
add_interface_port avst_sink avst_sink_valid valid Input 1


# 
# connection point avst_source
# 
add_interface avst_source avalon_streaming start
set_interface_property avst_source associatedClock clock
set_interface_property avst_source associatedReset reset
set_interface_property avst_source dataBitsPerSymbol 8
set_interface_property avst_source errorDescriptor ""
set_interface_property avst_source firstSymbolInHighOrderBits false
set_interface_property avst_source maxChannel 0
set_interface_property avst_source readyLatency 0
set_interface_property avst_source ENABLED true
set_interface_property avst_source EXPORT_OF ""
set_interface_property avst_source PORT_NAME_MAP ""
set_interface_property avst_source CMSIS_SVD_VARIABLES ""
set_interface_property avst_source SVD_ADDRESS_GROUP ""

add_interface_port avst_source avst_source_data data Output 128
add_interface_port avst_source avst_source_ready ready Input 1
add_interface_port avst_source avst_source_valid valid Output 1

