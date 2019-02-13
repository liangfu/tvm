module Compute(
  input          clock,
  input          reset,
  output         io_done_waitrequest,
  input          io_done_address,
  input          io_done_read,
  output         io_done_readdata,
  input          io_done_write,
  input          io_done_writedata,
  input          io_uops_waitrequest,
  output [31:0]  io_uops_address,
  output         io_uops_read,
  input  [31:0]  io_uops_readdata,
  output         io_uops_write,
  output [31:0]  io_uops_writedata,
  input          io_biases_waitrequest,
  output [31:0]  io_biases_address,
  output         io_biases_read,
  input  [127:0] io_biases_readdata,
  output         io_biases_write,
  output [127:0] io_biases_writedata,
  output         io_gemm_queue_ready,
  input          io_gemm_queue_valid,
  input  [127:0] io_gemm_queue_data,
  output         io_l2g_dep_queue_ready,
  input          io_l2g_dep_queue_valid,
  input          io_l2g_dep_queue_data,
  output         io_s2g_dep_queue_ready,
  input          io_s2g_dep_queue_valid,
  input          io_s2g_dep_queue_data,
  input          io_g2l_dep_queue_ready,
  output         io_g2l_dep_queue_valid,
  output         io_g2l_dep_queue_data,
  input          io_g2s_dep_queue_ready,
  output         io_g2s_dep_queue_valid,
  output         io_g2s_dep_queue_data,
  input          io_inp_mem_waitrequest,
  output [14:0]  io_inp_mem_address,
  output         io_inp_mem_read,
  input  [63:0]  io_inp_mem_readdata,
  output         io_inp_mem_write,
  output [63:0]  io_inp_mem_writedata,
  input          io_wgt_mem_waitrequest,
  output [17:0]  io_wgt_mem_address,
  output         io_wgt_mem_read,
  input  [63:0]  io_wgt_mem_readdata,
  output         io_wgt_mem_write,
  output [63:0]  io_wgt_mem_writedata,
  input          io_out_mem_waitrequest,
  output [16:0]  io_out_mem_address,
  output         io_out_mem_read,
  input  [127:0] io_out_mem_readdata,
  output         io_out_mem_write,
  output [127:0] io_out_mem_writedata
);
  reg [511:0] acc_mem [0:255] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 33:20]
  reg [511:0] _RAND_0;
  wire [511:0] acc_mem__T_416_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_416_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_418_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_418_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_394_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_394_addr; // @[Compute.scala 33:20]
  wire  acc_mem__T_394_mask; // @[Compute.scala 33:20]
  wire  acc_mem__T_394_en; // @[Compute.scala 33:20]
  reg [31:0] uop_mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 34:20]
  reg [31:0] _RAND_1;
  wire [31:0] uop_mem_uop_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem_uop_addr; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_352_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_352_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_352_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_352_en; // @[Compute.scala 34:20]
  wire  started; // @[Compute.scala 31:17]
  reg [127:0] insn; // @[Compute.scala 36:28]
  reg [127:0] _RAND_2;
  wire  _T_201; // @[Compute.scala 37:31]
  wire  insn_valid; // @[Compute.scala 37:40]
  wire [2:0] opcode; // @[Compute.scala 39:29]
  wire  pop_prev_dep; // @[Compute.scala 40:29]
  wire  pop_next_dep; // @[Compute.scala 41:29]
  wire  push_prev_dep; // @[Compute.scala 42:29]
  wire  push_next_dep; // @[Compute.scala 43:29]
  wire [1:0] memory_type; // @[Compute.scala 45:25]
  wire [15:0] sram_base; // @[Compute.scala 46:25]
  wire [31:0] dram_base; // @[Compute.scala 47:25]
  wire [15:0] uop_cntr_max; // @[Compute.scala 49:25]
  wire [3:0] y_pad_0; // @[Compute.scala 51:25]
  wire [3:0] x_pad_0; // @[Compute.scala 53:25]
  wire [3:0] x_pad_1; // @[Compute.scala 54:25]
  wire [15:0] _GEN_290; // @[Compute.scala 58:30]
  wire [15:0] _GEN_292; // @[Compute.scala 59:30]
  wire [16:0] _T_205; // @[Compute.scala 59:30]
  wire [15:0] _T_206; // @[Compute.scala 59:30]
  wire [15:0] _GEN_293; // @[Compute.scala 59:39]
  wire [16:0] _T_207; // @[Compute.scala 59:39]
  wire [15:0] x_size_total; // @[Compute.scala 59:39]
  wire [19:0] y_offset; // @[Compute.scala 60:31]
  wire  _T_210; // @[Compute.scala 64:32]
  wire  _T_212; // @[Compute.scala 64:60]
  wire  opcode_load_en; // @[Compute.scala 64:50]
  wire  opcode_gemm_en; // @[Compute.scala 65:32]
  wire  opcode_alu_en; // @[Compute.scala 66:31]
  wire  memory_type_uop_en; // @[Compute.scala 68:40]
  wire  memory_type_acc_en; // @[Compute.scala 69:40]
  reg [2:0] state; // @[Compute.scala 72:22]
  reg [31:0] _RAND_3;
  wire  idle; // @[Compute.scala 74:20]
  wire  dump; // @[Compute.scala 75:20]
  wire  busy; // @[Compute.scala 76:20]
  wire  push; // @[Compute.scala 77:20]
  wire  done; // @[Compute.scala 78:20]
  reg  uops_read; // @[Compute.scala 81:24]
  reg [31:0] _RAND_4;
  reg [31:0] uops_data; // @[Compute.scala 82:24]
  reg [31:0] _RAND_5;
  reg  biases_read; // @[Compute.scala 84:24]
  reg [31:0] _RAND_6;
  reg [127:0] biases_data_0; // @[Compute.scala 87:24]
  reg [127:0] _RAND_7;
  reg [127:0] biases_data_1; // @[Compute.scala 87:24]
  reg [127:0] _RAND_8;
  reg [127:0] biases_data_2; // @[Compute.scala 87:24]
  reg [127:0] _RAND_9;
  reg [127:0] biases_data_3; // @[Compute.scala 87:24]
  reg [127:0] _RAND_10;
  reg  out_mem_write; // @[Compute.scala 89:31]
  reg [31:0] _RAND_11;
  wire  _T_234; // @[Compute.scala 93:37]
  wire  uop_cntr_en; // @[Compute.scala 93:59]
  reg [15:0] uop_cntr_val; // @[Compute.scala 95:25]
  reg [31:0] _RAND_12;
  wire  _T_236; // @[Compute.scala 96:38]
  wire  _T_237; // @[Compute.scala 96:56]
  wire  uop_cntr_wrap; // @[Compute.scala 96:71]
  wire [18:0] _T_239; // @[Compute.scala 98:29]
  wire [19:0] _T_241; // @[Compute.scala 98:46]
  wire [18:0] acc_cntr_max; // @[Compute.scala 98:46]
  wire  _T_242; // @[Compute.scala 99:37]
  wire  acc_cntr_en; // @[Compute.scala 99:59]
  reg [15:0] acc_cntr_val; // @[Compute.scala 101:25]
  reg [31:0] _RAND_13;
  wire [18:0] _GEN_295; // @[Compute.scala 102:38]
  wire  _T_244; // @[Compute.scala 102:38]
  wire  _T_245; // @[Compute.scala 102:56]
  wire  acc_cntr_wrap; // @[Compute.scala 102:71]
  wire  _T_249; // @[Compute.scala 105:37]
  wire  out_cntr_en; // @[Compute.scala 105:56]
  reg [15:0] dst_offset_in; // @[Compute.scala 107:25]
  reg [31:0] _RAND_14;
  wire  _T_251; // @[Compute.scala 108:38]
  wire  _T_252; // @[Compute.scala 108:56]
  wire  out_cntr_wrap; // @[Compute.scala 108:71]
  reg  pop_prev_dep_ready; // @[Compute.scala 111:35]
  reg [31:0] _RAND_15;
  reg  pop_next_dep_ready; // @[Compute.scala 112:35]
  reg [31:0] _RAND_16;
  wire  push_prev_dep_valid; // @[Compute.scala 113:43]
  wire  push_next_dep_valid; // @[Compute.scala 114:43]
  reg  push_prev_dep_ready; // @[Compute.scala 115:36]
  reg [31:0] _RAND_17;
  reg  push_next_dep_ready; // @[Compute.scala 116:36]
  reg [31:0] _RAND_18;
  reg  gemm_queue_ready; // @[Compute.scala 118:33]
  reg [31:0] _RAND_19;
  wire  _T_263; // @[Compute.scala 121:23]
  wire  _T_264; // @[Compute.scala 121:40]
  wire  _T_265; // @[Compute.scala 122:25]
  wire [2:0] _GEN_0; // @[Compute.scala 122:43]
  wire [2:0] _GEN_1; // @[Compute.scala 121:58]
  wire  _T_267; // @[Compute.scala 130:18]
  wire  _T_269; // @[Compute.scala 130:41]
  wire  _T_270; // @[Compute.scala 130:38]
  wire  _T_271; // @[Compute.scala 130:14]
  wire  _T_272; // @[Compute.scala 130:79]
  wire  _T_273; // @[Compute.scala 130:62]
  wire [2:0] _GEN_2; // @[Compute.scala 130:97]
  wire  _T_274; // @[Compute.scala 131:38]
  wire  _T_275; // @[Compute.scala 131:14]
  wire [2:0] _GEN_3; // @[Compute.scala 131:63]
  wire  _T_276; // @[Compute.scala 132:38]
  wire  _T_277; // @[Compute.scala 132:14]
  wire [2:0] _GEN_4; // @[Compute.scala 132:63]
  wire  _T_280; // @[Compute.scala 139:22]
  wire  _T_281; // @[Compute.scala 139:30]
  wire  _GEN_5; // @[Compute.scala 139:57]
  wire  _T_283; // @[Compute.scala 142:22]
  wire  _T_284; // @[Compute.scala 142:30]
  wire  _GEN_6; // @[Compute.scala 142:57]
  wire  _T_288; // @[Compute.scala 149:29]
  wire  _T_289; // @[Compute.scala 149:55]
  wire  _GEN_7; // @[Compute.scala 149:64]
  wire  _T_291; // @[Compute.scala 152:29]
  wire  _T_292; // @[Compute.scala 152:55]
  wire  _GEN_8; // @[Compute.scala 152:64]
  wire  _T_295; // @[Compute.scala 157:22]
  wire  _T_296; // @[Compute.scala 157:19]
  wire  _T_297; // @[Compute.scala 157:37]
  wire  _T_298; // @[Compute.scala 157:61]
  wire  _T_299; // @[Compute.scala 157:45]
  wire [16:0] _T_301; // @[Compute.scala 158:34]
  wire [15:0] _T_302; // @[Compute.scala 158:34]
  wire [15:0] _GEN_9; // @[Compute.scala 157:77]
  wire  _T_304; // @[Compute.scala 160:24]
  wire  _T_305; // @[Compute.scala 160:21]
  wire  _T_306; // @[Compute.scala 160:39]
  wire  _T_307; // @[Compute.scala 160:63]
  wire  _T_308; // @[Compute.scala 160:47]
  wire [16:0] _T_310; // @[Compute.scala 161:34]
  wire [15:0] _T_311; // @[Compute.scala 161:34]
  wire [15:0] _GEN_10; // @[Compute.scala 160:79]
  wire  _T_313; // @[Compute.scala 163:26]
  wire  _T_314; // @[Compute.scala 163:23]
  wire  _T_315; // @[Compute.scala 163:41]
  wire  _T_316; // @[Compute.scala 163:65]
  wire  _T_317; // @[Compute.scala 163:49]
  wire [16:0] _T_319; // @[Compute.scala 164:34]
  wire [15:0] _T_320; // @[Compute.scala 164:34]
  wire [15:0] _GEN_11; // @[Compute.scala 163:81]
  wire  _GEN_16; // @[Compute.scala 168:27]
  wire  _GEN_17; // @[Compute.scala 168:27]
  wire  _GEN_18; // @[Compute.scala 168:27]
  wire  _GEN_19; // @[Compute.scala 168:27]
  wire [2:0] _GEN_20; // @[Compute.scala 168:27]
  wire  _T_328; // @[Compute.scala 181:52]
  wire  _T_329; // @[Compute.scala 181:43]
  wire  _GEN_21; // @[Compute.scala 183:27]
  wire [31:0] _GEN_297; // @[Compute.scala 193:33]
  wire [32:0] _T_334; // @[Compute.scala 193:33]
  wire [31:0] _T_335; // @[Compute.scala 193:33]
  wire [34:0] _GEN_298; // @[Compute.scala 193:49]
  wire [34:0] uop_dram_addr; // @[Compute.scala 193:49]
  wire [16:0] _T_337; // @[Compute.scala 194:33]
  wire [15:0] uop_sram_addr; // @[Compute.scala 194:33]
  wire  _T_339; // @[Compute.scala 195:31]
  wire  _T_340; // @[Compute.scala 195:28]
  wire  _T_341; // @[Compute.scala 195:46]
  wire [16:0] _T_346; // @[Compute.scala 202:42]
  wire [16:0] _T_347; // @[Compute.scala 202:42]
  wire [15:0] _T_348; // @[Compute.scala 202:42]
  wire  _T_349; // @[Compute.scala 202:24]
  wire  _GEN_22; // @[Compute.scala 202:50]
  wire [31:0] _GEN_299; // @[Compute.scala 207:36]
  wire [32:0] _T_353; // @[Compute.scala 207:36]
  wire [31:0] _T_354; // @[Compute.scala 207:36]
  wire [31:0] _GEN_300; // @[Compute.scala 207:47]
  wire [32:0] _T_355; // @[Compute.scala 207:47]
  wire [31:0] _T_356; // @[Compute.scala 207:47]
  wire [34:0] _GEN_301; // @[Compute.scala 207:58]
  wire [34:0] _T_358; // @[Compute.scala 207:58]
  wire [35:0] _T_360; // @[Compute.scala 207:66]
  wire [35:0] _GEN_302; // @[Compute.scala 207:76]
  wire [36:0] _T_361; // @[Compute.scala 207:76]
  wire [35:0] _T_362; // @[Compute.scala 207:76]
  wire [42:0] _GEN_303; // @[Compute.scala 207:92]
  wire [42:0] acc_dram_addr; // @[Compute.scala 207:92]
  wire [19:0] _GEN_304; // @[Compute.scala 208:36]
  wire [20:0] _T_364; // @[Compute.scala 208:36]
  wire [19:0] _T_365; // @[Compute.scala 208:36]
  wire [19:0] _GEN_305; // @[Compute.scala 208:47]
  wire [20:0] _T_366; // @[Compute.scala 208:47]
  wire [19:0] _T_367; // @[Compute.scala 208:47]
  wire [22:0] _GEN_306; // @[Compute.scala 208:58]
  wire [22:0] _T_369; // @[Compute.scala 208:58]
  wire [23:0] _T_371; // @[Compute.scala 208:66]
  wire [23:0] _GEN_307; // @[Compute.scala 208:76]
  wire [24:0] _T_372; // @[Compute.scala 208:76]
  wire [23:0] _T_373; // @[Compute.scala 208:76]
  wire [23:0] _T_375; // @[Compute.scala 208:92]
  wire [24:0] _T_377; // @[Compute.scala 208:100]
  wire [24:0] _T_378; // @[Compute.scala 208:100]
  wire [23:0] acc_sram_addr; // @[Compute.scala 208:100]
  wire  _T_380; // @[Compute.scala 209:33]
  wire [15:0] _GEN_12; // @[Compute.scala 215:30]
  wire [2:0] _T_386; // @[Compute.scala 215:30]
  wire [127:0] _GEN_25; // @[Compute.scala 215:48]
  wire [127:0] _GEN_26; // @[Compute.scala 215:48]
  wire [127:0] _GEN_27; // @[Compute.scala 215:48]
  wire [127:0] _GEN_28; // @[Compute.scala 215:48]
  wire  _T_392; // @[Compute.scala 219:43]
  wire [255:0] _T_395; // @[Cat.scala 30:58]
  wire [255:0] _T_396; // @[Cat.scala 30:58]
  wire [1:0] alu_opcode; // @[Compute.scala 229:24]
  wire  use_imm; // @[Compute.scala 230:21]
  wire [15:0] imm_raw; // @[Compute.scala 231:21]
  wire [15:0] _T_398; // @[Compute.scala 232:25]
  wire  _T_400; // @[Compute.scala 232:32]
  wire [31:0] _T_402; // @[Cat.scala 30:58]
  wire [16:0] _T_404; // @[Cat.scala 30:58]
  wire [31:0] _T_405; // @[Compute.scala 232:16]
  wire [31:0] imm; // @[Compute.scala 232:89]
  wire [10:0] _T_406; // @[Compute.scala 240:20]
  wire [15:0] _GEN_308; // @[Compute.scala 240:47]
  wire [16:0] _T_407; // @[Compute.scala 240:47]
  wire [15:0] dst_idx; // @[Compute.scala 240:47]
  wire [10:0] _T_408; // @[Compute.scala 241:20]
  wire [15:0] _GEN_309; // @[Compute.scala 241:47]
  wire [16:0] _T_409; // @[Compute.scala 241:47]
  wire [15:0] src_idx; // @[Compute.scala 241:47]
  reg [511:0] dst_vector; // @[Compute.scala 244:23]
  reg [511:0] _RAND_20;
  reg [511:0] src_vector; // @[Compute.scala 245:23]
  reg [511:0] _RAND_21;
  wire  alu_opcode_min_en; // @[Compute.scala 263:38]
  wire  alu_opcode_max_en; // @[Compute.scala 264:38]
  wire  _T_850; // @[Compute.scala 283:20]
  wire [31:0] _T_851; // @[Compute.scala 286:31]
  wire [31:0] _T_852; // @[Compute.scala 286:72]
  wire [31:0] _T_853; // @[Compute.scala 287:31]
  wire [31:0] _T_854; // @[Compute.scala 287:72]
  wire [31:0] _T_855; // @[Compute.scala 286:31]
  wire [31:0] _T_856; // @[Compute.scala 286:72]
  wire [31:0] _T_857; // @[Compute.scala 287:31]
  wire [31:0] _T_858; // @[Compute.scala 287:72]
  wire [31:0] _T_859; // @[Compute.scala 286:31]
  wire [31:0] _T_860; // @[Compute.scala 286:72]
  wire [31:0] _T_861; // @[Compute.scala 287:31]
  wire [31:0] _T_862; // @[Compute.scala 287:72]
  wire [31:0] _T_863; // @[Compute.scala 286:31]
  wire [31:0] _T_864; // @[Compute.scala 286:72]
  wire [31:0] _T_865; // @[Compute.scala 287:31]
  wire [31:0] _T_866; // @[Compute.scala 287:72]
  wire [31:0] _T_867; // @[Compute.scala 286:31]
  wire [31:0] _T_868; // @[Compute.scala 286:72]
  wire [31:0] _T_869; // @[Compute.scala 287:31]
  wire [31:0] _T_870; // @[Compute.scala 287:72]
  wire [31:0] _T_871; // @[Compute.scala 286:31]
  wire [31:0] _T_872; // @[Compute.scala 286:72]
  wire [31:0] _T_873; // @[Compute.scala 287:31]
  wire [31:0] _T_874; // @[Compute.scala 287:72]
  wire [31:0] _T_875; // @[Compute.scala 286:31]
  wire [31:0] _T_876; // @[Compute.scala 286:72]
  wire [31:0] _T_877; // @[Compute.scala 287:31]
  wire [31:0] _T_878; // @[Compute.scala 287:72]
  wire [31:0] _T_879; // @[Compute.scala 286:31]
  wire [31:0] _T_880; // @[Compute.scala 286:72]
  wire [31:0] _T_881; // @[Compute.scala 287:31]
  wire [31:0] _T_882; // @[Compute.scala 287:72]
  wire [31:0] _T_883; // @[Compute.scala 286:31]
  wire [31:0] _T_884; // @[Compute.scala 286:72]
  wire [31:0] _T_885; // @[Compute.scala 287:31]
  wire [31:0] _T_886; // @[Compute.scala 287:72]
  wire [31:0] _T_887; // @[Compute.scala 286:31]
  wire [31:0] _T_888; // @[Compute.scala 286:72]
  wire [31:0] _T_889; // @[Compute.scala 287:31]
  wire [31:0] _T_890; // @[Compute.scala 287:72]
  wire [31:0] _T_891; // @[Compute.scala 286:31]
  wire [31:0] _T_892; // @[Compute.scala 286:72]
  wire [31:0] _T_893; // @[Compute.scala 287:31]
  wire [31:0] _T_894; // @[Compute.scala 287:72]
  wire [31:0] _T_895; // @[Compute.scala 286:31]
  wire [31:0] _T_896; // @[Compute.scala 286:72]
  wire [31:0] _T_897; // @[Compute.scala 287:31]
  wire [31:0] _T_898; // @[Compute.scala 287:72]
  wire [31:0] _T_899; // @[Compute.scala 286:31]
  wire [31:0] _T_900; // @[Compute.scala 286:72]
  wire [31:0] _T_901; // @[Compute.scala 287:31]
  wire [31:0] _T_902; // @[Compute.scala 287:72]
  wire [31:0] _T_903; // @[Compute.scala 286:31]
  wire [31:0] _T_904; // @[Compute.scala 286:72]
  wire [31:0] _T_905; // @[Compute.scala 287:31]
  wire [31:0] _T_906; // @[Compute.scala 287:72]
  wire [31:0] _T_907; // @[Compute.scala 286:31]
  wire [31:0] _T_908; // @[Compute.scala 286:72]
  wire [31:0] _T_909; // @[Compute.scala 287:31]
  wire [31:0] _T_910; // @[Compute.scala 287:72]
  wire [31:0] _T_911; // @[Compute.scala 286:31]
  wire [31:0] _T_912; // @[Compute.scala 286:72]
  wire [31:0] _T_913; // @[Compute.scala 287:31]
  wire [31:0] _T_914; // @[Compute.scala 287:72]
  wire [31:0] _GEN_51; // @[Compute.scala 284:30]
  wire [31:0] _GEN_52; // @[Compute.scala 284:30]
  wire [31:0] _GEN_53; // @[Compute.scala 284:30]
  wire [31:0] _GEN_54; // @[Compute.scala 284:30]
  wire [31:0] _GEN_55; // @[Compute.scala 284:30]
  wire [31:0] _GEN_56; // @[Compute.scala 284:30]
  wire [31:0] _GEN_57; // @[Compute.scala 284:30]
  wire [31:0] _GEN_58; // @[Compute.scala 284:30]
  wire [31:0] _GEN_59; // @[Compute.scala 284:30]
  wire [31:0] _GEN_60; // @[Compute.scala 284:30]
  wire [31:0] _GEN_61; // @[Compute.scala 284:30]
  wire [31:0] _GEN_62; // @[Compute.scala 284:30]
  wire [31:0] _GEN_63; // @[Compute.scala 284:30]
  wire [31:0] _GEN_64; // @[Compute.scala 284:30]
  wire [31:0] _GEN_65; // @[Compute.scala 284:30]
  wire [31:0] _GEN_66; // @[Compute.scala 284:30]
  wire [31:0] _GEN_67; // @[Compute.scala 284:30]
  wire [31:0] _GEN_68; // @[Compute.scala 284:30]
  wire [31:0] _GEN_69; // @[Compute.scala 284:30]
  wire [31:0] _GEN_70; // @[Compute.scala 284:30]
  wire [31:0] _GEN_71; // @[Compute.scala 284:30]
  wire [31:0] _GEN_72; // @[Compute.scala 284:30]
  wire [31:0] _GEN_73; // @[Compute.scala 284:30]
  wire [31:0] _GEN_74; // @[Compute.scala 284:30]
  wire [31:0] _GEN_75; // @[Compute.scala 284:30]
  wire [31:0] _GEN_76; // @[Compute.scala 284:30]
  wire [31:0] _GEN_77; // @[Compute.scala 284:30]
  wire [31:0] _GEN_78; // @[Compute.scala 284:30]
  wire [31:0] _GEN_79; // @[Compute.scala 284:30]
  wire [31:0] _GEN_80; // @[Compute.scala 284:30]
  wire [31:0] _GEN_81; // @[Compute.scala 284:30]
  wire [31:0] _GEN_82; // @[Compute.scala 284:30]
  wire [31:0] _GEN_83; // @[Compute.scala 295:20]
  wire [31:0] _GEN_84; // @[Compute.scala 295:20]
  wire [31:0] _GEN_85; // @[Compute.scala 295:20]
  wire [31:0] _GEN_86; // @[Compute.scala 295:20]
  wire [31:0] _GEN_87; // @[Compute.scala 295:20]
  wire [31:0] _GEN_88; // @[Compute.scala 295:20]
  wire [31:0] _GEN_89; // @[Compute.scala 295:20]
  wire [31:0] _GEN_90; // @[Compute.scala 295:20]
  wire [31:0] _GEN_91; // @[Compute.scala 295:20]
  wire [31:0] _GEN_92; // @[Compute.scala 295:20]
  wire [31:0] _GEN_93; // @[Compute.scala 295:20]
  wire [31:0] _GEN_94; // @[Compute.scala 295:20]
  wire [31:0] _GEN_95; // @[Compute.scala 295:20]
  wire [31:0] _GEN_96; // @[Compute.scala 295:20]
  wire [31:0] _GEN_97; // @[Compute.scala 295:20]
  wire [31:0] _GEN_98; // @[Compute.scala 295:20]
  wire [31:0] src_0_0; // @[Compute.scala 283:36]
  wire [31:0] src_1_0; // @[Compute.scala 283:36]
  wire  _T_979; // @[Compute.scala 300:34]
  wire [31:0] _T_980; // @[Compute.scala 300:24]
  wire [31:0] mix_val_0; // @[Compute.scala 283:36]
  wire [7:0] _T_981; // @[Compute.scala 302:37]
  wire [31:0] _T_982; // @[Compute.scala 303:30]
  wire [31:0] _T_983; // @[Compute.scala 303:59]
  wire [32:0] _T_984; // @[Compute.scala 303:49]
  wire [31:0] _T_985; // @[Compute.scala 303:49]
  wire [31:0] _T_986; // @[Compute.scala 303:79]
  wire [31:0] add_val_0; // @[Compute.scala 283:36]
  wire [31:0] add_res_0; // @[Compute.scala 283:36]
  wire [7:0] _T_987; // @[Compute.scala 305:37]
  wire [4:0] _T_989; // @[Compute.scala 306:60]
  wire [31:0] _T_990; // @[Compute.scala 306:49]
  wire [31:0] _T_991; // @[Compute.scala 306:84]
  wire [31:0] shr_val_0; // @[Compute.scala 283:36]
  wire [31:0] shr_res_0; // @[Compute.scala 283:36]
  wire [7:0] _T_992; // @[Compute.scala 308:37]
  wire [31:0] src_0_1; // @[Compute.scala 283:36]
  wire [31:0] src_1_1; // @[Compute.scala 283:36]
  wire  _T_993; // @[Compute.scala 300:34]
  wire [31:0] _T_994; // @[Compute.scala 300:24]
  wire [31:0] mix_val_1; // @[Compute.scala 283:36]
  wire [7:0] _T_995; // @[Compute.scala 302:37]
  wire [31:0] _T_996; // @[Compute.scala 303:30]
  wire [31:0] _T_997; // @[Compute.scala 303:59]
  wire [32:0] _T_998; // @[Compute.scala 303:49]
  wire [31:0] _T_999; // @[Compute.scala 303:49]
  wire [31:0] _T_1000; // @[Compute.scala 303:79]
  wire [31:0] add_val_1; // @[Compute.scala 283:36]
  wire [31:0] add_res_1; // @[Compute.scala 283:36]
  wire [7:0] _T_1001; // @[Compute.scala 305:37]
  wire [4:0] _T_1003; // @[Compute.scala 306:60]
  wire [31:0] _T_1004; // @[Compute.scala 306:49]
  wire [31:0] _T_1005; // @[Compute.scala 306:84]
  wire [31:0] shr_val_1; // @[Compute.scala 283:36]
  wire [31:0] shr_res_1; // @[Compute.scala 283:36]
  wire [7:0] _T_1006; // @[Compute.scala 308:37]
  wire [31:0] src_0_2; // @[Compute.scala 283:36]
  wire [31:0] src_1_2; // @[Compute.scala 283:36]
  wire  _T_1007; // @[Compute.scala 300:34]
  wire [31:0] _T_1008; // @[Compute.scala 300:24]
  wire [31:0] mix_val_2; // @[Compute.scala 283:36]
  wire [7:0] _T_1009; // @[Compute.scala 302:37]
  wire [31:0] _T_1010; // @[Compute.scala 303:30]
  wire [31:0] _T_1011; // @[Compute.scala 303:59]
  wire [32:0] _T_1012; // @[Compute.scala 303:49]
  wire [31:0] _T_1013; // @[Compute.scala 303:49]
  wire [31:0] _T_1014; // @[Compute.scala 303:79]
  wire [31:0] add_val_2; // @[Compute.scala 283:36]
  wire [31:0] add_res_2; // @[Compute.scala 283:36]
  wire [7:0] _T_1015; // @[Compute.scala 305:37]
  wire [4:0] _T_1017; // @[Compute.scala 306:60]
  wire [31:0] _T_1018; // @[Compute.scala 306:49]
  wire [31:0] _T_1019; // @[Compute.scala 306:84]
  wire [31:0] shr_val_2; // @[Compute.scala 283:36]
  wire [31:0] shr_res_2; // @[Compute.scala 283:36]
  wire [7:0] _T_1020; // @[Compute.scala 308:37]
  wire [31:0] src_0_3; // @[Compute.scala 283:36]
  wire [31:0] src_1_3; // @[Compute.scala 283:36]
  wire  _T_1021; // @[Compute.scala 300:34]
  wire [31:0] _T_1022; // @[Compute.scala 300:24]
  wire [31:0] mix_val_3; // @[Compute.scala 283:36]
  wire [7:0] _T_1023; // @[Compute.scala 302:37]
  wire [31:0] _T_1024; // @[Compute.scala 303:30]
  wire [31:0] _T_1025; // @[Compute.scala 303:59]
  wire [32:0] _T_1026; // @[Compute.scala 303:49]
  wire [31:0] _T_1027; // @[Compute.scala 303:49]
  wire [31:0] _T_1028; // @[Compute.scala 303:79]
  wire [31:0] add_val_3; // @[Compute.scala 283:36]
  wire [31:0] add_res_3; // @[Compute.scala 283:36]
  wire [7:0] _T_1029; // @[Compute.scala 305:37]
  wire [4:0] _T_1031; // @[Compute.scala 306:60]
  wire [31:0] _T_1032; // @[Compute.scala 306:49]
  wire [31:0] _T_1033; // @[Compute.scala 306:84]
  wire [31:0] shr_val_3; // @[Compute.scala 283:36]
  wire [31:0] shr_res_3; // @[Compute.scala 283:36]
  wire [7:0] _T_1034; // @[Compute.scala 308:37]
  wire [31:0] src_0_4; // @[Compute.scala 283:36]
  wire [31:0] src_1_4; // @[Compute.scala 283:36]
  wire  _T_1035; // @[Compute.scala 300:34]
  wire [31:0] _T_1036; // @[Compute.scala 300:24]
  wire [31:0] mix_val_4; // @[Compute.scala 283:36]
  wire [7:0] _T_1037; // @[Compute.scala 302:37]
  wire [31:0] _T_1038; // @[Compute.scala 303:30]
  wire [31:0] _T_1039; // @[Compute.scala 303:59]
  wire [32:0] _T_1040; // @[Compute.scala 303:49]
  wire [31:0] _T_1041; // @[Compute.scala 303:49]
  wire [31:0] _T_1042; // @[Compute.scala 303:79]
  wire [31:0] add_val_4; // @[Compute.scala 283:36]
  wire [31:0] add_res_4; // @[Compute.scala 283:36]
  wire [7:0] _T_1043; // @[Compute.scala 305:37]
  wire [4:0] _T_1045; // @[Compute.scala 306:60]
  wire [31:0] _T_1046; // @[Compute.scala 306:49]
  wire [31:0] _T_1047; // @[Compute.scala 306:84]
  wire [31:0] shr_val_4; // @[Compute.scala 283:36]
  wire [31:0] shr_res_4; // @[Compute.scala 283:36]
  wire [7:0] _T_1048; // @[Compute.scala 308:37]
  wire [31:0] src_0_5; // @[Compute.scala 283:36]
  wire [31:0] src_1_5; // @[Compute.scala 283:36]
  wire  _T_1049; // @[Compute.scala 300:34]
  wire [31:0] _T_1050; // @[Compute.scala 300:24]
  wire [31:0] mix_val_5; // @[Compute.scala 283:36]
  wire [7:0] _T_1051; // @[Compute.scala 302:37]
  wire [31:0] _T_1052; // @[Compute.scala 303:30]
  wire [31:0] _T_1053; // @[Compute.scala 303:59]
  wire [32:0] _T_1054; // @[Compute.scala 303:49]
  wire [31:0] _T_1055; // @[Compute.scala 303:49]
  wire [31:0] _T_1056; // @[Compute.scala 303:79]
  wire [31:0] add_val_5; // @[Compute.scala 283:36]
  wire [31:0] add_res_5; // @[Compute.scala 283:36]
  wire [7:0] _T_1057; // @[Compute.scala 305:37]
  wire [4:0] _T_1059; // @[Compute.scala 306:60]
  wire [31:0] _T_1060; // @[Compute.scala 306:49]
  wire [31:0] _T_1061; // @[Compute.scala 306:84]
  wire [31:0] shr_val_5; // @[Compute.scala 283:36]
  wire [31:0] shr_res_5; // @[Compute.scala 283:36]
  wire [7:0] _T_1062; // @[Compute.scala 308:37]
  wire [31:0] src_0_6; // @[Compute.scala 283:36]
  wire [31:0] src_1_6; // @[Compute.scala 283:36]
  wire  _T_1063; // @[Compute.scala 300:34]
  wire [31:0] _T_1064; // @[Compute.scala 300:24]
  wire [31:0] mix_val_6; // @[Compute.scala 283:36]
  wire [7:0] _T_1065; // @[Compute.scala 302:37]
  wire [31:0] _T_1066; // @[Compute.scala 303:30]
  wire [31:0] _T_1067; // @[Compute.scala 303:59]
  wire [32:0] _T_1068; // @[Compute.scala 303:49]
  wire [31:0] _T_1069; // @[Compute.scala 303:49]
  wire [31:0] _T_1070; // @[Compute.scala 303:79]
  wire [31:0] add_val_6; // @[Compute.scala 283:36]
  wire [31:0] add_res_6; // @[Compute.scala 283:36]
  wire [7:0] _T_1071; // @[Compute.scala 305:37]
  wire [4:0] _T_1073; // @[Compute.scala 306:60]
  wire [31:0] _T_1074; // @[Compute.scala 306:49]
  wire [31:0] _T_1075; // @[Compute.scala 306:84]
  wire [31:0] shr_val_6; // @[Compute.scala 283:36]
  wire [31:0] shr_res_6; // @[Compute.scala 283:36]
  wire [7:0] _T_1076; // @[Compute.scala 308:37]
  wire [31:0] src_0_7; // @[Compute.scala 283:36]
  wire [31:0] src_1_7; // @[Compute.scala 283:36]
  wire  _T_1077; // @[Compute.scala 300:34]
  wire [31:0] _T_1078; // @[Compute.scala 300:24]
  wire [31:0] mix_val_7; // @[Compute.scala 283:36]
  wire [7:0] _T_1079; // @[Compute.scala 302:37]
  wire [31:0] _T_1080; // @[Compute.scala 303:30]
  wire [31:0] _T_1081; // @[Compute.scala 303:59]
  wire [32:0] _T_1082; // @[Compute.scala 303:49]
  wire [31:0] _T_1083; // @[Compute.scala 303:49]
  wire [31:0] _T_1084; // @[Compute.scala 303:79]
  wire [31:0] add_val_7; // @[Compute.scala 283:36]
  wire [31:0] add_res_7; // @[Compute.scala 283:36]
  wire [7:0] _T_1085; // @[Compute.scala 305:37]
  wire [4:0] _T_1087; // @[Compute.scala 306:60]
  wire [31:0] _T_1088; // @[Compute.scala 306:49]
  wire [31:0] _T_1089; // @[Compute.scala 306:84]
  wire [31:0] shr_val_7; // @[Compute.scala 283:36]
  wire [31:0] shr_res_7; // @[Compute.scala 283:36]
  wire [7:0] _T_1090; // @[Compute.scala 308:37]
  wire [31:0] src_0_8; // @[Compute.scala 283:36]
  wire [31:0] src_1_8; // @[Compute.scala 283:36]
  wire  _T_1091; // @[Compute.scala 300:34]
  wire [31:0] _T_1092; // @[Compute.scala 300:24]
  wire [31:0] mix_val_8; // @[Compute.scala 283:36]
  wire [7:0] _T_1093; // @[Compute.scala 302:37]
  wire [31:0] _T_1094; // @[Compute.scala 303:30]
  wire [31:0] _T_1095; // @[Compute.scala 303:59]
  wire [32:0] _T_1096; // @[Compute.scala 303:49]
  wire [31:0] _T_1097; // @[Compute.scala 303:49]
  wire [31:0] _T_1098; // @[Compute.scala 303:79]
  wire [31:0] add_val_8; // @[Compute.scala 283:36]
  wire [31:0] add_res_8; // @[Compute.scala 283:36]
  wire [7:0] _T_1099; // @[Compute.scala 305:37]
  wire [4:0] _T_1101; // @[Compute.scala 306:60]
  wire [31:0] _T_1102; // @[Compute.scala 306:49]
  wire [31:0] _T_1103; // @[Compute.scala 306:84]
  wire [31:0] shr_val_8; // @[Compute.scala 283:36]
  wire [31:0] shr_res_8; // @[Compute.scala 283:36]
  wire [7:0] _T_1104; // @[Compute.scala 308:37]
  wire [31:0] src_0_9; // @[Compute.scala 283:36]
  wire [31:0] src_1_9; // @[Compute.scala 283:36]
  wire  _T_1105; // @[Compute.scala 300:34]
  wire [31:0] _T_1106; // @[Compute.scala 300:24]
  wire [31:0] mix_val_9; // @[Compute.scala 283:36]
  wire [7:0] _T_1107; // @[Compute.scala 302:37]
  wire [31:0] _T_1108; // @[Compute.scala 303:30]
  wire [31:0] _T_1109; // @[Compute.scala 303:59]
  wire [32:0] _T_1110; // @[Compute.scala 303:49]
  wire [31:0] _T_1111; // @[Compute.scala 303:49]
  wire [31:0] _T_1112; // @[Compute.scala 303:79]
  wire [31:0] add_val_9; // @[Compute.scala 283:36]
  wire [31:0] add_res_9; // @[Compute.scala 283:36]
  wire [7:0] _T_1113; // @[Compute.scala 305:37]
  wire [4:0] _T_1115; // @[Compute.scala 306:60]
  wire [31:0] _T_1116; // @[Compute.scala 306:49]
  wire [31:0] _T_1117; // @[Compute.scala 306:84]
  wire [31:0] shr_val_9; // @[Compute.scala 283:36]
  wire [31:0] shr_res_9; // @[Compute.scala 283:36]
  wire [7:0] _T_1118; // @[Compute.scala 308:37]
  wire [31:0] src_0_10; // @[Compute.scala 283:36]
  wire [31:0] src_1_10; // @[Compute.scala 283:36]
  wire  _T_1119; // @[Compute.scala 300:34]
  wire [31:0] _T_1120; // @[Compute.scala 300:24]
  wire [31:0] mix_val_10; // @[Compute.scala 283:36]
  wire [7:0] _T_1121; // @[Compute.scala 302:37]
  wire [31:0] _T_1122; // @[Compute.scala 303:30]
  wire [31:0] _T_1123; // @[Compute.scala 303:59]
  wire [32:0] _T_1124; // @[Compute.scala 303:49]
  wire [31:0] _T_1125; // @[Compute.scala 303:49]
  wire [31:0] _T_1126; // @[Compute.scala 303:79]
  wire [31:0] add_val_10; // @[Compute.scala 283:36]
  wire [31:0] add_res_10; // @[Compute.scala 283:36]
  wire [7:0] _T_1127; // @[Compute.scala 305:37]
  wire [4:0] _T_1129; // @[Compute.scala 306:60]
  wire [31:0] _T_1130; // @[Compute.scala 306:49]
  wire [31:0] _T_1131; // @[Compute.scala 306:84]
  wire [31:0] shr_val_10; // @[Compute.scala 283:36]
  wire [31:0] shr_res_10; // @[Compute.scala 283:36]
  wire [7:0] _T_1132; // @[Compute.scala 308:37]
  wire [31:0] src_0_11; // @[Compute.scala 283:36]
  wire [31:0] src_1_11; // @[Compute.scala 283:36]
  wire  _T_1133; // @[Compute.scala 300:34]
  wire [31:0] _T_1134; // @[Compute.scala 300:24]
  wire [31:0] mix_val_11; // @[Compute.scala 283:36]
  wire [7:0] _T_1135; // @[Compute.scala 302:37]
  wire [31:0] _T_1136; // @[Compute.scala 303:30]
  wire [31:0] _T_1137; // @[Compute.scala 303:59]
  wire [32:0] _T_1138; // @[Compute.scala 303:49]
  wire [31:0] _T_1139; // @[Compute.scala 303:49]
  wire [31:0] _T_1140; // @[Compute.scala 303:79]
  wire [31:0] add_val_11; // @[Compute.scala 283:36]
  wire [31:0] add_res_11; // @[Compute.scala 283:36]
  wire [7:0] _T_1141; // @[Compute.scala 305:37]
  wire [4:0] _T_1143; // @[Compute.scala 306:60]
  wire [31:0] _T_1144; // @[Compute.scala 306:49]
  wire [31:0] _T_1145; // @[Compute.scala 306:84]
  wire [31:0] shr_val_11; // @[Compute.scala 283:36]
  wire [31:0] shr_res_11; // @[Compute.scala 283:36]
  wire [7:0] _T_1146; // @[Compute.scala 308:37]
  wire [31:0] src_0_12; // @[Compute.scala 283:36]
  wire [31:0] src_1_12; // @[Compute.scala 283:36]
  wire  _T_1147; // @[Compute.scala 300:34]
  wire [31:0] _T_1148; // @[Compute.scala 300:24]
  wire [31:0] mix_val_12; // @[Compute.scala 283:36]
  wire [7:0] _T_1149; // @[Compute.scala 302:37]
  wire [31:0] _T_1150; // @[Compute.scala 303:30]
  wire [31:0] _T_1151; // @[Compute.scala 303:59]
  wire [32:0] _T_1152; // @[Compute.scala 303:49]
  wire [31:0] _T_1153; // @[Compute.scala 303:49]
  wire [31:0] _T_1154; // @[Compute.scala 303:79]
  wire [31:0] add_val_12; // @[Compute.scala 283:36]
  wire [31:0] add_res_12; // @[Compute.scala 283:36]
  wire [7:0] _T_1155; // @[Compute.scala 305:37]
  wire [4:0] _T_1157; // @[Compute.scala 306:60]
  wire [31:0] _T_1158; // @[Compute.scala 306:49]
  wire [31:0] _T_1159; // @[Compute.scala 306:84]
  wire [31:0] shr_val_12; // @[Compute.scala 283:36]
  wire [31:0] shr_res_12; // @[Compute.scala 283:36]
  wire [7:0] _T_1160; // @[Compute.scala 308:37]
  wire [31:0] src_0_13; // @[Compute.scala 283:36]
  wire [31:0] src_1_13; // @[Compute.scala 283:36]
  wire  _T_1161; // @[Compute.scala 300:34]
  wire [31:0] _T_1162; // @[Compute.scala 300:24]
  wire [31:0] mix_val_13; // @[Compute.scala 283:36]
  wire [7:0] _T_1163; // @[Compute.scala 302:37]
  wire [31:0] _T_1164; // @[Compute.scala 303:30]
  wire [31:0] _T_1165; // @[Compute.scala 303:59]
  wire [32:0] _T_1166; // @[Compute.scala 303:49]
  wire [31:0] _T_1167; // @[Compute.scala 303:49]
  wire [31:0] _T_1168; // @[Compute.scala 303:79]
  wire [31:0] add_val_13; // @[Compute.scala 283:36]
  wire [31:0] add_res_13; // @[Compute.scala 283:36]
  wire [7:0] _T_1169; // @[Compute.scala 305:37]
  wire [4:0] _T_1171; // @[Compute.scala 306:60]
  wire [31:0] _T_1172; // @[Compute.scala 306:49]
  wire [31:0] _T_1173; // @[Compute.scala 306:84]
  wire [31:0] shr_val_13; // @[Compute.scala 283:36]
  wire [31:0] shr_res_13; // @[Compute.scala 283:36]
  wire [7:0] _T_1174; // @[Compute.scala 308:37]
  wire [31:0] src_0_14; // @[Compute.scala 283:36]
  wire [31:0] src_1_14; // @[Compute.scala 283:36]
  wire  _T_1175; // @[Compute.scala 300:34]
  wire [31:0] _T_1176; // @[Compute.scala 300:24]
  wire [31:0] mix_val_14; // @[Compute.scala 283:36]
  wire [7:0] _T_1177; // @[Compute.scala 302:37]
  wire [31:0] _T_1178; // @[Compute.scala 303:30]
  wire [31:0] _T_1179; // @[Compute.scala 303:59]
  wire [32:0] _T_1180; // @[Compute.scala 303:49]
  wire [31:0] _T_1181; // @[Compute.scala 303:49]
  wire [31:0] _T_1182; // @[Compute.scala 303:79]
  wire [31:0] add_val_14; // @[Compute.scala 283:36]
  wire [31:0] add_res_14; // @[Compute.scala 283:36]
  wire [7:0] _T_1183; // @[Compute.scala 305:37]
  wire [4:0] _T_1185; // @[Compute.scala 306:60]
  wire [31:0] _T_1186; // @[Compute.scala 306:49]
  wire [31:0] _T_1187; // @[Compute.scala 306:84]
  wire [31:0] shr_val_14; // @[Compute.scala 283:36]
  wire [31:0] shr_res_14; // @[Compute.scala 283:36]
  wire [7:0] _T_1188; // @[Compute.scala 308:37]
  wire [31:0] src_0_15; // @[Compute.scala 283:36]
  wire [31:0] src_1_15; // @[Compute.scala 283:36]
  wire  _T_1189; // @[Compute.scala 300:34]
  wire [31:0] _T_1190; // @[Compute.scala 300:24]
  wire [31:0] mix_val_15; // @[Compute.scala 283:36]
  wire [7:0] _T_1191; // @[Compute.scala 302:37]
  wire [31:0] _T_1192; // @[Compute.scala 303:30]
  wire [31:0] _T_1193; // @[Compute.scala 303:59]
  wire [32:0] _T_1194; // @[Compute.scala 303:49]
  wire [31:0] _T_1195; // @[Compute.scala 303:49]
  wire [31:0] _T_1196; // @[Compute.scala 303:79]
  wire [31:0] add_val_15; // @[Compute.scala 283:36]
  wire [31:0] add_res_15; // @[Compute.scala 283:36]
  wire [7:0] _T_1197; // @[Compute.scala 305:37]
  wire [4:0] _T_1199; // @[Compute.scala 306:60]
  wire [31:0] _T_1200; // @[Compute.scala 306:49]
  wire [31:0] _T_1201; // @[Compute.scala 306:84]
  wire [31:0] shr_val_15; // @[Compute.scala 283:36]
  wire [31:0] shr_res_15; // @[Compute.scala 283:36]
  wire [7:0] _T_1202; // @[Compute.scala 308:37]
  wire [7:0] short_cmp_res_0; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_0; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_0; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_1; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_1; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_1; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_2; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_2; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_2; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_3; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_3; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_3; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_4; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_4; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_4; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_5; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_5; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_5; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_6; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_6; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_6; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_7; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_7; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_7; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_8; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_8; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_8; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_9; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_9; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_9; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_10; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_10; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_10; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_11; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_11; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_11; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_12; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_12; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_12; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_13; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_13; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_13; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_14; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_14; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_14; // @[Compute.scala 283:36]
  wire [7:0] short_cmp_res_15; // @[Compute.scala 283:36]
  wire [7:0] short_add_res_15; // @[Compute.scala 283:36]
  wire [7:0] short_shr_res_15; // @[Compute.scala 283:36]
  wire  alu_opcode_minmax_en; // @[Compute.scala 313:48]
  wire  alu_opcode_add_en; // @[Compute.scala 314:39]
  wire  _T_1205; // @[Compute.scala 315:37]
  wire  _T_1206; // @[Compute.scala 315:34]
  wire  _T_1207; // @[Compute.scala 315:52]
  wire [4:0] _T_1209; // @[Compute.scala 316:58]
  wire [4:0] _T_1210; // @[Compute.scala 316:58]
  wire [3:0] _T_1211; // @[Compute.scala 316:58]
  wire [15:0] _GEN_310; // @[Compute.scala 316:40]
  wire  _T_1212; // @[Compute.scala 316:40]
  wire  _T_1213; // @[Compute.scala 316:23]
  wire  _T_1216; // @[Compute.scala 316:66]
  wire  _GEN_275; // @[Compute.scala 316:85]
  wire [16:0] _T_1219; // @[Compute.scala 319:34]
  wire [16:0] _T_1220; // @[Compute.scala 319:34]
  wire [15:0] _T_1221; // @[Compute.scala 319:34]
  wire [22:0] _GEN_311; // @[Compute.scala 319:41]
  wire [22:0] _T_1223; // @[Compute.scala 319:41]
  wire [63:0] _T_1230; // @[Cat.scala 30:58]
  wire [127:0] _T_1238; // @[Cat.scala 30:58]
  wire [63:0] _T_1245; // @[Cat.scala 30:58]
  wire [127:0] _T_1253; // @[Cat.scala 30:58]
  wire [63:0] _T_1260; // @[Cat.scala 30:58]
  wire [127:0] _T_1268; // @[Cat.scala 30:58]
  wire [127:0] _T_1269; // @[Compute.scala 323:8]
  assign acc_mem__T_416_addr = dst_idx[7:0];
  assign acc_mem__T_416_data = acc_mem[acc_mem__T_416_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_418_addr = src_idx[7:0];
  assign acc_mem__T_418_data = acc_mem[acc_mem__T_418_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_394_data = {_T_396,_T_395};
  assign acc_mem__T_394_addr = acc_sram_addr[7:0];
  assign acc_mem__T_394_mask = 1'h1;
  assign acc_mem__T_394_en = _T_305 ? _T_392 : 1'h0;
  assign uop_mem_uop_addr = 10'h0;
  assign uop_mem_uop_data = uop_mem[uop_mem_uop_addr]; // @[Compute.scala 34:20]
  assign uop_mem__T_352_data = uops_data;
  assign uop_mem__T_352_addr = uop_sram_addr[9:0];
  assign uop_mem__T_352_mask = 1'h1;
  assign uop_mem__T_352_en = 1'h1;
  assign started = reset == 1'h0; // @[Compute.scala 31:17]
  assign _T_201 = insn != 128'h0; // @[Compute.scala 37:31]
  assign insn_valid = _T_201 & started; // @[Compute.scala 37:40]
  assign opcode = insn[2:0]; // @[Compute.scala 39:29]
  assign pop_prev_dep = insn[3]; // @[Compute.scala 40:29]
  assign pop_next_dep = insn[4]; // @[Compute.scala 41:29]
  assign push_prev_dep = insn[5]; // @[Compute.scala 42:29]
  assign push_next_dep = insn[6]; // @[Compute.scala 43:29]
  assign memory_type = insn[8:7]; // @[Compute.scala 45:25]
  assign sram_base = insn[24:9]; // @[Compute.scala 46:25]
  assign dram_base = insn[56:25]; // @[Compute.scala 47:25]
  assign uop_cntr_max = insn[95:80]; // @[Compute.scala 49:25]
  assign y_pad_0 = insn[115:112]; // @[Compute.scala 51:25]
  assign x_pad_0 = insn[123:120]; // @[Compute.scala 53:25]
  assign x_pad_1 = insn[127:124]; // @[Compute.scala 54:25]
  assign _GEN_290 = {{12'd0}, y_pad_0}; // @[Compute.scala 58:30]
  assign _GEN_292 = {{12'd0}, x_pad_0}; // @[Compute.scala 59:30]
  assign _T_205 = _GEN_292 + uop_cntr_max; // @[Compute.scala 59:30]
  assign _T_206 = _GEN_292 + uop_cntr_max; // @[Compute.scala 59:30]
  assign _GEN_293 = {{12'd0}, x_pad_1}; // @[Compute.scala 59:39]
  assign _T_207 = _T_206 + _GEN_293; // @[Compute.scala 59:39]
  assign x_size_total = _T_206 + _GEN_293; // @[Compute.scala 59:39]
  assign y_offset = x_size_total * _GEN_290; // @[Compute.scala 60:31]
  assign _T_210 = opcode == 3'h0; // @[Compute.scala 64:32]
  assign _T_212 = opcode == 3'h1; // @[Compute.scala 64:60]
  assign opcode_load_en = _T_210 | _T_212; // @[Compute.scala 64:50]
  assign opcode_gemm_en = opcode == 3'h2; // @[Compute.scala 65:32]
  assign opcode_alu_en = opcode == 3'h4; // @[Compute.scala 66:31]
  assign memory_type_uop_en = memory_type == 2'h0; // @[Compute.scala 68:40]
  assign memory_type_acc_en = memory_type == 2'h3; // @[Compute.scala 69:40]
  assign idle = state == 3'h0; // @[Compute.scala 74:20]
  assign dump = state == 3'h1; // @[Compute.scala 75:20]
  assign busy = state == 3'h2; // @[Compute.scala 76:20]
  assign push = state == 3'h3; // @[Compute.scala 77:20]
  assign done = state == 3'h4; // @[Compute.scala 78:20]
  assign _T_234 = opcode_load_en & memory_type_uop_en; // @[Compute.scala 93:37]
  assign uop_cntr_en = _T_234 & insn_valid; // @[Compute.scala 93:59]
  assign _T_236 = uop_cntr_val == uop_cntr_max; // @[Compute.scala 96:38]
  assign _T_237 = _T_236 & uop_cntr_en; // @[Compute.scala 96:56]
  assign uop_cntr_wrap = _T_237 & busy; // @[Compute.scala 96:71]
  assign _T_239 = uop_cntr_max * 16'h4; // @[Compute.scala 98:29]
  assign _T_241 = _T_239 + 19'h1; // @[Compute.scala 98:46]
  assign acc_cntr_max = _T_239 + 19'h1; // @[Compute.scala 98:46]
  assign _T_242 = opcode_load_en & memory_type_acc_en; // @[Compute.scala 99:37]
  assign acc_cntr_en = _T_242 & insn_valid; // @[Compute.scala 99:59]
  assign _GEN_295 = {{3'd0}, acc_cntr_val}; // @[Compute.scala 102:38]
  assign _T_244 = _GEN_295 == acc_cntr_max; // @[Compute.scala 102:38]
  assign _T_245 = _T_244 & acc_cntr_en; // @[Compute.scala 102:56]
  assign acc_cntr_wrap = _T_245 & busy; // @[Compute.scala 102:71]
  assign _T_249 = opcode_alu_en | opcode_gemm_en; // @[Compute.scala 105:37]
  assign out_cntr_en = _T_249 & insn_valid; // @[Compute.scala 105:56]
  assign _T_251 = dst_offset_in == 16'h9; // @[Compute.scala 108:38]
  assign _T_252 = _T_251 & out_cntr_en; // @[Compute.scala 108:56]
  assign out_cntr_wrap = _T_252 & busy; // @[Compute.scala 108:71]
  assign push_prev_dep_valid = push_prev_dep & push; // @[Compute.scala 113:43]
  assign push_next_dep_valid = push_next_dep & push; // @[Compute.scala 114:43]
  assign _T_263 = uop_cntr_wrap | acc_cntr_wrap; // @[Compute.scala 121:23]
  assign _T_264 = _T_263 | out_cntr_wrap; // @[Compute.scala 121:40]
  assign _T_265 = push_prev_dep | push_next_dep; // @[Compute.scala 122:25]
  assign _GEN_0 = _T_265 ? 3'h3 : 3'h4; // @[Compute.scala 122:43]
  assign _GEN_1 = _T_264 ? _GEN_0 : state; // @[Compute.scala 121:58]
  assign _T_267 = pop_prev_dep_ready == 1'h0; // @[Compute.scala 130:18]
  assign _T_269 = pop_next_dep_ready == 1'h0; // @[Compute.scala 130:41]
  assign _T_270 = _T_267 & _T_269; // @[Compute.scala 130:38]
  assign _T_271 = busy & _T_270; // @[Compute.scala 130:14]
  assign _T_272 = pop_prev_dep | pop_next_dep; // @[Compute.scala 130:79]
  assign _T_273 = _T_271 & _T_272; // @[Compute.scala 130:62]
  assign _GEN_2 = _T_273 ? 3'h1 : _GEN_1; // @[Compute.scala 130:97]
  assign _T_274 = pop_prev_dep_ready | pop_next_dep_ready; // @[Compute.scala 131:38]
  assign _T_275 = dump & _T_274; // @[Compute.scala 131:14]
  assign _GEN_3 = _T_275 ? 3'h2 : _GEN_2; // @[Compute.scala 131:63]
  assign _T_276 = push_prev_dep_ready | push_next_dep_ready; // @[Compute.scala 132:38]
  assign _T_277 = push & _T_276; // @[Compute.scala 132:14]
  assign _GEN_4 = _T_277 ? 3'h4 : _GEN_3; // @[Compute.scala 132:63]
  assign _T_280 = pop_prev_dep & dump; // @[Compute.scala 139:22]
  assign _T_281 = _T_280 & io_l2g_dep_queue_valid; // @[Compute.scala 139:30]
  assign _GEN_5 = _T_281 ? 1'h1 : pop_prev_dep_ready; // @[Compute.scala 139:57]
  assign _T_283 = pop_next_dep & dump; // @[Compute.scala 142:22]
  assign _T_284 = _T_283 & io_s2g_dep_queue_valid; // @[Compute.scala 142:30]
  assign _GEN_6 = _T_284 ? 1'h1 : pop_next_dep_ready; // @[Compute.scala 142:57]
  assign _T_288 = push_prev_dep_valid & io_g2l_dep_queue_ready; // @[Compute.scala 149:29]
  assign _T_289 = _T_288 & push; // @[Compute.scala 149:55]
  assign _GEN_7 = _T_289 ? 1'h1 : push_prev_dep_ready; // @[Compute.scala 149:64]
  assign _T_291 = push_next_dep_valid & io_g2s_dep_queue_ready; // @[Compute.scala 152:29]
  assign _T_292 = _T_291 & push; // @[Compute.scala 152:55]
  assign _GEN_8 = _T_292 ? 1'h1 : push_next_dep_ready; // @[Compute.scala 152:64]
  assign _T_295 = io_uops_waitrequest == 1'h0; // @[Compute.scala 157:22]
  assign _T_296 = uops_read & _T_295; // @[Compute.scala 157:19]
  assign _T_297 = _T_296 & busy; // @[Compute.scala 157:37]
  assign _T_298 = uop_cntr_val < uop_cntr_max; // @[Compute.scala 157:61]
  assign _T_299 = _T_297 & _T_298; // @[Compute.scala 157:45]
  assign _T_301 = uop_cntr_val + 16'h1; // @[Compute.scala 158:34]
  assign _T_302 = uop_cntr_val + 16'h1; // @[Compute.scala 158:34]
  assign _GEN_9 = _T_299 ? _T_302 : uop_cntr_val; // @[Compute.scala 157:77]
  assign _T_304 = io_biases_waitrequest == 1'h0; // @[Compute.scala 160:24]
  assign _T_305 = biases_read & _T_304; // @[Compute.scala 160:21]
  assign _T_306 = _T_305 & busy; // @[Compute.scala 160:39]
  assign _T_307 = _GEN_295 < acc_cntr_max; // @[Compute.scala 160:63]
  assign _T_308 = _T_306 & _T_307; // @[Compute.scala 160:47]
  assign _T_310 = acc_cntr_val + 16'h1; // @[Compute.scala 161:34]
  assign _T_311 = acc_cntr_val + 16'h1; // @[Compute.scala 161:34]
  assign _GEN_10 = _T_308 ? _T_311 : acc_cntr_val; // @[Compute.scala 160:79]
  assign _T_313 = io_out_mem_waitrequest == 1'h0; // @[Compute.scala 163:26]
  assign _T_314 = out_mem_write & _T_313; // @[Compute.scala 163:23]
  assign _T_315 = _T_314 & busy; // @[Compute.scala 163:41]
  assign _T_316 = dst_offset_in < 16'h9; // @[Compute.scala 163:65]
  assign _T_317 = _T_315 & _T_316; // @[Compute.scala 163:49]
  assign _T_319 = dst_offset_in + 16'h1; // @[Compute.scala 164:34]
  assign _T_320 = dst_offset_in + 16'h1; // @[Compute.scala 164:34]
  assign _GEN_11 = _T_317 ? _T_320 : dst_offset_in; // @[Compute.scala 163:81]
  assign _GEN_16 = gemm_queue_ready ? 1'h0 : _GEN_5; // @[Compute.scala 168:27]
  assign _GEN_17 = gemm_queue_ready ? 1'h0 : _GEN_6; // @[Compute.scala 168:27]
  assign _GEN_18 = gemm_queue_ready ? 1'h0 : _GEN_7; // @[Compute.scala 168:27]
  assign _GEN_19 = gemm_queue_ready ? 1'h0 : _GEN_8; // @[Compute.scala 168:27]
  assign _GEN_20 = gemm_queue_ready ? 3'h2 : _GEN_4; // @[Compute.scala 168:27]
  assign _T_328 = idle | done; // @[Compute.scala 181:52]
  assign _T_329 = io_gemm_queue_valid & _T_328; // @[Compute.scala 181:43]
  assign _GEN_21 = gemm_queue_ready ? 1'h0 : _T_329; // @[Compute.scala 183:27]
  assign _GEN_297 = {{16'd0}, uop_cntr_val}; // @[Compute.scala 193:33]
  assign _T_334 = dram_base + _GEN_297; // @[Compute.scala 193:33]
  assign _T_335 = dram_base + _GEN_297; // @[Compute.scala 193:33]
  assign _GEN_298 = {{3'd0}, _T_335}; // @[Compute.scala 193:49]
  assign uop_dram_addr = _GEN_298 << 2'h2; // @[Compute.scala 193:49]
  assign _T_337 = sram_base + uop_cntr_val; // @[Compute.scala 194:33]
  assign uop_sram_addr = sram_base + uop_cntr_val; // @[Compute.scala 194:33]
  assign _T_339 = uop_cntr_wrap == 1'h0; // @[Compute.scala 195:31]
  assign _T_340 = uop_cntr_en & _T_339; // @[Compute.scala 195:28]
  assign _T_341 = _T_340 & busy; // @[Compute.scala 195:46]
  assign _T_346 = uop_cntr_max - 16'h1; // @[Compute.scala 202:42]
  assign _T_347 = $unsigned(_T_346); // @[Compute.scala 202:42]
  assign _T_348 = _T_347[15:0]; // @[Compute.scala 202:42]
  assign _T_349 = uop_cntr_val == _T_348; // @[Compute.scala 202:24]
  assign _GEN_22 = _T_349 ? 1'h0 : _T_341; // @[Compute.scala 202:50]
  assign _GEN_299 = {{12'd0}, y_offset}; // @[Compute.scala 207:36]
  assign _T_353 = dram_base + _GEN_299; // @[Compute.scala 207:36]
  assign _T_354 = dram_base + _GEN_299; // @[Compute.scala 207:36]
  assign _GEN_300 = {{28'd0}, x_pad_0}; // @[Compute.scala 207:47]
  assign _T_355 = _T_354 + _GEN_300; // @[Compute.scala 207:47]
  assign _T_356 = _T_354 + _GEN_300; // @[Compute.scala 207:47]
  assign _GEN_301 = {{3'd0}, _T_356}; // @[Compute.scala 207:58]
  assign _T_358 = _GEN_301 << 2'h2; // @[Compute.scala 207:58]
  assign _T_360 = _T_358 * 35'h1; // @[Compute.scala 207:66]
  assign _GEN_302 = {{20'd0}, acc_cntr_val}; // @[Compute.scala 207:76]
  assign _T_361 = _T_360 + _GEN_302; // @[Compute.scala 207:76]
  assign _T_362 = _T_360 + _GEN_302; // @[Compute.scala 207:76]
  assign _GEN_303 = {{7'd0}, _T_362}; // @[Compute.scala 207:92]
  assign acc_dram_addr = _GEN_303 << 3'h4; // @[Compute.scala 207:92]
  assign _GEN_304 = {{4'd0}, sram_base}; // @[Compute.scala 208:36]
  assign _T_364 = _GEN_304 + y_offset; // @[Compute.scala 208:36]
  assign _T_365 = _GEN_304 + y_offset; // @[Compute.scala 208:36]
  assign _GEN_305 = {{16'd0}, x_pad_0}; // @[Compute.scala 208:47]
  assign _T_366 = _T_365 + _GEN_305; // @[Compute.scala 208:47]
  assign _T_367 = _T_365 + _GEN_305; // @[Compute.scala 208:47]
  assign _GEN_306 = {{3'd0}, _T_367}; // @[Compute.scala 208:58]
  assign _T_369 = _GEN_306 << 2'h2; // @[Compute.scala 208:58]
  assign _T_371 = _T_369 * 23'h1; // @[Compute.scala 208:66]
  assign _GEN_307 = {{8'd0}, acc_cntr_val}; // @[Compute.scala 208:76]
  assign _T_372 = _T_371 + _GEN_307; // @[Compute.scala 208:76]
  assign _T_373 = _T_371 + _GEN_307; // @[Compute.scala 208:76]
  assign _T_375 = _T_373 >> 2'h2; // @[Compute.scala 208:92]
  assign _T_377 = _T_375 - 24'h1; // @[Compute.scala 208:100]
  assign _T_378 = $unsigned(_T_377); // @[Compute.scala 208:100]
  assign acc_sram_addr = _T_378[23:0]; // @[Compute.scala 208:100]
  assign _T_380 = done == 1'h0; // @[Compute.scala 209:33]
  assign _GEN_12 = acc_cntr_val % 16'h4; // @[Compute.scala 215:30]
  assign _T_386 = _GEN_12[2:0]; // @[Compute.scala 215:30]
  assign _GEN_25 = 3'h0 == _T_386 ? io_biases_readdata : biases_data_0; // @[Compute.scala 215:48]
  assign _GEN_26 = 3'h1 == _T_386 ? io_biases_readdata : biases_data_1; // @[Compute.scala 215:48]
  assign _GEN_27 = 3'h2 == _T_386 ? io_biases_readdata : biases_data_2; // @[Compute.scala 215:48]
  assign _GEN_28 = 3'h3 == _T_386 ? io_biases_readdata : biases_data_3; // @[Compute.scala 215:48]
  assign _T_392 = _T_386 == 3'h0; // @[Compute.scala 219:43]
  assign _T_395 = {biases_data_1,biases_data_0}; // @[Cat.scala 30:58]
  assign _T_396 = {biases_data_3,biases_data_2}; // @[Cat.scala 30:58]
  assign alu_opcode = insn[109:108]; // @[Compute.scala 229:24]
  assign use_imm = insn[110]; // @[Compute.scala 230:21]
  assign imm_raw = insn[126:111]; // @[Compute.scala 231:21]
  assign _T_398 = $signed(imm_raw); // @[Compute.scala 232:25]
  assign _T_400 = $signed(_T_398) < $signed(16'sh0); // @[Compute.scala 232:32]
  assign _T_402 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_404 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_405 = _T_400 ? _T_402 : {{15'd0}, _T_404}; // @[Compute.scala 232:16]
  assign imm = $signed(_T_405); // @[Compute.scala 232:89]
  assign _T_406 = uop_mem_uop_data[10:0]; // @[Compute.scala 240:20]
  assign _GEN_308 = {{5'd0}, _T_406}; // @[Compute.scala 240:47]
  assign _T_407 = _GEN_308 + dst_offset_in; // @[Compute.scala 240:47]
  assign dst_idx = _GEN_308 + dst_offset_in; // @[Compute.scala 240:47]
  assign _T_408 = uop_mem_uop_data[21:11]; // @[Compute.scala 241:20]
  assign _GEN_309 = {{5'd0}, _T_408}; // @[Compute.scala 241:47]
  assign _T_409 = _GEN_309 + dst_offset_in; // @[Compute.scala 241:47]
  assign src_idx = _GEN_309 + dst_offset_in; // @[Compute.scala 241:47]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Compute.scala 263:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Compute.scala 264:38]
  assign _T_850 = insn_valid & out_cntr_en; // @[Compute.scala 283:20]
  assign _T_851 = src_vector[31:0]; // @[Compute.scala 286:31]
  assign _T_852 = $signed(_T_851); // @[Compute.scala 286:72]
  assign _T_853 = dst_vector[31:0]; // @[Compute.scala 287:31]
  assign _T_854 = $signed(_T_853); // @[Compute.scala 287:72]
  assign _T_855 = src_vector[63:32]; // @[Compute.scala 286:31]
  assign _T_856 = $signed(_T_855); // @[Compute.scala 286:72]
  assign _T_857 = dst_vector[63:32]; // @[Compute.scala 287:31]
  assign _T_858 = $signed(_T_857); // @[Compute.scala 287:72]
  assign _T_859 = src_vector[95:64]; // @[Compute.scala 286:31]
  assign _T_860 = $signed(_T_859); // @[Compute.scala 286:72]
  assign _T_861 = dst_vector[95:64]; // @[Compute.scala 287:31]
  assign _T_862 = $signed(_T_861); // @[Compute.scala 287:72]
  assign _T_863 = src_vector[127:96]; // @[Compute.scala 286:31]
  assign _T_864 = $signed(_T_863); // @[Compute.scala 286:72]
  assign _T_865 = dst_vector[127:96]; // @[Compute.scala 287:31]
  assign _T_866 = $signed(_T_865); // @[Compute.scala 287:72]
  assign _T_867 = src_vector[159:128]; // @[Compute.scala 286:31]
  assign _T_868 = $signed(_T_867); // @[Compute.scala 286:72]
  assign _T_869 = dst_vector[159:128]; // @[Compute.scala 287:31]
  assign _T_870 = $signed(_T_869); // @[Compute.scala 287:72]
  assign _T_871 = src_vector[191:160]; // @[Compute.scala 286:31]
  assign _T_872 = $signed(_T_871); // @[Compute.scala 286:72]
  assign _T_873 = dst_vector[191:160]; // @[Compute.scala 287:31]
  assign _T_874 = $signed(_T_873); // @[Compute.scala 287:72]
  assign _T_875 = src_vector[223:192]; // @[Compute.scala 286:31]
  assign _T_876 = $signed(_T_875); // @[Compute.scala 286:72]
  assign _T_877 = dst_vector[223:192]; // @[Compute.scala 287:31]
  assign _T_878 = $signed(_T_877); // @[Compute.scala 287:72]
  assign _T_879 = src_vector[255:224]; // @[Compute.scala 286:31]
  assign _T_880 = $signed(_T_879); // @[Compute.scala 286:72]
  assign _T_881 = dst_vector[255:224]; // @[Compute.scala 287:31]
  assign _T_882 = $signed(_T_881); // @[Compute.scala 287:72]
  assign _T_883 = src_vector[287:256]; // @[Compute.scala 286:31]
  assign _T_884 = $signed(_T_883); // @[Compute.scala 286:72]
  assign _T_885 = dst_vector[287:256]; // @[Compute.scala 287:31]
  assign _T_886 = $signed(_T_885); // @[Compute.scala 287:72]
  assign _T_887 = src_vector[319:288]; // @[Compute.scala 286:31]
  assign _T_888 = $signed(_T_887); // @[Compute.scala 286:72]
  assign _T_889 = dst_vector[319:288]; // @[Compute.scala 287:31]
  assign _T_890 = $signed(_T_889); // @[Compute.scala 287:72]
  assign _T_891 = src_vector[351:320]; // @[Compute.scala 286:31]
  assign _T_892 = $signed(_T_891); // @[Compute.scala 286:72]
  assign _T_893 = dst_vector[351:320]; // @[Compute.scala 287:31]
  assign _T_894 = $signed(_T_893); // @[Compute.scala 287:72]
  assign _T_895 = src_vector[383:352]; // @[Compute.scala 286:31]
  assign _T_896 = $signed(_T_895); // @[Compute.scala 286:72]
  assign _T_897 = dst_vector[383:352]; // @[Compute.scala 287:31]
  assign _T_898 = $signed(_T_897); // @[Compute.scala 287:72]
  assign _T_899 = src_vector[415:384]; // @[Compute.scala 286:31]
  assign _T_900 = $signed(_T_899); // @[Compute.scala 286:72]
  assign _T_901 = dst_vector[415:384]; // @[Compute.scala 287:31]
  assign _T_902 = $signed(_T_901); // @[Compute.scala 287:72]
  assign _T_903 = src_vector[447:416]; // @[Compute.scala 286:31]
  assign _T_904 = $signed(_T_903); // @[Compute.scala 286:72]
  assign _T_905 = dst_vector[447:416]; // @[Compute.scala 287:31]
  assign _T_906 = $signed(_T_905); // @[Compute.scala 287:72]
  assign _T_907 = src_vector[479:448]; // @[Compute.scala 286:31]
  assign _T_908 = $signed(_T_907); // @[Compute.scala 286:72]
  assign _T_909 = dst_vector[479:448]; // @[Compute.scala 287:31]
  assign _T_910 = $signed(_T_909); // @[Compute.scala 287:72]
  assign _T_911 = src_vector[511:480]; // @[Compute.scala 286:31]
  assign _T_912 = $signed(_T_911); // @[Compute.scala 286:72]
  assign _T_913 = dst_vector[511:480]; // @[Compute.scala 287:31]
  assign _T_914 = $signed(_T_913); // @[Compute.scala 287:72]
  assign _GEN_51 = alu_opcode_max_en ? $signed(_T_852) : $signed(_T_854); // @[Compute.scala 284:30]
  assign _GEN_52 = alu_opcode_max_en ? $signed(_T_854) : $signed(_T_852); // @[Compute.scala 284:30]
  assign _GEN_53 = alu_opcode_max_en ? $signed(_T_856) : $signed(_T_858); // @[Compute.scala 284:30]
  assign _GEN_54 = alu_opcode_max_en ? $signed(_T_858) : $signed(_T_856); // @[Compute.scala 284:30]
  assign _GEN_55 = alu_opcode_max_en ? $signed(_T_860) : $signed(_T_862); // @[Compute.scala 284:30]
  assign _GEN_56 = alu_opcode_max_en ? $signed(_T_862) : $signed(_T_860); // @[Compute.scala 284:30]
  assign _GEN_57 = alu_opcode_max_en ? $signed(_T_864) : $signed(_T_866); // @[Compute.scala 284:30]
  assign _GEN_58 = alu_opcode_max_en ? $signed(_T_866) : $signed(_T_864); // @[Compute.scala 284:30]
  assign _GEN_59 = alu_opcode_max_en ? $signed(_T_868) : $signed(_T_870); // @[Compute.scala 284:30]
  assign _GEN_60 = alu_opcode_max_en ? $signed(_T_870) : $signed(_T_868); // @[Compute.scala 284:30]
  assign _GEN_61 = alu_opcode_max_en ? $signed(_T_872) : $signed(_T_874); // @[Compute.scala 284:30]
  assign _GEN_62 = alu_opcode_max_en ? $signed(_T_874) : $signed(_T_872); // @[Compute.scala 284:30]
  assign _GEN_63 = alu_opcode_max_en ? $signed(_T_876) : $signed(_T_878); // @[Compute.scala 284:30]
  assign _GEN_64 = alu_opcode_max_en ? $signed(_T_878) : $signed(_T_876); // @[Compute.scala 284:30]
  assign _GEN_65 = alu_opcode_max_en ? $signed(_T_880) : $signed(_T_882); // @[Compute.scala 284:30]
  assign _GEN_66 = alu_opcode_max_en ? $signed(_T_882) : $signed(_T_880); // @[Compute.scala 284:30]
  assign _GEN_67 = alu_opcode_max_en ? $signed(_T_884) : $signed(_T_886); // @[Compute.scala 284:30]
  assign _GEN_68 = alu_opcode_max_en ? $signed(_T_886) : $signed(_T_884); // @[Compute.scala 284:30]
  assign _GEN_69 = alu_opcode_max_en ? $signed(_T_888) : $signed(_T_890); // @[Compute.scala 284:30]
  assign _GEN_70 = alu_opcode_max_en ? $signed(_T_890) : $signed(_T_888); // @[Compute.scala 284:30]
  assign _GEN_71 = alu_opcode_max_en ? $signed(_T_892) : $signed(_T_894); // @[Compute.scala 284:30]
  assign _GEN_72 = alu_opcode_max_en ? $signed(_T_894) : $signed(_T_892); // @[Compute.scala 284:30]
  assign _GEN_73 = alu_opcode_max_en ? $signed(_T_896) : $signed(_T_898); // @[Compute.scala 284:30]
  assign _GEN_74 = alu_opcode_max_en ? $signed(_T_898) : $signed(_T_896); // @[Compute.scala 284:30]
  assign _GEN_75 = alu_opcode_max_en ? $signed(_T_900) : $signed(_T_902); // @[Compute.scala 284:30]
  assign _GEN_76 = alu_opcode_max_en ? $signed(_T_902) : $signed(_T_900); // @[Compute.scala 284:30]
  assign _GEN_77 = alu_opcode_max_en ? $signed(_T_904) : $signed(_T_906); // @[Compute.scala 284:30]
  assign _GEN_78 = alu_opcode_max_en ? $signed(_T_906) : $signed(_T_904); // @[Compute.scala 284:30]
  assign _GEN_79 = alu_opcode_max_en ? $signed(_T_908) : $signed(_T_910); // @[Compute.scala 284:30]
  assign _GEN_80 = alu_opcode_max_en ? $signed(_T_910) : $signed(_T_908); // @[Compute.scala 284:30]
  assign _GEN_81 = alu_opcode_max_en ? $signed(_T_912) : $signed(_T_914); // @[Compute.scala 284:30]
  assign _GEN_82 = alu_opcode_max_en ? $signed(_T_914) : $signed(_T_912); // @[Compute.scala 284:30]
  assign _GEN_83 = use_imm ? $signed(imm) : $signed(_GEN_52); // @[Compute.scala 295:20]
  assign _GEN_84 = use_imm ? $signed(imm) : $signed(_GEN_54); // @[Compute.scala 295:20]
  assign _GEN_85 = use_imm ? $signed(imm) : $signed(_GEN_56); // @[Compute.scala 295:20]
  assign _GEN_86 = use_imm ? $signed(imm) : $signed(_GEN_58); // @[Compute.scala 295:20]
  assign _GEN_87 = use_imm ? $signed(imm) : $signed(_GEN_60); // @[Compute.scala 295:20]
  assign _GEN_88 = use_imm ? $signed(imm) : $signed(_GEN_62); // @[Compute.scala 295:20]
  assign _GEN_89 = use_imm ? $signed(imm) : $signed(_GEN_64); // @[Compute.scala 295:20]
  assign _GEN_90 = use_imm ? $signed(imm) : $signed(_GEN_66); // @[Compute.scala 295:20]
  assign _GEN_91 = use_imm ? $signed(imm) : $signed(_GEN_68); // @[Compute.scala 295:20]
  assign _GEN_92 = use_imm ? $signed(imm) : $signed(_GEN_70); // @[Compute.scala 295:20]
  assign _GEN_93 = use_imm ? $signed(imm) : $signed(_GEN_72); // @[Compute.scala 295:20]
  assign _GEN_94 = use_imm ? $signed(imm) : $signed(_GEN_74); // @[Compute.scala 295:20]
  assign _GEN_95 = use_imm ? $signed(imm) : $signed(_GEN_76); // @[Compute.scala 295:20]
  assign _GEN_96 = use_imm ? $signed(imm) : $signed(_GEN_78); // @[Compute.scala 295:20]
  assign _GEN_97 = use_imm ? $signed(imm) : $signed(_GEN_80); // @[Compute.scala 295:20]
  assign _GEN_98 = use_imm ? $signed(imm) : $signed(_GEN_82); // @[Compute.scala 295:20]
  assign src_0_0 = _T_850 ? $signed(_GEN_51) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_0 = _T_850 ? $signed(_GEN_83) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_979 = $signed(src_0_0) < $signed(src_1_0); // @[Compute.scala 300:34]
  assign _T_980 = _T_979 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 300:24]
  assign mix_val_0 = _T_850 ? $signed(_T_980) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_981 = mix_val_0[7:0]; // @[Compute.scala 302:37]
  assign _T_982 = $unsigned(src_0_0); // @[Compute.scala 303:30]
  assign _T_983 = $unsigned(src_1_0); // @[Compute.scala 303:59]
  assign _T_984 = _T_982 + _T_983; // @[Compute.scala 303:49]
  assign _T_985 = _T_982 + _T_983; // @[Compute.scala 303:49]
  assign _T_986 = $signed(_T_985); // @[Compute.scala 303:79]
  assign add_val_0 = _T_850 ? $signed(_T_986) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_0 = _T_850 ? $signed(add_val_0) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_987 = add_res_0[7:0]; // @[Compute.scala 305:37]
  assign _T_989 = src_1_0[4:0]; // @[Compute.scala 306:60]
  assign _T_990 = _T_982 >> _T_989; // @[Compute.scala 306:49]
  assign _T_991 = $signed(_T_990); // @[Compute.scala 306:84]
  assign shr_val_0 = _T_850 ? $signed(_T_991) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_0 = _T_850 ? $signed(shr_val_0) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_992 = shr_res_0[7:0]; // @[Compute.scala 308:37]
  assign src_0_1 = _T_850 ? $signed(_GEN_53) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_1 = _T_850 ? $signed(_GEN_84) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_993 = $signed(src_0_1) < $signed(src_1_1); // @[Compute.scala 300:34]
  assign _T_994 = _T_993 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 300:24]
  assign mix_val_1 = _T_850 ? $signed(_T_994) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_995 = mix_val_1[7:0]; // @[Compute.scala 302:37]
  assign _T_996 = $unsigned(src_0_1); // @[Compute.scala 303:30]
  assign _T_997 = $unsigned(src_1_1); // @[Compute.scala 303:59]
  assign _T_998 = _T_996 + _T_997; // @[Compute.scala 303:49]
  assign _T_999 = _T_996 + _T_997; // @[Compute.scala 303:49]
  assign _T_1000 = $signed(_T_999); // @[Compute.scala 303:79]
  assign add_val_1 = _T_850 ? $signed(_T_1000) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_1 = _T_850 ? $signed(add_val_1) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1001 = add_res_1[7:0]; // @[Compute.scala 305:37]
  assign _T_1003 = src_1_1[4:0]; // @[Compute.scala 306:60]
  assign _T_1004 = _T_996 >> _T_1003; // @[Compute.scala 306:49]
  assign _T_1005 = $signed(_T_1004); // @[Compute.scala 306:84]
  assign shr_val_1 = _T_850 ? $signed(_T_1005) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_1 = _T_850 ? $signed(shr_val_1) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1006 = shr_res_1[7:0]; // @[Compute.scala 308:37]
  assign src_0_2 = _T_850 ? $signed(_GEN_55) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_2 = _T_850 ? $signed(_GEN_85) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1007 = $signed(src_0_2) < $signed(src_1_2); // @[Compute.scala 300:34]
  assign _T_1008 = _T_1007 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 300:24]
  assign mix_val_2 = _T_850 ? $signed(_T_1008) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1009 = mix_val_2[7:0]; // @[Compute.scala 302:37]
  assign _T_1010 = $unsigned(src_0_2); // @[Compute.scala 303:30]
  assign _T_1011 = $unsigned(src_1_2); // @[Compute.scala 303:59]
  assign _T_1012 = _T_1010 + _T_1011; // @[Compute.scala 303:49]
  assign _T_1013 = _T_1010 + _T_1011; // @[Compute.scala 303:49]
  assign _T_1014 = $signed(_T_1013); // @[Compute.scala 303:79]
  assign add_val_2 = _T_850 ? $signed(_T_1014) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_2 = _T_850 ? $signed(add_val_2) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1015 = add_res_2[7:0]; // @[Compute.scala 305:37]
  assign _T_1017 = src_1_2[4:0]; // @[Compute.scala 306:60]
  assign _T_1018 = _T_1010 >> _T_1017; // @[Compute.scala 306:49]
  assign _T_1019 = $signed(_T_1018); // @[Compute.scala 306:84]
  assign shr_val_2 = _T_850 ? $signed(_T_1019) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_2 = _T_850 ? $signed(shr_val_2) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1020 = shr_res_2[7:0]; // @[Compute.scala 308:37]
  assign src_0_3 = _T_850 ? $signed(_GEN_57) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_3 = _T_850 ? $signed(_GEN_86) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1021 = $signed(src_0_3) < $signed(src_1_3); // @[Compute.scala 300:34]
  assign _T_1022 = _T_1021 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 300:24]
  assign mix_val_3 = _T_850 ? $signed(_T_1022) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1023 = mix_val_3[7:0]; // @[Compute.scala 302:37]
  assign _T_1024 = $unsigned(src_0_3); // @[Compute.scala 303:30]
  assign _T_1025 = $unsigned(src_1_3); // @[Compute.scala 303:59]
  assign _T_1026 = _T_1024 + _T_1025; // @[Compute.scala 303:49]
  assign _T_1027 = _T_1024 + _T_1025; // @[Compute.scala 303:49]
  assign _T_1028 = $signed(_T_1027); // @[Compute.scala 303:79]
  assign add_val_3 = _T_850 ? $signed(_T_1028) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_3 = _T_850 ? $signed(add_val_3) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1029 = add_res_3[7:0]; // @[Compute.scala 305:37]
  assign _T_1031 = src_1_3[4:0]; // @[Compute.scala 306:60]
  assign _T_1032 = _T_1024 >> _T_1031; // @[Compute.scala 306:49]
  assign _T_1033 = $signed(_T_1032); // @[Compute.scala 306:84]
  assign shr_val_3 = _T_850 ? $signed(_T_1033) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_3 = _T_850 ? $signed(shr_val_3) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1034 = shr_res_3[7:0]; // @[Compute.scala 308:37]
  assign src_0_4 = _T_850 ? $signed(_GEN_59) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_4 = _T_850 ? $signed(_GEN_87) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1035 = $signed(src_0_4) < $signed(src_1_4); // @[Compute.scala 300:34]
  assign _T_1036 = _T_1035 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 300:24]
  assign mix_val_4 = _T_850 ? $signed(_T_1036) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1037 = mix_val_4[7:0]; // @[Compute.scala 302:37]
  assign _T_1038 = $unsigned(src_0_4); // @[Compute.scala 303:30]
  assign _T_1039 = $unsigned(src_1_4); // @[Compute.scala 303:59]
  assign _T_1040 = _T_1038 + _T_1039; // @[Compute.scala 303:49]
  assign _T_1041 = _T_1038 + _T_1039; // @[Compute.scala 303:49]
  assign _T_1042 = $signed(_T_1041); // @[Compute.scala 303:79]
  assign add_val_4 = _T_850 ? $signed(_T_1042) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_4 = _T_850 ? $signed(add_val_4) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1043 = add_res_4[7:0]; // @[Compute.scala 305:37]
  assign _T_1045 = src_1_4[4:0]; // @[Compute.scala 306:60]
  assign _T_1046 = _T_1038 >> _T_1045; // @[Compute.scala 306:49]
  assign _T_1047 = $signed(_T_1046); // @[Compute.scala 306:84]
  assign shr_val_4 = _T_850 ? $signed(_T_1047) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_4 = _T_850 ? $signed(shr_val_4) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1048 = shr_res_4[7:0]; // @[Compute.scala 308:37]
  assign src_0_5 = _T_850 ? $signed(_GEN_61) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_5 = _T_850 ? $signed(_GEN_88) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1049 = $signed(src_0_5) < $signed(src_1_5); // @[Compute.scala 300:34]
  assign _T_1050 = _T_1049 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 300:24]
  assign mix_val_5 = _T_850 ? $signed(_T_1050) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1051 = mix_val_5[7:0]; // @[Compute.scala 302:37]
  assign _T_1052 = $unsigned(src_0_5); // @[Compute.scala 303:30]
  assign _T_1053 = $unsigned(src_1_5); // @[Compute.scala 303:59]
  assign _T_1054 = _T_1052 + _T_1053; // @[Compute.scala 303:49]
  assign _T_1055 = _T_1052 + _T_1053; // @[Compute.scala 303:49]
  assign _T_1056 = $signed(_T_1055); // @[Compute.scala 303:79]
  assign add_val_5 = _T_850 ? $signed(_T_1056) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_5 = _T_850 ? $signed(add_val_5) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1057 = add_res_5[7:0]; // @[Compute.scala 305:37]
  assign _T_1059 = src_1_5[4:0]; // @[Compute.scala 306:60]
  assign _T_1060 = _T_1052 >> _T_1059; // @[Compute.scala 306:49]
  assign _T_1061 = $signed(_T_1060); // @[Compute.scala 306:84]
  assign shr_val_5 = _T_850 ? $signed(_T_1061) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_5 = _T_850 ? $signed(shr_val_5) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1062 = shr_res_5[7:0]; // @[Compute.scala 308:37]
  assign src_0_6 = _T_850 ? $signed(_GEN_63) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_6 = _T_850 ? $signed(_GEN_89) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1063 = $signed(src_0_6) < $signed(src_1_6); // @[Compute.scala 300:34]
  assign _T_1064 = _T_1063 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 300:24]
  assign mix_val_6 = _T_850 ? $signed(_T_1064) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1065 = mix_val_6[7:0]; // @[Compute.scala 302:37]
  assign _T_1066 = $unsigned(src_0_6); // @[Compute.scala 303:30]
  assign _T_1067 = $unsigned(src_1_6); // @[Compute.scala 303:59]
  assign _T_1068 = _T_1066 + _T_1067; // @[Compute.scala 303:49]
  assign _T_1069 = _T_1066 + _T_1067; // @[Compute.scala 303:49]
  assign _T_1070 = $signed(_T_1069); // @[Compute.scala 303:79]
  assign add_val_6 = _T_850 ? $signed(_T_1070) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_6 = _T_850 ? $signed(add_val_6) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1071 = add_res_6[7:0]; // @[Compute.scala 305:37]
  assign _T_1073 = src_1_6[4:0]; // @[Compute.scala 306:60]
  assign _T_1074 = _T_1066 >> _T_1073; // @[Compute.scala 306:49]
  assign _T_1075 = $signed(_T_1074); // @[Compute.scala 306:84]
  assign shr_val_6 = _T_850 ? $signed(_T_1075) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_6 = _T_850 ? $signed(shr_val_6) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1076 = shr_res_6[7:0]; // @[Compute.scala 308:37]
  assign src_0_7 = _T_850 ? $signed(_GEN_65) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_7 = _T_850 ? $signed(_GEN_90) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1077 = $signed(src_0_7) < $signed(src_1_7); // @[Compute.scala 300:34]
  assign _T_1078 = _T_1077 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 300:24]
  assign mix_val_7 = _T_850 ? $signed(_T_1078) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1079 = mix_val_7[7:0]; // @[Compute.scala 302:37]
  assign _T_1080 = $unsigned(src_0_7); // @[Compute.scala 303:30]
  assign _T_1081 = $unsigned(src_1_7); // @[Compute.scala 303:59]
  assign _T_1082 = _T_1080 + _T_1081; // @[Compute.scala 303:49]
  assign _T_1083 = _T_1080 + _T_1081; // @[Compute.scala 303:49]
  assign _T_1084 = $signed(_T_1083); // @[Compute.scala 303:79]
  assign add_val_7 = _T_850 ? $signed(_T_1084) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_7 = _T_850 ? $signed(add_val_7) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1085 = add_res_7[7:0]; // @[Compute.scala 305:37]
  assign _T_1087 = src_1_7[4:0]; // @[Compute.scala 306:60]
  assign _T_1088 = _T_1080 >> _T_1087; // @[Compute.scala 306:49]
  assign _T_1089 = $signed(_T_1088); // @[Compute.scala 306:84]
  assign shr_val_7 = _T_850 ? $signed(_T_1089) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_7 = _T_850 ? $signed(shr_val_7) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1090 = shr_res_7[7:0]; // @[Compute.scala 308:37]
  assign src_0_8 = _T_850 ? $signed(_GEN_67) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_8 = _T_850 ? $signed(_GEN_91) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1091 = $signed(src_0_8) < $signed(src_1_8); // @[Compute.scala 300:34]
  assign _T_1092 = _T_1091 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 300:24]
  assign mix_val_8 = _T_850 ? $signed(_T_1092) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1093 = mix_val_8[7:0]; // @[Compute.scala 302:37]
  assign _T_1094 = $unsigned(src_0_8); // @[Compute.scala 303:30]
  assign _T_1095 = $unsigned(src_1_8); // @[Compute.scala 303:59]
  assign _T_1096 = _T_1094 + _T_1095; // @[Compute.scala 303:49]
  assign _T_1097 = _T_1094 + _T_1095; // @[Compute.scala 303:49]
  assign _T_1098 = $signed(_T_1097); // @[Compute.scala 303:79]
  assign add_val_8 = _T_850 ? $signed(_T_1098) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_8 = _T_850 ? $signed(add_val_8) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1099 = add_res_8[7:0]; // @[Compute.scala 305:37]
  assign _T_1101 = src_1_8[4:0]; // @[Compute.scala 306:60]
  assign _T_1102 = _T_1094 >> _T_1101; // @[Compute.scala 306:49]
  assign _T_1103 = $signed(_T_1102); // @[Compute.scala 306:84]
  assign shr_val_8 = _T_850 ? $signed(_T_1103) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_8 = _T_850 ? $signed(shr_val_8) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1104 = shr_res_8[7:0]; // @[Compute.scala 308:37]
  assign src_0_9 = _T_850 ? $signed(_GEN_69) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_9 = _T_850 ? $signed(_GEN_92) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1105 = $signed(src_0_9) < $signed(src_1_9); // @[Compute.scala 300:34]
  assign _T_1106 = _T_1105 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 300:24]
  assign mix_val_9 = _T_850 ? $signed(_T_1106) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1107 = mix_val_9[7:0]; // @[Compute.scala 302:37]
  assign _T_1108 = $unsigned(src_0_9); // @[Compute.scala 303:30]
  assign _T_1109 = $unsigned(src_1_9); // @[Compute.scala 303:59]
  assign _T_1110 = _T_1108 + _T_1109; // @[Compute.scala 303:49]
  assign _T_1111 = _T_1108 + _T_1109; // @[Compute.scala 303:49]
  assign _T_1112 = $signed(_T_1111); // @[Compute.scala 303:79]
  assign add_val_9 = _T_850 ? $signed(_T_1112) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_9 = _T_850 ? $signed(add_val_9) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1113 = add_res_9[7:0]; // @[Compute.scala 305:37]
  assign _T_1115 = src_1_9[4:0]; // @[Compute.scala 306:60]
  assign _T_1116 = _T_1108 >> _T_1115; // @[Compute.scala 306:49]
  assign _T_1117 = $signed(_T_1116); // @[Compute.scala 306:84]
  assign shr_val_9 = _T_850 ? $signed(_T_1117) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_9 = _T_850 ? $signed(shr_val_9) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1118 = shr_res_9[7:0]; // @[Compute.scala 308:37]
  assign src_0_10 = _T_850 ? $signed(_GEN_71) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_10 = _T_850 ? $signed(_GEN_93) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1119 = $signed(src_0_10) < $signed(src_1_10); // @[Compute.scala 300:34]
  assign _T_1120 = _T_1119 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 300:24]
  assign mix_val_10 = _T_850 ? $signed(_T_1120) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1121 = mix_val_10[7:0]; // @[Compute.scala 302:37]
  assign _T_1122 = $unsigned(src_0_10); // @[Compute.scala 303:30]
  assign _T_1123 = $unsigned(src_1_10); // @[Compute.scala 303:59]
  assign _T_1124 = _T_1122 + _T_1123; // @[Compute.scala 303:49]
  assign _T_1125 = _T_1122 + _T_1123; // @[Compute.scala 303:49]
  assign _T_1126 = $signed(_T_1125); // @[Compute.scala 303:79]
  assign add_val_10 = _T_850 ? $signed(_T_1126) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_10 = _T_850 ? $signed(add_val_10) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1127 = add_res_10[7:0]; // @[Compute.scala 305:37]
  assign _T_1129 = src_1_10[4:0]; // @[Compute.scala 306:60]
  assign _T_1130 = _T_1122 >> _T_1129; // @[Compute.scala 306:49]
  assign _T_1131 = $signed(_T_1130); // @[Compute.scala 306:84]
  assign shr_val_10 = _T_850 ? $signed(_T_1131) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_10 = _T_850 ? $signed(shr_val_10) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1132 = shr_res_10[7:0]; // @[Compute.scala 308:37]
  assign src_0_11 = _T_850 ? $signed(_GEN_73) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_11 = _T_850 ? $signed(_GEN_94) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1133 = $signed(src_0_11) < $signed(src_1_11); // @[Compute.scala 300:34]
  assign _T_1134 = _T_1133 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 300:24]
  assign mix_val_11 = _T_850 ? $signed(_T_1134) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1135 = mix_val_11[7:0]; // @[Compute.scala 302:37]
  assign _T_1136 = $unsigned(src_0_11); // @[Compute.scala 303:30]
  assign _T_1137 = $unsigned(src_1_11); // @[Compute.scala 303:59]
  assign _T_1138 = _T_1136 + _T_1137; // @[Compute.scala 303:49]
  assign _T_1139 = _T_1136 + _T_1137; // @[Compute.scala 303:49]
  assign _T_1140 = $signed(_T_1139); // @[Compute.scala 303:79]
  assign add_val_11 = _T_850 ? $signed(_T_1140) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_11 = _T_850 ? $signed(add_val_11) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1141 = add_res_11[7:0]; // @[Compute.scala 305:37]
  assign _T_1143 = src_1_11[4:0]; // @[Compute.scala 306:60]
  assign _T_1144 = _T_1136 >> _T_1143; // @[Compute.scala 306:49]
  assign _T_1145 = $signed(_T_1144); // @[Compute.scala 306:84]
  assign shr_val_11 = _T_850 ? $signed(_T_1145) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_11 = _T_850 ? $signed(shr_val_11) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1146 = shr_res_11[7:0]; // @[Compute.scala 308:37]
  assign src_0_12 = _T_850 ? $signed(_GEN_75) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_12 = _T_850 ? $signed(_GEN_95) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1147 = $signed(src_0_12) < $signed(src_1_12); // @[Compute.scala 300:34]
  assign _T_1148 = _T_1147 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 300:24]
  assign mix_val_12 = _T_850 ? $signed(_T_1148) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1149 = mix_val_12[7:0]; // @[Compute.scala 302:37]
  assign _T_1150 = $unsigned(src_0_12); // @[Compute.scala 303:30]
  assign _T_1151 = $unsigned(src_1_12); // @[Compute.scala 303:59]
  assign _T_1152 = _T_1150 + _T_1151; // @[Compute.scala 303:49]
  assign _T_1153 = _T_1150 + _T_1151; // @[Compute.scala 303:49]
  assign _T_1154 = $signed(_T_1153); // @[Compute.scala 303:79]
  assign add_val_12 = _T_850 ? $signed(_T_1154) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_12 = _T_850 ? $signed(add_val_12) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1155 = add_res_12[7:0]; // @[Compute.scala 305:37]
  assign _T_1157 = src_1_12[4:0]; // @[Compute.scala 306:60]
  assign _T_1158 = _T_1150 >> _T_1157; // @[Compute.scala 306:49]
  assign _T_1159 = $signed(_T_1158); // @[Compute.scala 306:84]
  assign shr_val_12 = _T_850 ? $signed(_T_1159) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_12 = _T_850 ? $signed(shr_val_12) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1160 = shr_res_12[7:0]; // @[Compute.scala 308:37]
  assign src_0_13 = _T_850 ? $signed(_GEN_77) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_13 = _T_850 ? $signed(_GEN_96) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1161 = $signed(src_0_13) < $signed(src_1_13); // @[Compute.scala 300:34]
  assign _T_1162 = _T_1161 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 300:24]
  assign mix_val_13 = _T_850 ? $signed(_T_1162) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1163 = mix_val_13[7:0]; // @[Compute.scala 302:37]
  assign _T_1164 = $unsigned(src_0_13); // @[Compute.scala 303:30]
  assign _T_1165 = $unsigned(src_1_13); // @[Compute.scala 303:59]
  assign _T_1166 = _T_1164 + _T_1165; // @[Compute.scala 303:49]
  assign _T_1167 = _T_1164 + _T_1165; // @[Compute.scala 303:49]
  assign _T_1168 = $signed(_T_1167); // @[Compute.scala 303:79]
  assign add_val_13 = _T_850 ? $signed(_T_1168) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_13 = _T_850 ? $signed(add_val_13) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1169 = add_res_13[7:0]; // @[Compute.scala 305:37]
  assign _T_1171 = src_1_13[4:0]; // @[Compute.scala 306:60]
  assign _T_1172 = _T_1164 >> _T_1171; // @[Compute.scala 306:49]
  assign _T_1173 = $signed(_T_1172); // @[Compute.scala 306:84]
  assign shr_val_13 = _T_850 ? $signed(_T_1173) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_13 = _T_850 ? $signed(shr_val_13) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1174 = shr_res_13[7:0]; // @[Compute.scala 308:37]
  assign src_0_14 = _T_850 ? $signed(_GEN_79) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_14 = _T_850 ? $signed(_GEN_97) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1175 = $signed(src_0_14) < $signed(src_1_14); // @[Compute.scala 300:34]
  assign _T_1176 = _T_1175 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 300:24]
  assign mix_val_14 = _T_850 ? $signed(_T_1176) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1177 = mix_val_14[7:0]; // @[Compute.scala 302:37]
  assign _T_1178 = $unsigned(src_0_14); // @[Compute.scala 303:30]
  assign _T_1179 = $unsigned(src_1_14); // @[Compute.scala 303:59]
  assign _T_1180 = _T_1178 + _T_1179; // @[Compute.scala 303:49]
  assign _T_1181 = _T_1178 + _T_1179; // @[Compute.scala 303:49]
  assign _T_1182 = $signed(_T_1181); // @[Compute.scala 303:79]
  assign add_val_14 = _T_850 ? $signed(_T_1182) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_14 = _T_850 ? $signed(add_val_14) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1183 = add_res_14[7:0]; // @[Compute.scala 305:37]
  assign _T_1185 = src_1_14[4:0]; // @[Compute.scala 306:60]
  assign _T_1186 = _T_1178 >> _T_1185; // @[Compute.scala 306:49]
  assign _T_1187 = $signed(_T_1186); // @[Compute.scala 306:84]
  assign shr_val_14 = _T_850 ? $signed(_T_1187) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_14 = _T_850 ? $signed(shr_val_14) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1188 = shr_res_14[7:0]; // @[Compute.scala 308:37]
  assign src_0_15 = _T_850 ? $signed(_GEN_81) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign src_1_15 = _T_850 ? $signed(_GEN_98) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1189 = $signed(src_0_15) < $signed(src_1_15); // @[Compute.scala 300:34]
  assign _T_1190 = _T_1189 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 300:24]
  assign mix_val_15 = _T_850 ? $signed(_T_1190) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1191 = mix_val_15[7:0]; // @[Compute.scala 302:37]
  assign _T_1192 = $unsigned(src_0_15); // @[Compute.scala 303:30]
  assign _T_1193 = $unsigned(src_1_15); // @[Compute.scala 303:59]
  assign _T_1194 = _T_1192 + _T_1193; // @[Compute.scala 303:49]
  assign _T_1195 = _T_1192 + _T_1193; // @[Compute.scala 303:49]
  assign _T_1196 = $signed(_T_1195); // @[Compute.scala 303:79]
  assign add_val_15 = _T_850 ? $signed(_T_1196) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign add_res_15 = _T_850 ? $signed(add_val_15) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1197 = add_res_15[7:0]; // @[Compute.scala 305:37]
  assign _T_1199 = src_1_15[4:0]; // @[Compute.scala 306:60]
  assign _T_1200 = _T_1192 >> _T_1199; // @[Compute.scala 306:49]
  assign _T_1201 = $signed(_T_1200); // @[Compute.scala 306:84]
  assign shr_val_15 = _T_850 ? $signed(_T_1201) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign shr_res_15 = _T_850 ? $signed(shr_val_15) : $signed(32'sh0); // @[Compute.scala 283:36]
  assign _T_1202 = shr_res_15[7:0]; // @[Compute.scala 308:37]
  assign short_cmp_res_0 = _T_850 ? _T_981 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_0 = _T_850 ? _T_987 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_0 = _T_850 ? _T_992 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_1 = _T_850 ? _T_995 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_1 = _T_850 ? _T_1001 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_1 = _T_850 ? _T_1006 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_2 = _T_850 ? _T_1009 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_2 = _T_850 ? _T_1015 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_2 = _T_850 ? _T_1020 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_3 = _T_850 ? _T_1023 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_3 = _T_850 ? _T_1029 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_3 = _T_850 ? _T_1034 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_4 = _T_850 ? _T_1037 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_4 = _T_850 ? _T_1043 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_4 = _T_850 ? _T_1048 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_5 = _T_850 ? _T_1051 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_5 = _T_850 ? _T_1057 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_5 = _T_850 ? _T_1062 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_6 = _T_850 ? _T_1065 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_6 = _T_850 ? _T_1071 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_6 = _T_850 ? _T_1076 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_7 = _T_850 ? _T_1079 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_7 = _T_850 ? _T_1085 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_7 = _T_850 ? _T_1090 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_8 = _T_850 ? _T_1093 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_8 = _T_850 ? _T_1099 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_8 = _T_850 ? _T_1104 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_9 = _T_850 ? _T_1107 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_9 = _T_850 ? _T_1113 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_9 = _T_850 ? _T_1118 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_10 = _T_850 ? _T_1121 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_10 = _T_850 ? _T_1127 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_10 = _T_850 ? _T_1132 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_11 = _T_850 ? _T_1135 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_11 = _T_850 ? _T_1141 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_11 = _T_850 ? _T_1146 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_12 = _T_850 ? _T_1149 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_12 = _T_850 ? _T_1155 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_12 = _T_850 ? _T_1160 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_13 = _T_850 ? _T_1163 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_13 = _T_850 ? _T_1169 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_13 = _T_850 ? _T_1174 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_14 = _T_850 ? _T_1177 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_14 = _T_850 ? _T_1183 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_14 = _T_850 ? _T_1188 : 8'h0; // @[Compute.scala 283:36]
  assign short_cmp_res_15 = _T_850 ? _T_1191 : 8'h0; // @[Compute.scala 283:36]
  assign short_add_res_15 = _T_850 ? _T_1197 : 8'h0; // @[Compute.scala 283:36]
  assign short_shr_res_15 = _T_850 ? _T_1202 : 8'h0; // @[Compute.scala 283:36]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Compute.scala 313:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Compute.scala 314:39]
  assign _T_1205 = out_cntr_wrap == 1'h0; // @[Compute.scala 315:37]
  assign _T_1206 = opcode_alu_en & _T_1205; // @[Compute.scala 315:34]
  assign _T_1207 = _T_1206 & busy; // @[Compute.scala 315:52]
  assign _T_1209 = 4'h9 - 4'h1; // @[Compute.scala 316:58]
  assign _T_1210 = $unsigned(_T_1209); // @[Compute.scala 316:58]
  assign _T_1211 = _T_1210[3:0]; // @[Compute.scala 316:58]
  assign _GEN_310 = {{12'd0}, _T_1211}; // @[Compute.scala 316:40]
  assign _T_1212 = dst_offset_in == _GEN_310; // @[Compute.scala 316:40]
  assign _T_1213 = out_mem_write & _T_1212; // @[Compute.scala 316:23]
  assign _T_1216 = _T_1213 & _T_313; // @[Compute.scala 316:66]
  assign _GEN_275 = _T_1216 ? 1'h0 : _T_1207; // @[Compute.scala 316:85]
  assign _T_1219 = dst_idx - 16'h1; // @[Compute.scala 319:34]
  assign _T_1220 = $unsigned(_T_1219); // @[Compute.scala 319:34]
  assign _T_1221 = _T_1220[15:0]; // @[Compute.scala 319:34]
  assign _GEN_311 = {{7'd0}, _T_1221}; // @[Compute.scala 319:41]
  assign _T_1223 = _GEN_311 << 3'h4; // @[Compute.scala 319:41]
  assign _T_1230 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1238 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1230}; // @[Cat.scala 30:58]
  assign _T_1245 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1253 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1245}; // @[Cat.scala 30:58]
  assign _T_1260 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1268 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1260}; // @[Cat.scala 30:58]
  assign _T_1269 = alu_opcode_add_en ? _T_1253 : _T_1268; // @[Compute.scala 323:8]
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 186:23]
  assign io_done_readdata = opcode == 3'h3; // @[Compute.scala 189:20]
  assign io_uops_address = uop_dram_addr[31:0]; // @[Compute.scala 197:19]
  assign io_uops_read = uops_read; // @[Compute.scala 196:16]
  assign io_uops_write = 1'h0;
  assign io_uops_writedata = 32'h0;
  assign io_biases_address = acc_dram_addr[31:0]; // @[Compute.scala 210:21]
  assign io_biases_read = biases_read; // @[Compute.scala 211:18]
  assign io_biases_write = 1'h0;
  assign io_biases_writedata = 128'h0;
  assign io_gemm_queue_ready = gemm_queue_ready; // @[Compute.scala 182:23]
  assign io_l2g_dep_queue_ready = pop_prev_dep_ready & dump; // @[Compute.scala 135:26]
  assign io_s2g_dep_queue_ready = pop_next_dep_ready & dump; // @[Compute.scala 136:26]
  assign io_g2l_dep_queue_valid = push_prev_dep & push; // @[Compute.scala 147:26]
  assign io_g2l_dep_queue_data = 1'h1; // @[Compute.scala 145:25]
  assign io_g2s_dep_queue_valid = push_next_dep & push; // @[Compute.scala 148:26]
  assign io_g2s_dep_queue_data = 1'h1; // @[Compute.scala 146:25]
  assign io_inp_mem_address = 15'h0;
  assign io_inp_mem_read = 1'h0;
  assign io_inp_mem_write = 1'h0;
  assign io_inp_mem_writedata = 64'h0;
  assign io_wgt_mem_address = 18'h0;
  assign io_wgt_mem_read = 1'h0;
  assign io_wgt_mem_write = 1'h0;
  assign io_wgt_mem_writedata = 64'h0;
  assign io_out_mem_address = _T_1223[16:0]; // @[Compute.scala 319:22]
  assign io_out_mem_read = 1'h0;
  assign io_out_mem_write = out_mem_write; // @[Compute.scala 320:20]
  assign io_out_mem_writedata = alu_opcode_minmax_en ? _T_1238 : _T_1269; // @[Compute.scala 322:24]
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE
  integer initvar;
  initial begin
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      #0.002 begin end
    `endif
  _RAND_0 = {16{`RANDOM}};
  `ifdef RANDOMIZE_MEM_INIT
  for (initvar = 0; initvar < 256; initvar = initvar+1)
    acc_mem[initvar] = _RAND_0[511:0];
  `endif // RANDOMIZE_MEM_INIT
  _RAND_1 = {1{`RANDOM}};
  `ifdef RANDOMIZE_MEM_INIT
  for (initvar = 0; initvar < 1024; initvar = initvar+1)
    uop_mem[initvar] = _RAND_1[31:0];
  `endif // RANDOMIZE_MEM_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {4{`RANDOM}};
  insn = _RAND_2[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{`RANDOM}};
  state = _RAND_3[2:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{`RANDOM}};
  uops_read = _RAND_4[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {1{`RANDOM}};
  uops_data = _RAND_5[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {1{`RANDOM}};
  biases_read = _RAND_6[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_7 = {4{`RANDOM}};
  biases_data_0 = _RAND_7[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_8 = {4{`RANDOM}};
  biases_data_1 = _RAND_8[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_9 = {4{`RANDOM}};
  biases_data_2 = _RAND_9[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_10 = {4{`RANDOM}};
  biases_data_3 = _RAND_10[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_11 = {1{`RANDOM}};
  out_mem_write = _RAND_11[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_12 = {1{`RANDOM}};
  uop_cntr_val = _RAND_12[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_13 = {1{`RANDOM}};
  acc_cntr_val = _RAND_13[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_14 = {1{`RANDOM}};
  dst_offset_in = _RAND_14[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_15 = {1{`RANDOM}};
  pop_prev_dep_ready = _RAND_15[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_16 = {1{`RANDOM}};
  pop_next_dep_ready = _RAND_16[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_17 = {1{`RANDOM}};
  push_prev_dep_ready = _RAND_17[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_18 = {1{`RANDOM}};
  push_next_dep_ready = _RAND_18[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_19 = {1{`RANDOM}};
  gemm_queue_ready = _RAND_19[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_20 = {16{`RANDOM}};
  dst_vector = _RAND_20[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_21 = {16{`RANDOM}};
  src_vector = _RAND_21[511:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(acc_mem__T_394_en & acc_mem__T_394_mask) begin
      acc_mem[acc_mem__T_394_addr] <= acc_mem__T_394_data; // @[Compute.scala 33:20]
    end
    if(uop_mem__T_352_en & uop_mem__T_352_mask) begin
      uop_mem[uop_mem__T_352_addr] <= uop_mem__T_352_data; // @[Compute.scala 34:20]
    end
    if (gemm_queue_ready) begin
      insn <= io_gemm_queue_data;
    end
    if (reset) begin
      state <= 3'h0;
    end else begin
      if (gemm_queue_ready) begin
        state <= 3'h2;
      end else begin
        if (_T_277) begin
          state <= 3'h4;
        end else begin
          if (_T_275) begin
            state <= 3'h2;
          end else begin
            if (_T_273) begin
              state <= 3'h1;
            end else begin
              if (_T_264) begin
                if (_T_265) begin
                  state <= 3'h3;
                end else begin
                  state <= 3'h4;
                end
              end
            end
          end
        end
      end
    end
    if (_T_296) begin
      if (_T_349) begin
        uops_read <= 1'h0;
      end else begin
        uops_read <= _T_341;
      end
    end else begin
      uops_read <= _T_341;
    end
    if (_T_296) begin
      uops_data <= io_uops_readdata;
    end
    biases_read <= acc_cntr_en & _T_380;
    if (_T_305) begin
      if (3'h0 == _T_386) begin
        biases_data_0 <= io_biases_readdata;
      end
    end
    if (_T_305) begin
      if (3'h1 == _T_386) begin
        biases_data_1 <= io_biases_readdata;
      end
    end
    if (_T_305) begin
      if (3'h2 == _T_386) begin
        biases_data_2 <= io_biases_readdata;
      end
    end
    if (_T_305) begin
      if (3'h3 == _T_386) begin
        biases_data_3 <= io_biases_readdata;
      end
    end
    if (reset) begin
      out_mem_write <= 1'h0;
    end else begin
      if (_T_1216) begin
        out_mem_write <= 1'h0;
      end else begin
        out_mem_write <= _T_1207;
      end
    end
    if (gemm_queue_ready) begin
      uop_cntr_val <= 16'h0;
    end else begin
      if (_T_299) begin
        uop_cntr_val <= _T_302;
      end
    end
    if (gemm_queue_ready) begin
      acc_cntr_val <= 16'h0;
    end else begin
      if (_T_308) begin
        acc_cntr_val <= _T_311;
      end
    end
    if (gemm_queue_ready) begin
      dst_offset_in <= 16'h0;
    end else begin
      if (_T_317) begin
        dst_offset_in <= _T_320;
      end
    end
    if (reset) begin
      pop_prev_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        pop_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_281) begin
          pop_prev_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      pop_next_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        pop_next_dep_ready <= 1'h0;
      end else begin
        if (_T_284) begin
          pop_next_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      push_prev_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        push_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_289) begin
          push_prev_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      push_next_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        push_next_dep_ready <= 1'h0;
      end else begin
        if (_T_292) begin
          push_next_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      gemm_queue_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        gemm_queue_ready <= 1'h0;
      end else begin
        gemm_queue_ready <= _T_329;
      end
    end
    if (_T_314) begin
      dst_vector <= acc_mem__T_416_data;
    end
    if (_T_314) begin
      src_vector <= acc_mem__T_418_data;
    end
  end
endmodule
