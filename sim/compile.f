# Run Questa from the repository root:
#   vlib work
#   vlog -sv -work work -f sim/compile.f
#
# This file compiles the bundled iDMA RTL and the project wrapper. The UVM
# environment and a wrapper smoke-test top have not been added yet.

+incdir+rtl/include
+incdir+src/include
+incdir+rtl/vendor/pulp/common_cells/include
+incdir+rtl/vendor/pulp/axi/include
+incdir+rtl/vendor/pulp/apb/include
+incdir+rtl/vendor/pulp/axi_stream/include
+incdir+rtl/vendor/pulp/obi/include

rtl/vendor/pulp/common_cells/src/cc_pkg.sv
rtl/vendor/pulp/axi/src/axi_pkg.sv
rtl/vendor/pulp/apb/src/apb_pkg.sv
rtl/vendor/pulp/obi/src/obi_pkg.sv
src/idma_pkg.sv
rtl/idma_generated.sv
rtl/vendor/pulp/common_cells/src/cc_stream_fifo_optimal_wrap.sv
rtl/vendor/pulp/axi/src/axi_rw_join.sv
src/frontend/idma_transfer_id_gen.sv
src/backend/idma_axi_read.sv
src/backend/idma_axi_write.sv
src/backend/idma_channel_coupler.sv
src/backend/idma_dataflow_element.sv
src/backend/idma_error_handler.sv
src/backend/idma_legalizer_page_splitter.sv
src/backend/idma_legalizer_pow2_splitter.sv
rtl/idma_wrapper.sv
