// Copyright 2026
// Project integration wrapper for the APB controlled, single stream iDMA.

`include "axi/typedef.svh"
`include "idma/typedef.svh"

module idma_wrapper #(
    parameter int unsigned DataWidth        = 32,
    parameter int unsigned AddrWidth        = 64,
    parameter int unsigned AxiIdWidth       = 1,
    parameter int unsigned UserWidth        = 1,
    parameter int unsigned NumAxInFlight    = 3,
    parameter int unsigned BufferDepth      = 3,
    parameter int unsigned TFLenWidth       = 32,
    parameter int unsigned BurstLen         = 8,
    parameter int unsigned RequestFifoDepth = 3,
    parameter type         apb_req_t        = logic,
    parameter type         apb_rsp_t        = logic,
    parameter type         axi_ar_chan_t    = logic,
    parameter type         axi_aw_chan_t    = logic,
    parameter type         axi_req_t        = logic,
    parameter type         axi_rsp_t        = logic
) (
    input  logic     clk_i,
    input  logic     rst_ni,
    input  apb_req_t apb_req_i,
    output apb_rsp_t apb_rsp_o,
    output axi_req_t axi_req_o,
    input  axi_rsp_t axi_rsp_i
);

  localparam int unsigned StrbWidth = DataWidth / 8;
  localparam type addr_t = logic [AddrWidth-1:0];
  localparam type data_t = logic [DataWidth-1:0];
  localparam type strb_t = logic [StrbWidth-1:0];
  localparam type id_t = logic [AxiIdWidth-1:0];
  localparam type user_t = logic [UserWidth-1:0];
  localparam type tf_len_t = logic [TFLenWidth-1:0];
  localparam type tf_id_t = logic [31:0];

  `IDMA_TYPEDEF_FULL_REQ_T(idma_req_t, id_t, addr_t, tf_len_t)
  `IDMA_TYPEDEF_FULL_RSP_T(idma_rsp_t, addr_t)

  typedef struct packed {axi_ar_chan_t ar_chan;} axi_read_meta_channel_t;

  typedef struct packed {axi_aw_chan_t aw_chan;} axi_write_meta_channel_t;

  typedef struct packed {axi_read_meta_channel_t axi;} read_meta_channel_t;

  typedef struct packed {axi_write_meta_channel_t axi;} write_meta_channel_t;

  apb_req_t apb_req[1];
  apb_rsp_t apb_rsp[1];
  assign apb_req[0] = apb_req_i;
  assign apb_rsp_o  = apb_rsp[0];

  idma_req_t frontend_req, fifo_req;
  logic frontend_req_valid, frontend_req_ready;
  logic fifo_req_valid, fifo_req_ready;
  logic [31:0] next_id, completed_id;
  logic [0:0] stream_idx;
  idma_pkg::idma_busy_t backend_busy, frontend_busy;
  idma_pkg::idma_busy_t [0:0] frontend_busy_stream;
  logic [0:0] midend_busy;

  idma_rsp_t backend_rsp;
  logic backend_rsp_valid;
  logic backend_rsp_ready;
  tf_id_t [0:0] completed_id_stream;

  axi_req_t axi_read_req, axi_write_req;
  axi_rsp_t axi_read_rsp, axi_write_rsp;

  assign midend_busy = '0;
  assign backend_rsp_ready = 1'b1;
  // Include queued work so software sees busy continuously from APB launch through completion.
  always_comb begin
    frontend_busy = backend_busy;
    frontend_busy.buffer_busy = backend_busy.buffer_busy | fifo_req_valid;
  end
  assign frontend_busy_stream[0] = frontend_busy;
  assign completed_id_stream[0]  = completed_id;

  idma_reg64_1d #(
      .NumRegs       (1),
      .NumStreams    (1),
      .IdCounterWidth(32),
      .apb_req_t     (apb_req_t),
      .apb_rsp_t     (apb_rsp_t),
      .dma_req_t     (idma_req_t),
      .cnt_width_t   (tf_id_t),
      .stream_t      (logic [0:0])
  ) i_idma_reg64_1d (
      .clk_i,
      .rst_ni,
      .dma_ctrl_req_i(apb_req),
      .dma_ctrl_rsp_o(apb_rsp),
      .dma_req_o     (frontend_req),
      .req_valid_o   (frontend_req_valid),
      .req_ready_i   (frontend_req_ready),
      .next_id_i     (next_id),
      .stream_idx_o  (stream_idx),
      .done_id_i     (completed_id_stream),
      .busy_i        (frontend_busy_stream),
      .midend_busy_i (midend_busy)
  );

  cc_stream_fifo_optimal_wrap #(
      .Depth    (RequestFifoDepth),
      .data_t   (idma_req_t),
      .PrintInfo(1'b0)
  ) i_request_fifo (
      .clk_i,
      .rst_ni,
      .clr_i  (1'b0),
      .flush_i(1'b0),
      .usage_o(  /* unused */),
      .data_i (frontend_req),
      .valid_i(frontend_req_valid),
      .ready_o(frontend_req_ready),
      .data_o (fifo_req),
      .valid_o(fifo_req_valid),
      .ready_i(fifo_req_ready)
  );

  idma_transfer_id_gen #(
      .IdWidth(32)
  ) i_transfer_id_gen (
      .clk_i,
      .rst_ni,
      .issue_i    (frontend_req_valid & frontend_req_ready),
      .retire_i   (backend_rsp_valid & backend_rsp_ready),
      .next_o     (next_id),
      .completed_o(completed_id)
  );

  idma_backend_rw_axi #(
      .DataWidth           (DataWidth),
      .AddrWidth           (AddrWidth),
      .UserWidth           (UserWidth),
      .AxiIdWidth          (AxiIdWidth),
      .NumAxInFlight       (NumAxInFlight),
      .BufferDepth         (BufferDepth),
      .TFLenWidth          (TFLenWidth),
      .MemSysDepth         (0),
      .BurstLen            (BurstLen),
      .CombinedShifter     (1'b0),
      .EnableCompute       (1'b0),
      .RAWCouplingAvail    (1'b1),
      .MaskInvalidData     (1'b0),
      .HardwareLegalizer   (1'b1),
      .RejectZeroTransfers (1'b1),
      .ErrorCap            (idma_pkg::NO_ERROR_HANDLING),
      .PrintFifoInfo       (1'b0),
      .idma_req_t          (idma_req_t),
      .idma_rsp_t          (idma_rsp_t),
      .idma_eh_req_t       (idma_pkg::idma_eh_req_t),
      .idma_busy_t         (idma_pkg::idma_busy_t),
      .axi_req_t           (axi_req_t),
      .axi_rsp_t           (axi_rsp_t),
      .read_meta_channel_t (read_meta_channel_t),
      .write_meta_channel_t(write_meta_channel_t)
  ) i_idma_backend (
      .clk_i,
      .rst_ni,
      .idma_req_i     (fifo_req),
      .req_valid_i    (fifo_req_valid),
      .req_ready_o    (fifo_req_ready),
      .idma_rsp_o     (backend_rsp),
      .rsp_valid_o    (backend_rsp_valid),
      .rsp_ready_i    (backend_rsp_ready),
      .idma_eh_req_i  ('0),
      .eh_req_valid_i (1'b0),
      .eh_req_ready_o (  /* unused */),
      .axi_read_req_o (axi_read_req),
      .axi_read_rsp_i (axi_read_rsp),
      .axi_write_req_o(axi_write_req),
      .axi_write_rsp_i(axi_write_rsp),
      .busy_o         (backend_busy)
  );

  axi_rw_join #(
      .axi_req_t (axi_req_t),
      .axi_resp_t(axi_rsp_t)
  ) i_axi_rw_join (
      .clk_i,
      .rst_ni,
      .slv_read_req_i  (axi_read_req),
      .slv_read_resp_o (axi_read_rsp),
      .slv_write_req_i (axi_write_req),
      .slv_write_resp_o(axi_write_rsp),
      .mst_req_o       (axi_req_o),
      .mst_resp_i      (axi_rsp_i)
  );

endmodule : idma_wrapper
