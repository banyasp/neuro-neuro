// ALIF SNN (Model B): N_IN -> N_H -> N_OUT, per ML/handoff/alif_chb01/spec.md.
//   Layer 1 (hidden): ALIF neurons (alif_layer.v -> alif.v), adaptive threshold.
//   Layer 2 (output): plain LIF neurons (lif_layer.v -> lif.v), exactly Model A's math.
// All layer sizes are parameters; wiring, weight buses and generate loops resize automatically.
//
// Weights are loaded from .mem files (ML/handoff format, address i*N_out + j).
// A testbench can reload them at runtime with $readmemh(file, dut.w1_mem).
//
// Timing: one timestep per clock, layers pipelined (each neuron registers its spike).
//   Layer 2 at clock t+1 integrates layer 1's spikes from timestep t (the spec's
//   "same timestep" rule, shifted one clock). Layer 2's extra first step after reset sees
//   v=0 and no spikes, so it is a no-op.
//   => out_spikes are N_LAYERS clocks behind in_spikes; out_valid marks them.
//
// Constraint: neurons have no enable, so they leak every clock. Feed a window's T timesteps
// on consecutive clocks, let the pipeline drain N_LAYERS clocks, then pulse rst
// (clears membranes AND adaptation registers) before the next window.
module model_snn_alif #(
	parameter int N_IN        = 46,
	parameter int N_H         = 64,
	parameter int N_OUT       = 2,
	parameter int LEAK_SHIFT  = 5,    // K, both layers
	parameter int ADAPT_SHIFT = 7,    // K_A, hidden layer only
	parameter int A_INC       = 16,   // adaptation step, hidden layer only
	parameter int V_WIDTH     = 16,
	parameter int W_WIDTH     = 8,
	parameter signed [V_WIDTH-1:0] THETA = 64,
	parameter W1_FILE = "weights_l1.mem",
	parameter W2_FILE = "weights_l2.mem"
) (
	input  wire             clk,
	input  wire             rst,          // active low
	input  wire             in_valid,
	input  wire [N_IN-1:0]  in_spikes,    // bit i = input line i
	output wire             out_valid,
	output wire [N_OUT-1:0] out_spikes,   // bit 0 = non-seizure, bit 1 = seizure
	output wire [N_H-1:0]   hidden_spikes // layer-1 spikes (debug)
);

	localparam int N_LAYERS = 2;

	// ---------------- weights ----------------
	reg [W_WIDTH-1:0] w1_mem [0:N_IN*N_H-1];
	reg [W_WIDTH-1:0] w2_mem [0:N_H*N_OUT-1];
	initial begin
		$readmemh(W1_FILE, w1_mem);
		$readmemh(W2_FILE, w2_mem);
	end

	wire [N_IN*N_H*W_WIDTH-1:0]  w1_flat;
	wire [N_H*N_OUT*W_WIDTH-1:0] w2_flat;

	genvar a;
	generate
		for (a = 0; a < N_IN*N_H; a = a + 1) begin : W1
			assign w1_flat[a*W_WIDTH +: W_WIDTH] = w1_mem[a];
		end
		for (a = 0; a < N_H*N_OUT; a = a + 1) begin : W2
			assign w2_flat[a*W_WIDTH +: W_WIDTH] = w2_mem[a];
		end
	endgenerate

	// ---------------- layers ----------------
	alif_layer #(
		.N_IN(N_IN), .N_OUT(N_H),
		.LEAK_SHIFT(LEAK_SHIFT), .ADAPT_SHIFT(ADAPT_SHIFT), .A_INC(A_INC),
		.V_WIDTH(V_WIDTH), .W_WIDTH(W_WIDTH), .THETA(THETA)
	) L1 (
		.clk(clk), .rst(rst),
		.in_spikes(in_spikes), .w_flat(w1_flat), .out_spikes(hidden_spikes)
	);

	lif_layer #(
		.N_IN(N_H), .N_OUT(N_OUT),
		.LEAK_SHIFT(LEAK_SHIFT), .V_WIDTH(V_WIDTH), .W_WIDTH(W_WIDTH), .THETA(THETA)
	) L2 (
		.clk(clk), .rst(rst),
		.in_spikes(hidden_spikes), .w_flat(w2_flat), .out_spikes(out_spikes)
	);

	// ---------------- valid pipeline ----------------
	reg [N_LAYERS-1:0] valid_pipe;
	always @(posedge clk or negedge rst) begin
		if (!rst) begin
			valid_pipe <= '0;
		end else begin
			valid_pipe <= {valid_pipe[N_LAYERS-2:0], in_valid};
		end
	end
	assign out_valid = valid_pipe[N_LAYERS-1];

endmodule
