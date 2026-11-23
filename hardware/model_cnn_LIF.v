// LIF SNN: N_IN -> N_H -> N_OUT, built from lif_layer.v (which is built from lif.v).
// All layer sizes are parameters: change N_IN / N_H / N_OUT and the wiring, weight
// buses and generate loops all resize automatically.
//
// To add another hidden layer: add a size parameter, a weight memory + flat bus,
// one more lif_layer instance in the chain, and bump N_LAYERS.
//
// Weights are loaded from .mem files (ML/handoff/lif format, address i*N_out + j).
// A testbench can reload them at runtime with $readmemh(file, dut.w1_mem).
//
// Timing: one timestep per clock, layers are pipelined (each lif registers its spike).
//   Layer 2 at clock t+1 integrates layer 1's spikes from timestep t, which is exactly
//   the spec's "same timestep" rule, just shifted one clock. The extra first-cycle step
//   layer 2 takes after reset sees v=0 and no spikes, so it is a no-op (needs restV=0).
//   => out_spikes are N_LAYERS clocks behind in_spikes; out_valid marks them.
//
// Constraint: lif has no enable, so every neuron leaks every clock. Feed the T timesteps
// of a window on consecutive clocks (in_valid held high), let the pipeline drain
// N_LAYERS clocks, then pulse rst before the next window.
module model_cnn_lif #(
	parameter int N_IN       = 46,
	parameter int N_H        = 64,
	parameter int N_OUT      = 2,
	parameter int LEAK_SHIFT = 5,
	parameter int V_WIDTH    = 16,
	parameter int W_WIDTH    = 8,
	parameter signed [V_WIDTH-1:0] THETA = 64,
	parameter W1_FILE = "weights_l1.mem",
	parameter W2_FILE = "weights_l2.mem"
) (
	input  wire             clk,
	input  wire             rst,          // active low
	input  wire             in_valid,
	input  wire [N_IN-1:0]  in_spikes,    // bit i = input line i
	output wire             out_valid,
	output wire [N_OUT-1:0] out_spikes,   // bit j = output neuron j
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
	lif_layer #(
		.N_IN(N_IN), .N_OUT(N_H),
		.LEAK_SHIFT(LEAK_SHIFT), .V_WIDTH(V_WIDTH), .W_WIDTH(W_WIDTH), .THETA(THETA)
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
		else      valid_pipe <= {valid_pipe[N_LAYERS-2:0], in_valid};
	end
	assign out_valid = valid_pipe[N_LAYERS-1];

endmodule
