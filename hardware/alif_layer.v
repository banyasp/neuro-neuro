// One fully connected layer of ALIF neurons: N_IN input lines -> N_OUT neurons.
// Same as lif_layer.v but instantiates alif.v (adaptive threshold).
//
// Weights come in on one flat bus in the SAME order as the ML .mem files
// (see ML/handoff/alif_chb01/spec.md):  W[i][j] (input i -> neuron j) lives at
//     w_flat[(i*N_OUT + j)*W_WIDTH +: W_WIDTH]
// This module regroups them per neuron for alif.v.
//
// Latency: 1 clock. Spikes for the input sampled at edge e appear after edge e.
module alif_layer #(
	parameter int N_IN        = 46,
	parameter int N_OUT       = 64,
	parameter int LEAK_SHIFT  = 5,    // K
	parameter int ADAPT_SHIFT = 7,    // K_A
	parameter int A_INC       = 16,   // adaptation step
	parameter int V_WIDTH     = 16,
	parameter int W_WIDTH     = 8,
	parameter signed [V_WIDTH-1:0] THETA = 64
) (
	input  wire                          clk,
	input  wire                          rst,        // active low, clears membranes + adaptation
	input  wire [N_IN-1:0]               in_spikes,
	input  wire [N_IN*N_OUT*W_WIDTH-1:0] w_flat,
	output wire [N_OUT-1:0]              out_spikes
);

	genvar j, i;
	generate
		for (j = 0; j < N_OUT; j = j + 1) begin : N
			// neuron j's weights, input i at [i*W_WIDTH +: W_WIDTH] (what alif.v expects)
			wire [N_IN*W_WIDTH-1:0] w_j;
			for (i = 0; i < N_IN; i = i + 1) begin : W
				assign w_j[i*W_WIDTH +: W_WIDTH] = w_flat[(i*N_OUT + j)*W_WIDTH +: W_WIDTH];
			end

			alif #(
				.num_inputs      (N_IN),
				.leak_shift      (LEAK_SHIFT),
				.leak_adapt_shift(ADAPT_SHIFT),
				.bit_width       (V_WIDTH),
				.w_width         (W_WIDTH),
				.thresholdV      (THETA),
				.adaptation      (A_INC)
			) u (
				.clk         (clk),
				.rst         (rst),
				.input_spikes(in_spikes),
				.w_ij_packed (w_j),
				.spike       (out_spikes[j])
			);
		end
	endgenerate

endmodule
