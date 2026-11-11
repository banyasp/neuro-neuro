// Single LIF neuron, parameterized on fan-in so the same module is used for every layer.
// Math follows ML/handoff/lif/spec.md exactly:
//   1. leak:      v = v - (v >>> leak_shift)
//   2. integrate: for each active input i (increasing i): v = sat(v + w[i])  (saturate after EVERY add)
//   3. fire:      spike = (v >= thresholdV)   (signed compare)
//   4. reset:     if spike, v = v - thresholdV
// Registered: on each posedge clk, v_j and spike update to the values for the current timestep.
// rst is active low and clears the membrane (use it at the start of every window).
module lif #(
	parameter int num_inputs = 46,
	parameter int leak_shift = 5,
	parameter int bit_width  = 16,                        // membrane width (int16)
	parameter int w_width    = 8,                         // weight width (int8)
	parameter signed [bit_width-1:0] thresholdV = 64,
	parameter signed [bit_width-1:0] restV      = 0       // value v_j resets to on rst
) (
	input  wire clk,
	input  wire rst,
	input  wire [num_inputs-1:0] input_spikes,
	input  wire signed [(num_inputs*w_width)-1:0] w_ij_packed,  // weight i at [i*w_width +: w_width]
	output reg  spike
);

	// membrane potential
	reg  signed [bit_width-1:0] v_j;
	wire signed [bit_width-1:0] v_j_prev;   // after leak

	// integration stage variables (one extra bit so an add can't wrap before we saturate)
	reg  signed [bit_width:0]   u_j_accum;
	reg  signed [bit_width-1:0] u_j_sat;

	// next value
	wire signed [bit_width-1:0] v_j_next;

	// the leak equation
	assign v_j_prev = v_j - (v_j >>> leak_shift);

	// MAX and MIN of a bit_width signed value, expressed in bit_width+1 bits
	localparam signed [bit_width:0] MAX_VAL = {2'b00, {(bit_width-1){1'b1}}};
	localparam signed [bit_width:0] MIN_VAL = {2'b11, {(bit_width-1){1'b0}}};

	// integrate step
	integer i;
	always @(*) begin
		u_j_accum = {v_j_prev[bit_width-1], v_j_prev};

		for (i = 0; i < num_inputs; i = i + 1) begin
			if (input_spikes[i] == 1'b1) begin
				u_j_accum = u_j_accum + $signed(w_ij_packed[i*w_width +: w_width]);
				if (u_j_accum > MAX_VAL) begin
					u_j_accum = MAX_VAL;
				end else if (u_j_accum < MIN_VAL) begin
					u_j_accum = MIN_VAL;
				end
			end
		end
		u_j_sat = u_j_accum[bit_width-1:0];
	end

	// FIRE AND RESET STEPS
	assign v_j_next = (u_j_sat >= thresholdV) ? (u_j_sat - thresholdV) : u_j_sat;

	always @(posedge clk or negedge rst) begin
		if (!rst) begin
			v_j   <= restV;
			spike <= 1'b0;
		end else begin
			v_j   <= v_j_next;
			spike <= (u_j_sat >= thresholdV);
		end
	end
endmodule
