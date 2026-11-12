// Single ALIF (Adaptive Leaky-Integrate-Fire) neuron, extended from lif.v
// Math follows ML/handoff/alif/spec.md exactly:
//   1. leak:      v = v - (v >>> leak_shift)
//NEW2. leak adaptation: bonus_thresholdV = bonus_thresholdV - (bonus_thresholdV >>> leak_adapt_shift)
//   3. integrate: for each active input i (increasing i): v = sat(v + w[i])  (saturate after EVERY add)
//   4. fire:      spike = (v >= thresholdV + bonus_thresholdV)   (signed compare)
//   5. reset:     if spike, v = v - thresholdV             # OPTION FOR FUTURE CONSIDERATION: ALSO - bonus_thresholdV
//NEW6. adapt:     if spike, bonus_thresholdV = bonus_thresholdV + adaptation
// Registered: on each posedge clk, v_j and spike update to the values for the current timestep.
// rst is active low and clears the membrane (use it at the start of every window).
module alif #(
	parameter int num_inputs = 46,
	parameter int leak_shift = 5,
    parameter int leak_adapt_shift = 7,
	parameter int bit_width  = 16,                        // membrane width (int16)
	parameter int w_width    = 8,                         // weight width (int8)
	parameter signed [bit_width-1:0] thresholdV = 64,
    parameter int adaptation = 16,
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

    // bonus threshold
    reg  signed [bit_width-1:0] bonus_thresholdV;
    wire signed [bit_width-1:0] bonus_thresholdV_prev; // after leak

	// integration stage variables (one extra bit so an add can't wrap before we saturate)
	reg  signed [bit_width:0]   u_j_accum;
	reg  signed [bit_width-1:0] u_j_sat;

	// next value
	wire signed [bit_width-1:0] v_j_next;
    wire signed [bit_width-1:0] bonus_thresholdV_next;

	// the leak equation
	assign v_j_prev = v_j - (v_j >>> leak_shift);
    assign bonus_thresholdV_prev = bonus_thresholdV - (bonus_thresholdV >>> leak_adapt_shift);

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
	assign v_j_next = (u_j_sat >= thresholdV + bonus_thresholdV_prev) ? (u_j_sat - thresholdV) : u_j_sat;
    assign bonus_thresholdV_next = (u_j_sat >= thresholdV + bonus_thresholdV_prev) ? (bonus_thresholdV_prev + adaptation) : bonus_thresholdV_prev;

	always @(posedge clk or negedge rst) begin
		if (!rst) begin
			v_j   <= restV;
            bonus_thresholdV <= 1'b0;
			spike <= 1'b0;
		end else begin
			v_j   <= v_j_next;
            bonus_thresholdV <= bonus_thresholdV_next;
			spike <= (u_j_sat >= thresholdV + bonus_thresholdV_prev);
		end
	end
endmodule
