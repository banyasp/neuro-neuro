module lif #(parameter int num_inputs = 32, paramter int leak_shift, paramter int bit_width = 8, parameter [bit_width - 1 : 0] thresholdV =  120, parameter [bit_width - 1: 0] restV = 0)
	    (input wire clk, input wire rst, input wire [NUM_INPUTS-1:0] input_spikes, input wire signed [(num_inputs * bit_wdith)-1:0] w_ij_packed,  output reg spike);

	// we are designing a likf module that is modular in size, meaning that we can create a lif with input channels being variable sizes, this allows us to use on module for multiple layers 
	//membrane potential equations 
	reg signed [bit_wdith-1:0] v_j;
	wire signed [bit_width-1:0] v_j_prev;
	
	//integration stage variables
	reg signed [bit_width:0] u_j_accum;
	reg signed [bit_width-1:0] u_j_sat;
	
	//next value  
	wire signed [bit_width - 0: 0] v_j_next;
	
	
	// the leak equation bottom half 
	assign v_j_prev = v_j - (v_j >>> leak_shift);
	
	// MAX and MIN
	localparam signed [BIT_WIDTH:0] MAX_VAL = {2'b00, {(BIT_WIDTH-1){1'b1}}};
	localparam signed [BIT_WIDTH:0] MIN_VAL = {2'b11, {(BIT_WIDTH-1){1'b0}}};

	// integrate step 
	integer i;
	always @(*) begin
		u_j_accum = {v_j_prev_decayed[15], v_j_prev_decayed};
		
		for(i = 0; i < num_inputs; i = i + 1)begin 
			if(input_spike == 1'b1) begin 
				u_j_accum = u_j_accum + $signed(w_ij_packed[i*bit_width +: bit_width]);
				if(u_j_accum > MAX_VAL) begin
					u_j_accum = MAX_VAL
				end else if (u_j_accum < MIN_VAL) begin
					u_j_accum = MIN_VAL
				end
			end
		end
		u_j_saturated = u_j_accum[15:0];
	end

	// FIRE AND RESET STEPS
	assign v_j_next = (u_j_sat >= thresholdV) ? (u_j_sat - thresholdV) : u_j_sat;
   	
	always @(posedge clk or negedge rst) begin
		if(!rst) begin
			v_j <= {BIT_WIDTH{1'b0}};
			o_j <= 1'b0;
		end else begin
			v_j <= v_j_next;
            		o_j <= (u_j_sat >= thresholdV);
		end
	end 	
endmodule
