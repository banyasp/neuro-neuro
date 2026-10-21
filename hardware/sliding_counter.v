module sliding_window(input A, input clk, input rst, output reg [15:0] total_count);
	// in this module we are designing a sliding window that will add up the total number of classifications in a window
	// this includes a FIFO queue (1-bit shift register), and a adder that will sum up all the values of the queue (for a specific frame)
	
	reg [15:0] shift_reg;
	wire spike_out;
	
	assign spike_out = shift_reg[15];

	always @(posedge clk or negedge rst) begin 
		if(!rst) begin
			shift_reg <= {16{1'b0}};
			total_count <= {16{1'b0}};
		end else begin 
			// this is where the actual shifting begins (taking input A)
			shift_reg <= {shift_reg{14:0}, A};
			
			// case for the accumulator
			case({})
	
