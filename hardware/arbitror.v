module arbitrar(input stream_a, input stream_b, output class);
	// we need to design a hardware arbitrar that is able to take module classifications for multiple frames
	// a frame contains 32 spiked classification as either a zero or 1.
	// in this arbitrar we are going to use a sliding window to accumulate 32 values and then sum the number of classifications together
	// then compare the two accumulations via a > operator.
	// this is a simple hardware topology to compare outputs of the model.

	wire [31:0] model_A_out;
	wire [31:0] model_B_out;

	// accumulation for model A
        .sliding_window(.A(stream_A), .clk(clk), .rst(rst), .total_count(model_A_out));	
	
	// accumulation for model B
	.sliding_window(.A(stream_B), .clk(clk), .rst(rst), .total_count(model_B_out));

	// decision maker here 
