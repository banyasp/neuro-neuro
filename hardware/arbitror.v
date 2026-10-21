module arbitrar(input stream_a, input stream_b, output class);
	// we need to design a hardware arbitrar that is able to take module classifications for multiple frames
	// a frame contains 32 spiked classification as either a zero or 1.
	// in this arbitrar we are going to use a sliding window to accumulate 32 values and then sum the number of classifications together
	// then compare the two accumulations via a > operator.
	// this is a simple hardware topology to compare outputs of the model.

	
