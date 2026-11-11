`timescale 1ns/1ns

// Testbench for lif.v against the ML golden vectors in ML/handoff/lif/ (see spec.md there).
//
// Builds the 46 -> 64 -> 2 network out of lif instances:
//   layer 1: 64 neurons, 46 inputs each, driven by input_spikes.mem
//   layer 2:  2 neurons, 64 inputs each, driven by the GOLDEN layer-1 spikes (expected_spk_l1.mem)
// Layer 2 is fed the golden spikes on purpose: lif registers its spike output, so chaining the
// RTL layers directly would put layer 2 one timestep behind the reference (spec: layer 2 uses
// layer 1's spikes from the SAME timestep). This way each layer is checked on its own.
//
// Every timestep checks: all 64 l1 membranes + spikes, both l2 membranes + spikes.
// At the end of each window: output spike counts and the seizure decision.
//
// Run from the hardware/ directory:
//   iverilog -g2012 -o lif_tb.vvp lif.v lif_tb.v && vvp lif_tb.vvp
// Override the vector location with +HANDOFF=<path ending in />.

module lif_tb();

	localparam int N_IN  = 46;
	localparam int N_H   = 64;
	localparam int N_OUT = 2;
	localparam int T     = 128;
	localparam int K     = 5;
	localparam int THETA = 64;
	localparam int VW    = 16;   // membrane bits
	localparam int WW    = 8;    // weight bits
	localparam int MAX_PRINT = 10;  // mismatches printed per case

	// ---------------- golden data ----------------
	reg        [WW-1:0]    w1         [0:N_IN*N_H-1];
	reg        [WW-1:0]    w2         [0:N_H*N_OUT-1];
	reg        [N_IN-1:0]  in_spk     [0:T-1];
	reg        [N_H-1:0]   exp_spk1   [0:T-1];
	reg        [VW-1:0]    exp_mem1   [0:T*N_H-1];
	reg        [VW-1:0]    exp_mem2   [0:T*N_OUT-1];
	reg        [N_OUT-1:0] exp_out    [0:T-1];
	reg        [15:0]      exp_counts [0:N_OUT-1];
	reg        [0:0]       exp_dec    [0:0];

	// ---------------- DUT signals ----------------
	reg clk = 0;
	reg rst = 0;   // active low
	always #5 clk = ~clk;

	reg  [N_IN-1:0]         l1_in;
	reg  [N_H-1:0]          l2_in;
	reg  [N_IN*WW-1:0]      w1_bus [0:N_H-1];    // packed weights per l1 neuron
	reg  [N_H*WW-1:0]       w2_bus [0:N_OUT-1];  // packed weights per l2 neuron

	wire [N_H-1:0]          l1_spk;
	wire [N_OUT-1:0]        l2_spk;
	wire signed [VW-1:0]    l1_mem [0:N_H-1];
	wire signed [VW-1:0]    l2_mem [0:N_OUT-1];

	genvar g;
	generate
		for (g = 0; g < N_H; g = g + 1) begin : L1
			lif #(.num_inputs(N_IN), .leak_shift(K), .bit_width(VW), .w_width(WW), .thresholdV(THETA))
				u (.clk(clk), .rst(rst), .input_spikes(l1_in), .w_ij_packed(w1_bus[g]), .spike(l1_spk[g]));
			assign l1_mem[g] = u.v_j;
		end
		for (g = 0; g < N_OUT; g = g + 1) begin : L2
			lif #(.num_inputs(N_H), .leak_shift(K), .bit_width(VW), .w_width(WW), .thresholdV(THETA))
				u (.clk(clk), .rst(rst), .input_spikes(l2_in), .w_ij_packed(w2_bus[g]), .spike(l2_spk[g]));
			assign l2_mem[g] = u.v_j;
		end
	endgenerate

	// ---------------- test ----------------
	integer total_errors = 0;
	integer cases_failed = 0;

	task automatic run_case(input string dir);
		integer t, i, j, errs;
		integer counts [0:N_OUT-1];
		reg dec;
		begin
			errs = 0;

			$readmemh({dir, "weights_l1.mem"}, w1);
			$readmemh({dir, "weights_l2.mem"}, w2);
			$readmemh({dir, "input_spikes.mem"}, in_spk);
			$readmemh({dir, "expected_spk_l1.mem"}, exp_spk1);
			$readmemh({dir, "expected_mem_l1.mem"}, exp_mem1);
			$readmemh({dir, "expected_mem_l2.mem"}, exp_mem2);
			$readmemh({dir, "expected_out.mem"}, exp_out);
			$readmemh({dir, "expected_counts.mem"}, exp_counts);
			$readmemh({dir, "expected_decision.mem"}, exp_dec);

			// pack weights: neuron j's input i lives at bits [i*WW +: WW]
			for (j = 0; j < N_H; j = j + 1)
				for (i = 0; i < N_IN; i = i + 1)
					w1_bus[j][i*WW +: WW] = w1[i*N_H + j];
			for (j = 0; j < N_OUT; j = j + 1)
				for (i = 0; i < N_H; i = i + 1)
					w2_bus[j][i*WW +: WW] = w2[i*N_OUT + j];

			for (j = 0; j < N_OUT; j = j + 1) counts[j] = 0;

			// reset membranes at the start of the window
			@(negedge clk);
			rst   = 0;
			l1_in = '0;
			l2_in = '0;
			@(negedge clk);
			rst = 1;

			for (t = 0; t < T; t = t + 1) begin
				// drive timestep t away from the clock edge, check right after the edge
				l1_in = in_spk[t];
				l2_in = exp_spk1[t];
				@(posedge clk);
				#1;

				for (j = 0; j < N_H; j = j + 1) begin
					if (l1_mem[j] !== $signed(exp_mem1[t*N_H + j]) || l1_spk[j] !== exp_spk1[t][j]) begin
						if (errs < MAX_PRINT)
							$display("  MISMATCH t=%0d L1[%0d]: mem=%0d spk=%b  expected mem=%0d spk=%b",
							         t, j, l1_mem[j], l1_spk[j], $signed(exp_mem1[t*N_H + j]), exp_spk1[t][j]);
						errs = errs + 1;
					end
				end
				for (j = 0; j < N_OUT; j = j + 1) begin
					if (l2_mem[j] !== $signed(exp_mem2[t*N_OUT + j]) || l2_spk[j] !== exp_out[t][j]) begin
						if (errs < MAX_PRINT)
							$display("  MISMATCH t=%0d L2[%0d]: mem=%0d spk=%b  expected mem=%0d spk=%b",
							         t, j, l2_mem[j], l2_spk[j], $signed(exp_mem2[t*N_OUT + j]), exp_out[t][j]);
						errs = errs + 1;
					end
					counts[j] = counts[j] + l2_spk[j];
				end
				@(negedge clk);
			end

			// window-level results
			for (j = 0; j < N_OUT; j = j + 1) begin
				if (counts[j] != exp_counts[j]) begin
					$display("  MISMATCH count[%0d]: got %0d expected %0d", j, counts[j], exp_counts[j]);
					errs = errs + 1;
				end
			end
			dec = (counts[1] > counts[0]);
			if (dec !== exp_dec[0]) begin
				$display("  MISMATCH decision: got %b expected %b", dec, exp_dec[0]);
				errs = errs + 1;
			end

			if (errs > MAX_PRINT) $display("  ... %0d mismatches total", errs);
			$display("%s  %s  (counts %0d/%0d, decision %0d)", errs == 0 ? "PASS" : "FAIL",
			         dir, counts[0], counts[1], dec);
			total_errors = total_errors + errs;
			if (errs != 0) cases_failed = cases_failed + 1;
		end
	endtask

	string base;
	initial begin
		if (!$value$plusargs("HANDOFF=%s", base)) base = "../ML/handoff/lif/";

		run_case(base);
		run_case({base, "saturation/"});
		run_case({base, "dense/"});
		run_case({base, "seizure/"});

		if (total_errors == 0) $display("ALL 4 CASES PASSED");
		else                   $display("%0d of 4 cases FAILED (%0d mismatches)", cases_failed, total_errors);
		$finish;
	end

endmodule
