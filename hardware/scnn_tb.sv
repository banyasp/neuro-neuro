`timescale 1ns/1ps
module scnn_tb;
    localparam integer T=128, N1=184, N2=16;
    localparam integer TIMEOUT = 100;
    reg clk=0, rst_n=1, sample_valid=0;
    reg [45:0] sample_spikes=0;
    wire sample_ready, step_valid, window_valid, window_decision;
    wire [1:0] out_spikes;
    wire [7:0] count0, count1;
    wire [15:0] actual_v1 [0:N1-1], actual_v2 [0:N2-1], actual_v3 [0:1];
    genvar ch, f, u, o, n;
`ifdef SCNN_SERIAL
    localparam integer EXPECTED_CYCLES = 74, ABORT_AFTER = 40;
    scnn dut (.*);
    for (n=0; n<N1; n=n+1) assign actual_v1[n] = dut.v1[n];
    for (u=0; u<N2; u=u+1) assign actual_v2[u] = dut.v2[u];
    for (o=0; o<2; o=o+1) assign actual_v3[o] = dut.v3[o];
`else
    localparam integer EXPECTED_CYCLES = 4, ABORT_AFTER = 1;
    scnn_lif dut (.*);
    // Exercise the existing named-port API with both new inputs deliberately omitted.
    wire legacy_spike;
    lif_layer #(.N_IN(1), .N_OUT(1)) legacy (
        .clk(clk), .rst(rst_n), .in_spikes(1'b1), .w_flat(8'd40), .out_spikes(legacy_spike)
    );
    // Simulation-only probes into Arnav's neuron instances, in the ML trace order.
    for (ch=0; ch<23; ch=ch+1) begin : probe_channel
        for (f=0; f<8; f=f+1) begin : probe_filter
            assign actual_v1[ch*8+f] =
                dut.temporal_group[f/4].active.channel[ch].neurons.N[f%4].u.v_j;
        end
    end
    for (u=0; u<N2; u=u+1) begin : probe_spatial
        assign actual_v2[u] = dut.spatial_filter[u/2].neurons.N[u%2].u.v_j;
    end
    for (o=0; o<2; o=o+1) begin : probe_output
        assign actual_v3[o] = dut.output_neurons.N[o].u.v_j;
    end
`endif
    always #5 clk = ~clk;

    reg [45:0] inputs [0:T-1];
    reg [N1-1:0] spk1 [0:T-1];
    reg [N2-1:0] spk2 [0:T-1];
    reg [1:0] outs [0:T-1];
    reg [15:0] mem1 [0:T*N1-1], mem2 [0:T*N2-1], mem3 [0:T*2-1];
    reg [15:0] counts [0:1];
    reg [0:0] decision [0:0];
    integer t, pass, running0, running1;

    task check_traces(input integer k);
        integer n;
        begin
            if (dut.s1 !== spk1[k] || dut.s2 !== spk2[k] || out_spikes !== outs[k])
                $fatal(1,"spikes t=%0d L1=%h/%h L2=%h/%h out=%h/%h (actual/expected)",
                       k, dut.s1, spk1[k], dut.s2, spk2[k], out_spikes, outs[k]);
            for (n=0; n<N1; n=n+1)
                if (actual_v1[n] !== mem1[k*N1+n])
                    $fatal(1,"L1 t=%0d neuron=%0d got=%h expected=%h",k,n,actual_v1[n],mem1[k*N1+n]);
            for (n=0; n<N2; n=n+1)
                if (actual_v2[n] !== mem2[k*N2+n])
                    $fatal(1,"L2 t=%0d neuron=%0d got=%h expected=%h",k,n,actual_v2[n],mem2[k*N2+n]);
            for (n=0; n<2; n=n+1)
                if (actual_v3[n] !== mem3[k*2+n])
                    $fatal(1,"L3 t=%0d neuron=%0d got=%h expected=%h",k,n,actual_v3[n],mem3[k*2+n]);
            running0 = running0 + outs[k][0];
            running1 = running1 + outs[k][1];
            if (count0 !== running0[7:0] || count1 !== running1[7:0])
                $fatal(1,"running counts t=%0d got=%0d,%0d expected=%0d,%0d",k,count0,count1,running0,running1);
            if (window_valid !== (k == T-1))
                $fatal(1,"window_valid t=%0d",k);
            if (k == T-1 && (count0 !== counts[0] || count1 !== counts[1] || window_decision !== decision[0]))
                $fatal(1,"window result got=%0d,%0d decision=%b",count0,count1,window_decision);
        end
    endtask

    task check_quiet(input bit expect_ready);
        begin
            if (sample_ready !== expect_ready || step_valid !== 1'b0 || window_valid !== 1'b0)
                $fatal(1,"handshake at %0t: ready=%b step_valid=%b window_valid=%b",$time,sample_ready,step_valid,window_valid);
        end
    endtask

    task accept_sample(input integer k);
        begin
            @(negedge clk);
            if (sample_ready !== 1'b1) $fatal(1,"input unexpectedly busy t=%0d",k);
            sample_spikes = inputs[k];
            sample_valid = 1;
            @(posedge clk); #1;
            check_quiet(0);
        end
    endtask

    task run_step(input integer k, input bit continuous_input);
        integer cycle;
        begin
            accept_sample(k);
            cycle = 0;
            while (step_valid !== 1'b1 && cycle < TIMEOUT) begin
                cycle = cycle + 1;
                @(negedge clk);
                sample_valid = continuous_input;
                // A continuously valid producer holds its next frame. With valid low, vary all bits.
                sample_spikes = continuous_input ? inputs[(k+1)%T] : ((cycle%2) ? ~inputs[k] : 46'd0);
                @(posedge clk); #1;
                if (step_valid !== 1'b1) check_quiet(0);
            end
            if (step_valid !== 1'b1 || sample_ready !== 1'b1)
                $fatal(1,"completion timing t=%0d",k);
            check_traces(k);
            if (cycle != EXPECTED_CYCLES)
                $fatal(1,"latency t=%0d got=%0d expected=%0d",k,cycle,EXPECTED_CYCLES);
        end
    endtask

    task reset_dut;
        begin
            @(negedge clk);
            rst_n = 0;
            sample_valid = 0;
            #1;
            check_quiet(0);
            if ({count0,count1,out_spikes,window_decision} !== 0) $fatal(1,"reset outputs");
            @(negedge clk);
            rst_n = 1;
        end
    endtask

    initial begin
        $readmemh("input_spikes.mem", inputs);
        $readmemh("expected_mem_l1.mem", mem1);
        $readmemh("expected_spk_l1.mem", spk1);
        $readmemh("expected_mem_l2.mem", mem2);
        $readmemh("expected_spk_l2.mem", spk2);
        $readmemh("expected_mem_l3.mem", mem3);
        $readmemh("expected_out.mem", outs);
        $readmemh("expected_counts.mem", counts);
        $readmemh("expected_decision.mem", decision);
        reset_dut();
`ifndef SCNN_SERIAL
        @(posedge clk); #1;
        if (legacy_spike !== 1'b0 || legacy.N[0].u.v_j !== 16'sd40)
            $fatal(1,"legacy lif_layer failed on first clock");
        @(posedge clk); #1;
        if (legacy_spike !== 1'b1 || legacy.N[0].u.v_j !== 16'sd15)
            $fatal(1,"legacy lif_layer failed on second clock");
`endif
        // Abandon a partial window while the spatial layer is running.
        running0=0; running1=0;
        run_step(0,0);
        accept_sample(1);
        repeat(ABORT_AFTER) @(posedge clk);
        reset_dut();
        for (pass=0; pass<2; pass=pass+1) begin
            running0=0; running1=0;
            for (t=0; t<T; t=t+1) begin
                if (pass == 0) repeat(t%4) begin
                    @(negedge clk); sample_valid=0; sample_spikes=~inputs[t];
                    @(posedge clk); #1; check_quiet(1);
                end
                run_step(t, pass == 1);
            end
        end
        @(negedge clk); sample_valid=0;
        @(posedge clk); #1; check_quiet(1);
        $display("SCNN_RTL_PASS counts=%0d,%0d decision=%b",count0,count1,window_decision);
        $finish;
    end
endmodule
