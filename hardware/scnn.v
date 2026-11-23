`timescale 1ns/1ps
// Baseline shared-temporal-filter, depthwise-spatial SCNN from ML/handoff/scnn.
// One FPGA clock. Each active layer visits one input row per clock for all its neurons.
module scnn #(
    parameter integer N_CH = 23,
    parameter integer K = 16,
    parameter integer FILTERS_A = 4,  // first group
    parameter integer FILTERS_B = 4,  // second group
    parameter integer DILATION_A = 1,
    parameter integer DILATION_B = 3,
    parameter integer DEPTH_MULT = 2,
    parameter integer WINDOW_LEN = 128,
    parameter integer LEAK_SHIFT = 5,
    parameter signed [15:0] THETA = 16'sd64,
    parameter W1_FILE = "weights_l1.mem",
    parameter W2_FILE = "weights_l2.mem",
    parameter W3_FILE = "weights_l3.mem"
) (
    input wire clk,
    input wire rst_n,
    input wire sample_valid,
    output wire sample_ready,
    input wire [2*N_CH-1:0] sample_spikes,
    output reg step_valid,
    output reg [1:0] out_spikes,    // [0] nonseizure, [1] seizure
    output reg window_valid,
    output reg [$clog2(WINDOW_LEN+1)-1:0] count0, count1,
    output reg window_decision
);
    localparam integer C = FILTERS_A + FILTERS_B;
    localparam integer N1 = N_CH*C;
    localparam integer N2 = C*DEPTH_MULT;
    localparam integer MAX_D = (DILATION_A > DILATION_B) ? DILATION_A : DILATION_B;
    localparam integer HISTORY = (K-1)*MAX_D + 1;
    localparam integer MAX_12 = (2*K > N_CH) ? 2*K : N_CH;
    localparam integer MAX_ROWS = (MAX_12 > N2) ? MAX_12 : N2;
    localparam integer ROW_BITS = (MAX_ROWS > 1) ? $clog2(MAX_ROWS) : 1;
    localparam integer STEP_BITS = (WINDOW_LEN > 1) ? $clog2(WINDOW_LEN) : 1;
    localparam [2:0] IDLE=0, T_ADD=1, T_FIRE=2, S_ADD=3, S_FIRE=4, O_ADD=5, O_FIRE=6;

    reg signed [7:0] w1 [0:2*K*C-1];
    reg signed [7:0] w2 [0:N_CH*N2-1];
    reg signed [7:0] w3 [0:N2*2-1];
    initial begin
        $readmemh(W1_FILE, w1);
        $readmemh(W2_FILE, w2);
        $readmemh(W3_FILE, w3);
    end

    reg [2:0] state;
    reg [ROW_BITS-1:0] row;
    reg [STEP_BITS-1:0] sample_index;
    reg [2*N_CH-1:0] history [0:HISTORY-1]; // entry 0 is the accepted frame
    reg signed [15:0] v1 [0:N1-1];
    reg signed [15:0] v2 [0:N2-1];
    reg signed [15:0] v3 [0:1];
    reg [N1-1:0] s1;
    reg [N2-1:0] s2;

    assign sample_ready = rst_n && (state == IDLE);
    // In T_ADD row = 2*tap + polarity. All channels share these two frames.
    wire [ROW_BITS-1:0] tap = (state == T_ADD) ? (row >> 1) : 0;
    wire [2*N_CH-1:0] frame_a = history[tap*DILATION_A];
    wire [2*N_CH-1:0] frame_b = history[tap*DILATION_B];

    function signed [15:0] leak;
        input signed [15:0] v;
        begin
            leak = v - (v >>> LEAK_SHIFT);
        end
    endfunction

    function signed [15:0] sat_add;
        input signed [15:0] v;
        input signed [7:0] w;
        reg signed [16:0] sum;
        begin
            sum = v + w;
            if (sum > 17'sd32767)       sat_add = 16'sh7fff;
            else if (sum < -17'sd32768) sat_add = 16'sh8000;
            else                       sat_add = sum[15:0];
        end
    endfunction

    // These signals see the final O_ADD result before O_FIRE commits it.
    wire [1:0] fire_out;
    assign fire_out[0] = (v3[0] >= THETA);
    assign fire_out[1] = (v3[1] >= THETA);
    wire [$clog2(WINDOW_LEN+1)-1:0] next_count0 = count0 + fire_out[0];
    wire [$clog2(WINDOW_LEN+1)-1:0] next_count1 = count1 + fire_out[1];

    integer ch, f, u, o, n, h;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            row <= 0;
            sample_index <= 0;
            count0 <= 0;
            count1 <= 0;
            out_spikes <= 0;
            step_valid <= 0;
            window_valid <= 0;
            window_decision <= 0;
            // The first accepted sample initializes every history word and membrane.
        end else begin
            step_valid <= 0;
            window_valid <= 0;
            case (state)
                IDLE: if (sample_valid) begin
                    history[0] <= sample_spikes;
                    for (h=1; h<HISTORY; h=h+1)
                        history[h] <= (sample_index == 0) ? 0 : history[h-1];
                    // All layers leak their own previous state once, before any inputs arrive.
                    for (n=0; n<N1; n=n+1)
                        v1[n] <= (sample_index == 0) ? 0 : leak(v1[n]);
                    for (u=0; u<N2; u=u+1)
                        v2[u] <= (sample_index == 0) ? 0 : leak(v2[u]);
                    for (o=0; o<2; o=o+1)
                        v3[o] <= (sample_index == 0) ? 0 : leak(v3[o]);
                    if (sample_index == 0) begin
                        count0 <= 0;
                        count1 <= 0;
                    end
                    row <= 0;
                    state <= T_ADD;
                end
                T_ADD: begin
                    for (ch=0; ch<N_CH; ch=ch+1) begin
                        for (f=0; f<C; f=f+1) begin
                            if ((f < FILTERS_A) ? frame_a[2*ch+row[0]] : frame_b[2*ch+row[0]])
                                v1[ch*C+f] <= sat_add(v1[ch*C+f], w1[row*C+f]);
                        end
                    end
                    if (row == 2*K-1) begin
                        row <= 0;
                        state <= T_FIRE;
                    end else row <= row + 1'b1;
                end
                T_FIRE: begin
                    for (n=0; n<N1; n=n+1) begin
                        s1[n] <= (v1[n] >= THETA);
                        if (v1[n] >= THETA) v1[n] <= v1[n] - THETA;
                    end
                    state <= S_ADD;
                end
                S_ADD: begin
                    for (u=0; u<N2; u=u+1) begin
                        // Unit u belongs to filter u/DEPTH_MULT; row is the channel.
                        if (s1[row*C+(u/DEPTH_MULT)])
                            v2[u] <= sat_add(v2[u], w2[row*N2+u]);
                    end
                    if (row == N_CH-1) begin
                        row <= 0;
                        state <= S_FIRE;
                    end else row <= row + 1'b1;
                end
                S_FIRE: begin
                    for (u=0; u<N2; u=u+1) begin
                        s2[u] <= (v2[u] >= THETA);
                        if (v2[u] >= THETA) v2[u] <= v2[u] - THETA;
                    end
                    state <= O_ADD;
                end
                O_ADD: begin
                    if (s2[row]) begin
                        for (o=0; o<2; o=o+1)
                            v3[o] <= sat_add(v3[o], w3[row*2+o]);
                    end
                    if (row == N2-1) begin
                        row <= 0;
                        state <= O_FIRE;
                    end else row <= row + 1'b1;
                end
                O_FIRE: begin
                    for (o=0; o<2; o=o+1) begin
                        if (fire_out[o]) v3[o] <= v3[o] - THETA;
                    end
                    out_spikes <= fire_out;
                    step_valid <= 1;
                    count0 <= next_count0;
                    count1 <= next_count1;
                    if (sample_index == WINDOW_LEN-1) begin
                        window_decision <= (next_count1 > next_count0);
                        window_valid <= 1;
                        sample_index <= 0;
                    end else sample_index <= sample_index + 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
