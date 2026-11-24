`timescale 1ns/1ps
// Shared-kernel SCNN built from the team's lif_layer and lif modules.
// One accepted input frame runs through all three layers before the next is accepted.
module scnn_lif #(
    parameter integer N_CH = 23,
    parameter integer K = 16,
    parameter integer FILTERS_A = 4,
    parameter integer FILTERS_B = 4,
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
    input  wire clk, rst_n, sample_valid,
    output wire sample_ready,
    input  wire [2*N_CH-1:0] sample_spikes,
    output reg step_valid,
    output reg [1:0] out_spikes,
    output reg window_valid,
    output reg [$clog2(WINDOW_LEN+1)-1:0] count0, count1,
    output reg window_decision
);
    localparam integer C = FILTERS_A + FILTERS_B;
    localparam integer N1 = N_CH*C;
    localparam integer N2 = C*DEPTH_MULT;
    localparam integer MAX_D = (DILATION_A > DILATION_B) ? DILATION_A : DILATION_B;
    localparam integer HISTORY = (K-1)*MAX_D + 1;
    localparam integer INDEX_BITS = WINDOW_LEN > 1 ? $clog2(WINDOW_LEN) : 1;
    localparam [2:0] IDLE=0, TEMPORAL=1, SPATIAL=2, OUTPUT_LAYER=3, REPORT=4;

    reg [2:0] state;
    reg [INDEX_BITS-1:0] sample_index;
    reg [2*N_CH-1:0] history [0:HISTORY-1];
    assign sample_ready = rst_n && state == IDLE;
    wire accept = sample_valid && sample_ready;
    wire first_frame = accept && sample_index == 0;

    reg signed [7:0] w1 [0:2*K*C-1];
    reg signed [7:0] w2 [0:N_CH*N2-1];
    reg signed [7:0] w3 [0:N2*2-1];
    initial begin
        $readmemh(W1_FILE, w1);
        $readmemh(W2_FILE, w2);
        $readmemh(W3_FILE, w3);
    end

    wire [N1-1:0] s1;
    wire [N2-1:0] s2;
    wire [1:0] s3;
    genvar g, row, f, ch, k, j, i;
    generate
        // Each group has a different dilation. All channels reuse its weight bank.
        for (g=0; g<2; g=g+1) begin : temporal_group
            localparam integer NF = (g == 0) ? FILTERS_A : FILTERS_B;
            localparam integer FIRST = (g == 0) ? 0 : FILTERS_A;
            localparam integer DIL = (g == 0) ? DILATION_A : DILATION_B;
            if (NF > 0) begin : active
                wire [2*K*NF*8-1:0] weights;
                for (row=0; row<2*K; row=row+1) begin : weight_row
                    for (f=0; f<NF; f=f+1) begin : filter_weight
                        assign weights[(row*NF+f)*8 +: 8] = w1[row*C+FIRST+f];
                    end
                end
                for (ch=0; ch<N_CH; ch=ch+1) begin : channel
                    wire [2*K-1:0] taps;
                    for (k=0; k<K; k=k+1) begin : tap
                        // Bits 2*k and 2*k+1 receive UP then DOWN of the same channel.
                        assign taps[2*k +: 2] = history[k*DIL][2*ch +: 2];
                    end
                    lif_layer #(
                        .N_IN(2*K), .N_OUT(NF), .LEAK_SHIFT(LEAK_SHIFT),
                        .THETA(THETA), .USE_STEP_CONTROL(1)
                    ) neurons (
                        .clk(clk), .rst(rst_n), .step_en(state == TEMPORAL),
                        .window_clear(first_frame), .in_spikes(taps), .w_flat(weights),
                        .out_spikes(s1[ch*C+FIRST +: NF])
                    );
                end
            end
        end

        // Bank f sees only filter f's temporal response across the EEG channels.
        for (f=0; f<C; f=f+1) begin : spatial_filter
            wire [N_CH-1:0] inputs;
            wire [N_CH*DEPTH_MULT*8-1:0] weights;
            for (ch=0; ch<N_CH; ch=ch+1) begin : channel
                assign inputs[ch] = s1[ch*C+f];
                for (j=0; j<DEPTH_MULT; j=j+1) begin : unit_weight
                    assign weights[(ch*DEPTH_MULT+j)*8 +: 8] = w2[ch*N2+f*DEPTH_MULT+j];
                end
            end
            lif_layer #(
                .N_IN(N_CH), .N_OUT(DEPTH_MULT), .LEAK_SHIFT(LEAK_SHIFT),
                .THETA(THETA), .USE_STEP_CONTROL(1)
            ) neurons (
                .clk(clk), .rst(rst_n), .step_en(state == SPATIAL),
                .window_clear(first_frame), .in_spikes(inputs), .w_flat(weights),
                .out_spikes(s2[f*DEPTH_MULT +: DEPTH_MULT])
            );
        end
    endgenerate

    wire [N2*2*8-1:0] out_weights;
    generate
        for (i=0; i<N2*2; i=i+1) begin : output_weight
            assign out_weights[i*8 +: 8] = w3[i];
        end
    endgenerate
    lif_layer #(
        .N_IN(N2), .N_OUT(2), .LEAK_SHIFT(LEAK_SHIFT),
        .THETA(THETA), .USE_STEP_CONTROL(1)
    ) output_neurons (
        .clk(clk), .rst(rst_n), .step_en(state == OUTPUT_LAYER),
        .window_clear(first_frame), .in_spikes(s2), .w_flat(out_weights),
        .out_spikes(s3)
    );

    // The output neurons register s3 on OUTPUT_LAYER. REPORT consumes it once.
    wire [$clog2(WINDOW_LEN+1)-1:0] next_count0 = count0 + s3[0];
    wire [$clog2(WINDOW_LEN+1)-1:0] next_count1 = count1 + s3[1];
    integer h;
    always @(posedge clk) begin
        if (accept) begin
            history[0] <= sample_spikes;
            for (h=1; h<HISTORY; h=h+1)
                history[h] <= first_frame ? 0 : history[h-1];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            sample_index <= 0;
            out_spikes <= 0;
            count0 <= 0;
            count1 <= 0;
            step_valid <= 0;
            window_valid <= 0;
            window_decision <= 0;
        end else begin
            step_valid <= 0;
            window_valid <= 0;
            case (state)
                IDLE: if (accept) begin
                    if (first_frame) begin
                        count0 <= 0;
                        count1 <= 0;
                    end
                    state <= TEMPORAL;
                end
                TEMPORAL: state <= SPATIAL;
                SPATIAL: state <= OUTPUT_LAYER;
                OUTPUT_LAYER: state <= REPORT;
                REPORT: begin
                    out_spikes <= s3;
                    count0 <= next_count0;
                    count1 <= next_count1;
                    step_valid <= 1;
                    if (sample_index == WINDOW_LEN-1) begin
                        window_valid <= 1;
                        window_decision <= next_count1 > next_count0;
                        sample_index <= 0;
                    end else sample_index <= sample_index + 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
