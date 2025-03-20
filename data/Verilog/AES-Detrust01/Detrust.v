module Detrust(input [31:0] state, input clk, output reg t, output reg j);
reg [31:0] state_prev;
reg [15:0] Tj_Trig;
reg [3:0] Tj_Trig_t;

always @(posedge clk) begin
	state_prev <= state;
end

always @(posedge clk) begin
	if(state[3:0] == 4'h0) Tj_Trig[0] <= 1; 
	else Tj_Trig[0] <= 0;
end
always @(posedge clk) begin
	if(state[7:4] == 4'h0) Tj_Trig[1] <= 1; 
	else Tj_Trig[1] <= 0;
end
always @(posedge clk) begin
	if(state[11:8] == 4'h1) Tj_Trig[2] <= 1; 
	else Tj_Trig[2] <= 0;
end
always @(posedge clk) begin
	if(state[15:12] == 4'h1) Tj_Trig[3] <= 1; 
	else Tj_Trig[3] <= 0;
end
always @(posedge clk) begin
	if(state[19:16] == 4'h2) Tj_Trig[4] <= 1; 
	else Tj_Trig[4] <= 0;
end
always @(posedge clk) begin
	if(state[23:20] == 4'h2) Tj_Trig[5] <= 1; 
	else Tj_Trig[5] <= 0;
end
always @(posedge clk) begin
	if(state[27:24] == 4'h3) Tj_Trig[6] <= 1; 
	else Tj_Trig[6] <= 0;
end
always @(posedge clk) begin
	if(state[31:28] == 4'h3) Tj_Trig[7] <= 1; 
	else Tj_Trig[7] <= 0;
end

always @(posedge clk) begin
	if(Tj_Trig[3:0] == 4'hf) Tj_Trig_t[0] <= 1;
	else Tj_Trig_t[0] <= 0;
end
always @(posedge clk) begin
	if(Tj_Trig[7:4] == 4'hf) Tj_Trig_t[1] <= 1;
	else Tj_Trig_t[1] <= 0;
end

always @(posedge clk) begin
	if(Tj_Trig_t[1:0] == 2'h3) t <= 1;
	else t <= 0;
end


always @(posedge clk) begin
	if(state_prev[3:0] == 4'h8) Tj_Trig[8] <= 1; 
	else Tj_Trig[8] <= 0;
end
always @(posedge clk) begin
	if(state_prev[7:4] == 4'h8) Tj_Trig[9] <= 1; 
	else Tj_Trig[9] <= 0;
end
always @(posedge clk) begin
	if(state_prev[11:8] == 4'h9) Tj_Trig[10] <= 1; 
	else Tj_Trig[10] <= 0;
end
always @(posedge clk) begin
	if(state_prev[15:12] == 4'h9) Tj_Trig[11] <= 1; 
	else Tj_Trig[11] <= 0;
end
always @(posedge clk) begin
	if(state_prev[19:16] == 4'ha) Tj_Trig[12] <= 1; 
	else Tj_Trig[12] <= 0;
end
always @(posedge clk) begin
	if(state_prev[23:20] == 4'ha) Tj_Trig[13] <= 1; 
	else Tj_Trig[13] <= 0;
end
always @(posedge clk) begin
	if(state_prev[27:24] == 4'hb) Tj_Trig[14] <= 1; 
	else Tj_Trig[14] <= 0;
end
always @(posedge clk) begin
	if(state_prev[31:28] == 4'hb) Tj_Trig[15] <= 1; 
	else Tj_Trig[15] <= 0;
end

always @(posedge clk) begin
	if(Tj_Trig[11:8] == 4'hf) Tj_Trig_t[2] <= 1;
	else Tj_Trig_t[2] <= 0;
end
always @(posedge clk) begin
	if(Tj_Trig[15:12] == 4'hf) Tj_Trig_t[3] <= 1;
	else Tj_Trig_t[3] <= 0;
end

always @(posedge clk) begin
	if(Tj_Trig_t[3:2] == 2'h3) j <= 1;
	else j <= 0;
end
endmodule