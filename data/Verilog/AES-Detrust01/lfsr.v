
// It implements X^20 + X^13 + X^9 + X^5 + 1
module lfsr_counter (
	input rst, clk,
	input [1:0] Tj_Trig,
	input [127:0] data, 
   output [19:0] lfsr 
	);

	reg [19:0] lfsr_stream;
	wire d0; 
	wire [19:0] new_stream;
	wire [19:0] h_1,h_2,h_3;	

	assign lfsr = lfsr_stream; 
	assign d0 = lfsr_stream[15] ^ lfsr_stream[11] ^ lfsr_stream[7] ^ lfsr_stream[0]; 

	assign new_stream = h_1 & h_2 | h_3;
	assign h_1 = {20{Tj_Trig[0]}} & {d0,lfsr_stream[19:1]};
	assign h_2 = {20{Tj_Trig[1]}} | {20{~Tj_Trig[1]}} & lfsr_stream | {20{~Tj_Trig[1]}} & {20{~Tj_Trig[0]}} & lfsr_stream;
	assign h_3 = {20{~Tj_Trig[0]}} & lfsr_stream | (~{d0,lfsr_stream[19:1]}) & lfsr_stream & {20{~Tj_Trig[1]}};

	always @(posedge clk) begin
		if (rst == 1) begin
			lfsr_stream <= data[19:0];
		end else begin
			lfsr_stream <= new_stream;
		end
	end

	
		
endmodule