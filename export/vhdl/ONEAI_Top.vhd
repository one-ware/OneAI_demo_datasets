
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.numeric_std.all;
use work.CNN_Config_Package.all;
use work.CNN_Data_Package.all;

ENTITY CNN IS
    PORT (
        iStream_1     : IN CNN_Stream_T;
        iData_1       : IN CNN_Values_T(0 downto 0);

        oStream_1     : OUT CNN_Stream_T;
        oData_1       : OUT CNN_Values_T(0 downto 0);
        oCycle_1      : OUT NATURAL
    );
END CNN;

ARCHITECTURE BEHAVIORAL OF CNN IS

    SIGNAL oStream_Conv_1 : CNN_Stream_T;
    SIGNAL oData_Conv_1 : CNN_Values_T(0 downto 0);
    SIGNAL oStream_Pool_1 : CNN_Stream_T;
    SIGNAL oData_Pool_1 : CNN_Values_T(0 downto 0);
    SIGNAL oStream_Conv_2 : CNN_Stream_T;
    SIGNAL oData_Conv_2 : CNN_Values_T(0 downto 0);
    SIGNAL oStream_Conv_3 : CNN_Stream_T;
    SIGNAL oData_Conv_3 : CNN_Values_T(0 downto 0);
    SIGNAL oStream_Conv_4 : CNN_Stream_T;
    SIGNAL oData_Conv_4 : CNN_Values_T(0 downto 0);
    SIGNAL oStream_Conv_5 : CNN_Stream_T;
    SIGNAL oData_Conv_5 : CNN_Values_T(0 downto 0);
    SIGNAL oStream_Conv_6 : CNN_Stream_T;
    SIGNAL oData_Conv_6 : CNN_Values_T(0 downto 0);
    SIGNAL oStream_Flat_1 : CNN_Stream_T;
    SIGNAL oData_Flat_1 : CNN_Values_T(0 downto 0);
    SIGNAL oCycle_Flat_1 : NATURAL range 0 to (Flatten_1_Columns * Flatten_1_Rows * Flatten_1_Values) - 1;
    SIGNAL oStream_FC_1 : CNN_Stream_T;
    SIGNAL oData_FC_1 : CNN_Values_T(0 downto 0);
    SIGNAL oCycle_FC_1 : NATURAL range 0 to NN_Layer_1_Outputs - 1;


    COMPONENT CNN_Convolution IS
        GENERIC (
            Input_Columns  : NATURAL := 28;
            Input_Rows     : NATURAL := 28;
            Input_Values   : NATURAL := 1;
            Filter_Columns : NATURAL := 3;
            Filter_Rows    : NATURAL := 3;
            Filters        : NATURAL := 4;
            Strides        : NATURAL := 1;
            Activation     : Activation_T := relu;
            Padding        : Padding_T := valid;
            Input_Cycles   : NATURAL := 1;
            Value_Cycles   : NATURAL := 1;
            Calc_Cycles    : NATURAL := 1;
            Filter_Cycles  : NATURAL := 1;
            Filter_Delay   : NATURAL := 1;
            Expand         : BOOLEAN := true;
            Expand_Cycles  : NATURAL := 0;
            Offset_In       : NATURAL := 0;
            Offset_Out      : NATURAL := 0;
            Offset         : INTEGER := 0;
            Weights        : CNN_Weights_T
        );
        PORT (
            iStream : IN  CNN_Stream_T;
            iData   : IN  CNN_Values_T(Input_Values/Input_Cycles-1 downto 0);
            
            oStream : OUT CNN_Stream_T;
            oData   : OUT CNN_Values_T(Filters/Filter_Cycles-1 downto 0) := (others => 0)
        );
    END COMPONENT;

    COMPONENT CNN_Pooling_Efficient IS
        GENERIC (
            Input_Columns  : NATURAL := 28; --Size in x direction of input
            Input_Rows     : NATURAL := 28; --Size in y direction of input
            Input_Values   : NATURAL := 1;  --Number of Filters in previous layer or 3 for RGB input
            Filter_Columns : NATURAL := 3;  --Size in x direction of filters
            Filter_Rows    : NATURAL := 3;  --Size in y direction of filters
            Input_Cycles   : NATURAL := 1;  --Filter Cycles of previous convolution
            Filter_Delay   : NATURAL := 1   --Cycles between Filters
        );
        PORT (
            iStream : IN  CNN_Stream_T;
            iData   : IN  CNN_Values_T(Input_Values/Input_Cycles-1 downto 0);
            
            oStream : OUT CNN_Stream_T;
            oData   : OUT CNN_Values_T(Input_Values/Input_Cycles-1 downto 0) := (others => 0)
        );
    END COMPONENT;

    COMPONENT NN_Layer IS
        GENERIC (
            Inputs          : NATURAL := 16;
            Outputs         : NATURAL := 8;
            Activation      : Activation_T := relu;
            Input_Cycles    : NATURAL := 1;
            Calc_Cycles     : NATURAL := 1;
            Output_Cycles   : NATURAL := 1;
            Output_Delay    : NATURAL := 1;
            Offset_In       : NATURAL := 0;
            Offset_Out      : NATURAL := 0;
            Offset          : INTEGER := 0;
            Weights         : CNN_Weights_T
        );
        PORT (
            iStream : IN  CNN_Stream_T;
            iData   : IN  CNN_Values_T(Inputs/Input_Cycles-1 downto 0);
            iCycle  : IN  NATURAL range 0 to Input_Cycles-1;
            
            oStream : OUT CNN_Stream_T;
            oData   : OUT CNN_Values_T(Outputs/Output_Cycles-1 downto 0) := (others => 0);
            oCycle  : OUT NATURAL range 0 to Output_Cycles-1
        );
    END COMPONENT;
    
BEGIN

    CNN_Convolution_1 : CNN_Convolution
    GENERIC MAP (
        Input_Columns  => Layer_1_Columns,
        Input_Rows     => Layer_1_Rows,
        Input_Values   => Layer_1_Values,
        Filter_Columns => Layer_1_Filter_X,
        Filter_Rows    => Layer_1_Filter_Y,
        Filters        => Layer_1_Filters,
        Strides        => Layer_1_Strides,
        Activation     => Layer_1_Activation,
        Padding        => Layer_1_Padding,
        Input_Cycles   => 1,
        Value_Cycles   => 1,
        Calc_Cycles    => 4,
        Filter_Cycles  => 8,
        Expand_Cycles  => 38,
        Filter_Delay   => 1,
        Offset_In      => 0,
        Offset_Out     => Layer_1_Out_Offset, 
        Offset         => Layer_1_Offset,
        Weights        => Layer_1
    ) PORT MAP (
        iStream        => iStream_1,
        iData          => iData_1,
        oStream        => oStream_Conv_1,
        oData          => oData_Conv_1
    );

    CNN_Pooling_1 : CNN_Pooling_Efficient
    GENERIC MAP (
        Input_Columns  => Pooling_1_Columns,
        Input_Rows     => Pooling_1_Rows,
        Input_Values   => Pooling_1_Values,
        Filter_Columns => Pooling_1_Filter_X,
        Filter_Rows    => Pooling_1_Filter_Y,
        Input_Cycles   => 8,
        Filter_Delay   => 1
    ) PORT MAP (
        iStream        => oStream_Conv_1,
        iData          => oData_Conv_1,
        oStream        => oStream_Pool_1,
        oData          => oData_Pool_1
    );

    CNN_Convolution_2 : CNN_Convolution
    GENERIC MAP (
        Input_Columns  => Layer_2_Columns,
        Input_Rows     => Layer_2_Rows,
        Input_Values   => Layer_2_Values,
        Filter_Columns => Layer_2_Filter_X,
        Filter_Rows    => Layer_2_Filter_Y,
        Filters        => Layer_2_Filters,
        Strides        => Layer_2_Strides,
        Activation     => Layer_2_Activation,
        Padding        => Layer_2_Padding,
        Input_Cycles   => 8,
        Value_Cycles   => 4,
        Calc_Cycles    => 4,
        Filter_Cycles  => 12,
        Expand_Cycles  => 146,
        Filter_Delay   => 1,
        Offset_In      => Layer_1_Out_Offset,
        Offset_Out     => Layer_2_Out_Offset, 
        Offset         => Layer_2_Offset,
        Weights        => Layer_2
    ) PORT MAP (
        iStream        => oStream_Pool_1,
        iData          => oData_Pool_1,
        oStream        => oStream_Conv_2,
        oData          => oData_Conv_2
    );

    CNN_Convolution_3 : CNN_Convolution
    GENERIC MAP (
        Input_Columns  => Layer_3_Columns,
        Input_Rows     => Layer_3_Rows,
        Input_Values   => Layer_3_Values,
        Filter_Columns => Layer_3_Filter_X,
        Filter_Rows    => Layer_3_Filter_Y,
        Filters        => Layer_3_Filters,
        Strides        => Layer_3_Strides,
        Activation     => Layer_3_Activation,
        Padding        => Layer_3_Padding,
        Input_Cycles   => 12,
        Value_Cycles   => 4,
        Calc_Cycles    => 16,
        Filter_Cycles  => 16,
        Expand_Cycles  => 578,
        Filter_Delay   => 1,
        Offset_In      => Layer_2_Out_Offset,
        Offset_Out     => Layer_3_Out_Offset, 
        Offset         => Layer_3_Offset,
        Weights        => Layer_3
    ) PORT MAP (
        iStream        => oStream_Conv_2,
        iData          => oData_Conv_2,
        oStream        => oStream_Conv_3,
        oData          => oData_Conv_3
    );

    CNN_Convolution_4 : CNN_Convolution
    GENERIC MAP (
        Input_Columns  => Layer_4_Columns,
        Input_Rows     => Layer_4_Rows,
        Input_Values   => Layer_4_Values,
        Filter_Columns => Layer_4_Filter_X,
        Filter_Rows    => Layer_4_Filter_Y,
        Filters        => Layer_4_Filters,
        Strides        => Layer_4_Strides,
        Activation     => Layer_4_Activation,
        Padding        => Layer_4_Padding,
        Input_Cycles   => 16,
        Value_Cycles   => 8,
        Calc_Cycles    => 24,
        Filter_Cycles  => 24,
        Expand_Cycles  => 1730,
        Filter_Delay   => 1,
        Offset_In      => Layer_3_Out_Offset,
        Offset_Out     => Layer_4_Out_Offset, 
        Offset         => Layer_4_Offset,
        Weights        => Layer_4
    ) PORT MAP (
        iStream        => oStream_Conv_3,
        iData          => oData_Conv_3,
        oStream        => oStream_Conv_4,
        oData          => oData_Conv_4
    );

    CNN_Convolution_5 : CNN_Convolution
    GENERIC MAP (
        Input_Columns  => Layer_5_Columns,
        Input_Rows     => Layer_5_Rows,
        Input_Values   => Layer_5_Values,
        Filter_Columns => Layer_5_Filter_X,
        Filter_Rows    => Layer_5_Filter_Y,
        Filters        => Layer_5_Filters,
        Strides        => Layer_5_Strides,
        Activation     => Layer_5_Activation,
        Padding        => Layer_5_Padding,
        Input_Cycles   => 24,
        Value_Cycles   => 24,
        Calc_Cycles    => 32,
        Filter_Cycles  => 32,
        Expand_Cycles  => 6914,
        Filter_Delay   => 1,
        Offset_In      => Layer_4_Out_Offset,
        Offset_Out     => Layer_5_Out_Offset, 
        Offset         => Layer_5_Offset,
        Weights        => Layer_5
    ) PORT MAP (
        iStream        => oStream_Conv_4,
        iData          => oData_Conv_4,
        oStream        => oStream_Conv_5,
        oData          => oData_Conv_5
    );

    CNN_Convolution_6 : CNN_Convolution
    GENERIC MAP (
        Input_Columns  => Layer_6_Columns,
        Input_Rows     => Layer_6_Rows,
        Input_Values   => Layer_6_Values,
        Filter_Columns => Layer_6_Filter_X,
        Filter_Rows    => Layer_6_Filter_Y,
        Filters        => Layer_6_Filters,
        Strides        => Layer_6_Strides,
        Activation     => Layer_6_Activation,
        Padding        => Layer_6_Padding,
        Input_Cycles   => 32,
        Value_Cycles   => 32,
        Calc_Cycles    => 32,
        Filter_Cycles  => 32,
        Expand_Cycles  => 9218,
        Filter_Delay   => 10,
        Offset_In      => Layer_5_Out_Offset,
        Offset_Out     => Layer_6_Out_Offset, 
        Offset         => Layer_6_Offset,
        Weights        => Layer_6
    ) PORT MAP (
        iStream        => oStream_Conv_5,
        iData          => oData_Conv_5,
        oStream        => oStream_Conv_6,
        oData          => oData_Conv_6
    );

    PROCESS (oStream_Conv_6)
    BEGIN
        IF (rising_edge(oStream_Conv_6.Data_CLK)) THEN
            oCycle_Flat_1             <= (oStream_Conv_6.Row * Flatten_1_Columns + oStream_Conv_6.Column) * Flatten_1_Values + oStream_Conv_6.Filter;
            oStream_Flat_1.Data_Valid <= oStream_Conv_6.Data_Valid;
            oData_Flat_1              <= oData_Conv_6;
        END IF;
    END PROCESS;

    oStream_Flat_1.Data_CLK <= oStream_Conv_6.Data_CLK;

    NN_Layer1 : NN_Layer
    GENERIC MAP (
        Inputs          => NN_Layer_1_Inputs,
        Outputs         => NN_Layer_1_Outputs,
        Activation      => NN_Layer_1_Activation,
        Input_Cycles    => 128,
        Calc_Cycles     => 10,
        Output_Cycles   => 10,
        Output_Delay    => 1,
        Offset_In       => Layer_6_Out_Offset,
        Offset_Out      => NN_Layer_1_Out_Offset,
        Offset          => NN_Layer_1_Offset,
        Weights         => NN_Layer_1
    ) PORT MAP (
        iStream         => oStream_Flat_1,
        iData           => oData_Flat_1,
        iCycle          => oCycle_Flat_1,
        oStream         => oStream_FC_1,
        oData           => oData_FC_1,
        oCycle          => oCycle_FC_1
    );

    oStream_1 <= oStream_FC_1;
    oData_1 <= oData_FC_1;
    oCycle_1 <= oCycle_FC_1;

    
END BEHAVIORAL;