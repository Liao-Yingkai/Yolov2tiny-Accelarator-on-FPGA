   
    parameter PROC_NUM = 18;
    parameter ST_IDLE = 3'b000;
    parameter ST_FILTER_FAKE = 3'b001;
    parameter ST_DL_DETECTED = 3'b010;
    parameter ST_DL_REPORT = 3'b100;
   

    reg [2:0] CS_fsm;
    reg [2:0] NS_fsm;
    reg [PROC_NUM - 1:0] dl_detect_reg;
    reg [PROC_NUM - 1:0] dl_done_reg;
    reg [PROC_NUM - 1:0] origin_reg;
    reg [PROC_NUM - 1:0] dl_in_vec_reg;
    reg [31:0] dl_keep_cnt;
    integer i;
    integer fp;

    // FSM State machine
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            CS_fsm <= ST_IDLE;
        end
        else begin
            CS_fsm <= NS_fsm;
        end
    end
    always @ (CS_fsm or dl_in_vec or dl_detect_reg or dl_done_reg or dl_in_vec or origin_reg or dl_keep_cnt) begin
        case (CS_fsm)
            ST_IDLE : begin
                if (|dl_in_vec) begin
                    NS_fsm = ST_FILTER_FAKE;
                end
                else begin
                    NS_fsm = ST_IDLE;
                end
            end
            ST_FILTER_FAKE: begin
                if (dl_keep_cnt >= 32'd1000) begin
                    NS_fsm = ST_DL_DETECTED;
                end
                else if (dl_detect_reg != (dl_detect_reg & dl_in_vec)) begin
                    NS_fsm = ST_IDLE;
                end
                else begin
                    NS_fsm = ST_FILTER_FAKE;
                end
            end
            ST_DL_DETECTED: begin
                // has unreported deadlock cycle
                if (dl_detect_reg != dl_done_reg) begin
                    NS_fsm = ST_DL_REPORT;
                end
                else begin
                    NS_fsm = ST_DL_DETECTED;
                end
            end
            ST_DL_REPORT: begin
                if (|(dl_in_vec & origin_reg)) begin
                    NS_fsm = ST_DL_DETECTED;
                end
                else begin
                    NS_fsm = ST_DL_REPORT;
                end
            end
            default: NS_fsm = ST_IDLE;
        endcase
    end

    // dl_detect_reg record the procs that first detect deadlock
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            dl_detect_reg <= 'b0;
        end
        else begin
            if (CS_fsm == ST_IDLE) begin
                dl_detect_reg <= dl_in_vec;
            end
        end
    end

    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            dl_keep_cnt <= 32'h0;
        end
        else begin
            if (CS_fsm == ST_FILTER_FAKE && (dl_detect_reg == (dl_detect_reg & dl_in_vec))) begin
                dl_keep_cnt <= dl_keep_cnt + 32'h1;
            end
            else if (CS_fsm == ST_FILTER_FAKE && (dl_detect_reg != (dl_detect_reg & dl_in_vec))) begin
                dl_keep_cnt <= 32'h0;
            end
        end
    end

    // dl_detect_out keeps in high after deadlock detected
    assign dl_detect_out = (|dl_detect_reg) && (CS_fsm == ST_DL_DETECTED || CS_fsm == ST_DL_REPORT);

    // dl_done_reg record the cycles has been reported
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            dl_done_reg <= 'b0;
        end
        else begin
            if ((CS_fsm == ST_DL_REPORT) && (|(dl_in_vec & dl_detect_reg) == 'b1)) begin
                dl_done_reg <= dl_done_reg | dl_in_vec;
            end
        end
    end

    // clear token once a cycle is done
    assign token_clear = (CS_fsm == ST_DL_REPORT) ? ((|(dl_in_vec & origin_reg)) ? 'b1 : 'b0) : 'b0;

    // origin_reg record the current cycle start id
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            origin_reg <= 'b0;
        end
        else begin
            if (CS_fsm == ST_DL_DETECTED) begin
                origin_reg <= origin;
            end
        end
    end
   
    // origin will be valid for only one cycle
    wire [PROC_NUM*PROC_NUM - 1:0] origin_tmp;
    assign origin_tmp[PROC_NUM - 1:0] = (dl_detect_reg[0] & ~dl_done_reg[0]) ? 'b1 : 'b0;
    genvar j;
    generate
    for(j = 1;j < PROC_NUM;j = j + 1) begin: F1
        assign origin_tmp[j*PROC_NUM +: PROC_NUM] = (dl_detect_reg[j] & ~dl_done_reg[j]) ? ('b1 << j) : origin_tmp[(j - 1)*PROC_NUM +: PROC_NUM];
    end
    endgenerate
    always @ (CS_fsm or origin_tmp) begin
        if (CS_fsm == ST_DL_DETECTED) begin
            origin = origin_tmp[(PROC_NUM - 1)*PROC_NUM +: PROC_NUM];
        end
        else begin
            origin = 'b0;
        end
    end

    
    // dl_in_vec_reg record the current cycle dl_in_vec
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            dl_in_vec_reg <= 'b0;
        end
        else begin
            if (CS_fsm == ST_DL_DETECTED) begin
                dl_in_vec_reg <= origin;
            end
            else if (CS_fsm == ST_DL_REPORT) begin
                dl_in_vec_reg <= dl_in_vec;
            end
        end
    end
    
    // find_df_deadlock to report the deadlock
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            find_df_deadlock <= 1'b0;
        end
        else begin
            if (CS_fsm == ST_DL_DETECTED && dl_detect_reg == dl_done_reg) begin
                find_df_deadlock <= 1'b1;
            end
            else if (CS_fsm == ST_IDLE) begin
                find_df_deadlock <= 1'b0;
            end
        end
    end
    
    // get the first valid proc index in dl vector
    function integer proc_index(input [PROC_NUM - 1:0] dl_vec);
        begin
            proc_index = 0;
            for (i = 0; i < PROC_NUM; i = i + 1) begin
                if (dl_vec[i]) begin
                    proc_index = i;
                end
            end
        end
    endfunction

    // get the proc path based on dl vector
    function [1432:0] proc_path(input [PROC_NUM - 1:0] dl_vec);
        integer index;
        begin
            index = proc_index(dl_vec);
            case (index)
                0 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.entry_proc_U0";
                end
                1 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.image_loader_U0";
                end
                2 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0";
                end
                3 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0";
                end
                4 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0";
                end
                5 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0";
                end
                6 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0";
                end
                7 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0";
                end
                8 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0";
                end
                9 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0";
                end
                10 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0";
                end
                11 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0";
                end
                12 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0";
                end
                13 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0";
                end
                14 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0";
                end
                15 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0";
                end
                16 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_argmax_runner_U0";
                end
                17 : begin
                    proc_path = "cnn_accelerator_ddr_cnn_accelerator_ddr.result_writer_U0";
                end
                default : begin
                    proc_path = "unknown";
                end
            endcase
        end
    endfunction

    // print the headlines of deadlock detection
    task print_dl_head;
        begin
            $display("\n//////////////////////////////////////////////////////////////////////////////");
            $display("// ERROR!!! DEADLOCK DETECTED at %0t ns! SIMULATION WILL BE STOPPED! //", $time);
            $display("//////////////////////////////////////////////////////////////////////////////");
            fp = $fopen("deadlock_db.dat", "w");
        end
    endtask

    // print the start of a cycle
    task print_cycle_start(input reg [1432:0] proc_path, input integer cycle_id);
        begin
            $display("/////////////////////////");
            $display("// Dependence cycle %0d:", cycle_id);
            $display("// (1): Process: %0s", proc_path);
            $fdisplay(fp, "Dependence_Cycle_ID %0d", cycle_id);
            $fdisplay(fp, "Dependence_Process_ID 1");
            $fdisplay(fp, "Dependence_Process_path %0s", proc_path);
        end
    endtask

    // print the end of deadlock detection
    task print_dl_end(input integer num, input integer record_time);
        begin
            $display("////////////////////////////////////////////////////////////////////////");
            $display("// Totally %0d cycles detected!", num);
            $display("////////////////////////////////////////////////////////////////////////");
            $display("// ERROR!!! DEADLOCK DETECTED at %0t ns! SIMULATION WILL BE STOPPED! //", record_time);
            $display("//////////////////////////////////////////////////////////////////////////////");
            $fdisplay(fp, "Dependence_Cycle_Number %0d", num);
            $fclose(fp);
        end
    endtask

    // print one proc component in the cycle
    task print_cycle_proc_comp(input reg [1432:0] proc_path, input integer cycle_comp_id);
        begin
            $display("// (%0d): Process: %0s", cycle_comp_id, proc_path);
            $fdisplay(fp, "Dependence_Process_ID %0d", cycle_comp_id);
            $fdisplay(fp, "Dependence_Process_path %0s", proc_path);
        end
    endtask

    // print one channel component in the cycle
    task print_cycle_chan_comp(input [PROC_NUM - 1:0] dl_vec1, input [PROC_NUM - 1:0] dl_vec2);
        reg [1528:0] chan_path;
        integer index1;
        integer index2;
        begin
            index1 = proc_index(dl_vec1);
            index2 = proc_index(dl_vec2);
            case (index1)
                0 : begin
                    case(index2)
                    17: begin
                        if (~entry_proc_U0.result_c_blk_n) begin
                            if (~result_c_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_c_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_writer_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.result_c_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~result_c_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_c_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_writer_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.result_c_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~start_for_result_writer_U0_U.if_full_n & entry_proc_U0.ap_start & ~entry_proc_U0.real_start & (trans_in_cnt_10 == trans_out_cnt_10) & ~start_for_result_writer_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.start_for_result_writer_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_writer_U0',");
                        end
                    end
                    1: begin
                        if (ap_sync_entry_proc_U0_ap_ready & entry_proc_U0.ap_idle & ~ap_sync_image_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.image_loader_U0'");
                        end
                    end
                    2: begin
                        if (ap_sync_entry_proc_U0_ap_ready & entry_proc_U0.ap_idle & ~ap_sync_weight_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                        end
                    end
                    3: begin
                        if (ap_sync_entry_proc_U0_ap_ready & entry_proc_U0.ap_idle & ~ap_sync_param_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                        end
                    end
                    endcase
                end
                1 : begin
                    case(index2)
                    4: begin
                        if (~image_loader_U0.grp_image_loader_Pipeline_VITIS_LOOP_109_2_fu_80.pixels_blk_n) begin
                            if (~pixels_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.pixels_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.pixels_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~pixels_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.pixels_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.pixels_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~start_for_p_anonymous_namespace_compute_runner_U0_U.if_full_n & image_loader_U0.ap_start & ~image_loader_U0.real_start & (trans_in_cnt_0 == trans_out_cnt_0) & ~start_for_p_anonymous_namespace_compute_runner_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.start_for_p_anonymous_namespace_compute_runner_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0',");
                        end
                    end
                    0: begin
                        if (ap_sync_image_loader_U0_ap_ready & image_loader_U0.ap_idle & ~ap_sync_entry_proc_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.entry_proc_U0'");
                        end
                    end
                    2: begin
                        if (ap_sync_image_loader_U0_ap_ready & image_loader_U0.ap_idle & ~ap_sync_weight_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                        end
                    end
                    3: begin
                        if (ap_sync_image_loader_U0_ap_ready & image_loader_U0.ap_idle & ~ap_sync_param_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                        end
                    end
                    endcase
                end
                2 : begin
                    case(index2)
                    4: begin
                        if (~weight_loader_U0.w1_blk_n) begin
                            if (~w1_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.w1_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.w1_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~w1_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.w1_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.w1_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~weight_loader_U0.w2_blk_n) begin
                            if (~w2_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.w2_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.w2_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~w2_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.w2_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.w2_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~weight_loader_U0.grp_weight_loader_Pipeline_VITIS_LOOP_147_1_VITIS_LOOP_150_2_fu_517.wf_blk_n) begin
                            if (~wf_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.wf_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.wf_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~wf_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.wf_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.wf_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    0: begin
                        if (ap_sync_weight_loader_U0_ap_ready & weight_loader_U0.ap_idle & ~ap_sync_entry_proc_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.entry_proc_U0'");
                        end
                    end
                    1: begin
                        if (ap_sync_weight_loader_U0_ap_ready & weight_loader_U0.ap_idle & ~ap_sync_image_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.image_loader_U0'");
                        end
                    end
                    3: begin
                        if (ap_sync_weight_loader_U0_ap_ready & weight_loader_U0.ap_idle & ~ap_sync_param_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                        end
                    end
                    endcase
                end
                3 : begin
                    case(index2)
                    4: begin
                        if (~param_loader_U0.p1_blk_n) begin
                            if (~p1_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p1_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p1_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p1_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p1_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p1_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~param_loader_U0.p2_blk_n) begin
                            if (~p2_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p2_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p2_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p2_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p2_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p2_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~param_loader_U0.pf_blk_n) begin
                            if (~pf_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.pf_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.pf_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~pf_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.pf_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.pf_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    0: begin
                        if (ap_sync_param_loader_U0_ap_ready & param_loader_U0.ap_idle & ~ap_sync_entry_proc_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.entry_proc_U0'");
                        end
                    end
                    1: begin
                        if (ap_sync_param_loader_U0_ap_ready & param_loader_U0.ap_idle & ~ap_sync_image_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.image_loader_U0'");
                        end
                    end
                    2: begin
                        if (ap_sync_param_loader_U0_ap_ready & param_loader_U0.ap_idle & ~ap_sync_weight_loader_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                        end
                    end
                    endcase
                end
                4 : begin
                    case(index2)
                    1: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0.grp_p_anonymous_namespace_input_router_Pipeline_VITIS_LOOP_31_1_fu_64.pixels_blk_n) begin
                            if (~pixels_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.pixels_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.image_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.pixels_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~pixels_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.pixels_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.image_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.pixels_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~start_for_p_anonymous_namespace_compute_runner_U0_U.if_empty_n & p_anonymous_namespace_compute_runner_U0.ap_idle & ~start_for_p_anonymous_namespace_compute_runner_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.start_for_p_anonymous_namespace_compute_runner_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.image_loader_U0',");
                        end
                    end
                    2: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0.w1_blk_n) begin
                            if (~w1_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.w1_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.w1_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~w1_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.w1_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.w1_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0.w2_blk_n) begin
                            if (~w2_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.w2_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.w2_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~w2_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.w2_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.w2_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0.grp_fully_connected_fu_50.wf_blk_n) begin
                            if (~wf_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.wf_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.wf_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~wf_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.wf_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.weight_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.wf_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    3: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0.p1_blk_n) begin
                            if (~p1_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p1_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p1_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p1_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p1_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p1_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0.p2_blk_n) begin
                            if (~p2_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p2_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p2_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p2_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p2_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p2_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0.grp_fc_postprocess_fu_38.pf_blk_n) begin
                            if (~pf_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.pf_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.pf_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~pf_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.pf_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.param_loader_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.pf_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    15: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0.grp_fc_postprocess_fu_38.computed_blk_n) begin
                            if (~computed_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.computed_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.computed_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~computed_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.computed_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.computed_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~start_for_logits_split_U0_U.if_full_n & p_anonymous_namespace_compute_runner_U0.ap_start & ~p_anonymous_namespace_compute_runner_U0.real_start & (trans_in_cnt_8 == trans_out_cnt_8) & ~start_for_logits_split_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.start_for_logits_split_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0',");
                        end
                    end
                    endcase
                end
                5 : begin
                    case(index2)
                    8: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0.grp_p_anonymous_namespace_input_router_Pipeline_VITIS_LOOP_31_1_fu_64.conv_input_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0.layer_mode_c28_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_im2row_U0_U.if_full_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0.ap_start & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0.real_start & (trans_in_cnt_1 == trans_out_cnt_1) & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_im2row_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_im2row_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0',");
                        end
                    end
                    12: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0.grp_p_anonymous_namespace_input_router_Pipeline_VITIS_LOOP_31_1_fu_64.pool1_feature_fifo_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    6: begin
                        if (p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_input_router_U0_ap_ready & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_conv_weight_router_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0'");
                        end
                    end
                    7: begin
                        if (p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_input_router_U0_ap_ready & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_conv_param_router_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0'");
                        end
                    end
                    endcase
                end
                6 : begin
                    case(index2)
                    9: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0.conv_weight_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_systolic_OS_U0_U.if_full_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0.ap_start & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0.real_start & (trans_in_cnt_2 == trans_out_cnt_2) & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_systolic_OS_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_systolic_OS_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0',");
                        end
                    end
                    5: begin
                        if (p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_conv_weight_router_U0_ap_ready & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_input_router_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0'");
                        end
                    end
                    7: begin
                        if (p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_conv_weight_router_U0_ap_ready & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_conv_param_router_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0'");
                        end
                    end
                    endcase
                end
                7 : begin
                    case(index2)
                    10: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0.conv_param_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_conv_postprocess_U0_U.if_full_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0.ap_start & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0.real_start & (trans_in_cnt_3 == trans_out_cnt_3) & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_conv_postprocess_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_conv_postprocess_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0',");
                        end
                    end
                    5: begin
                        if (p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_conv_param_router_U0_ap_ready & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_input_router_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0'");
                        end
                    end
                    6: begin
                        if (p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_conv_param_router_U0_ap_ready & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.ap_sync_p_anonymous_namespace_conv_weight_router_U0_ap_ready) begin
                            $display("//      Blocked by input sync logic with process : 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0'");
                        end
                    end
                    endcase
                end
                8 : begin
                    case(index2)
                    5: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0.grp_im2row_Pipeline_VITIS_LOOP_317_3_fu_149.conv_input_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_input_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0.layer_mode_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c28_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_im2row_U0_U.if_empty_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_im2row_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_im2row_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0',");
                        end
                    end
                    9: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0.grp_im2row_Pipeline_VITIS_LOOP_317_3_fu_149.activation_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0.layer_mode_c27_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    endcase
                end
                9 : begin
                    case(index2)
                    8: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0.grp_systolic_OS_Pipeline_VITIS_LOOP_211_3_fu_235.activation_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.activation_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0.layer_mode_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.im2row_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c27_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    6: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0.grp_systolic_OS_Pipeline_VITIS_LOOP_211_3_fu_235.conv_weight_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_weight_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_systolic_OS_U0_U.if_empty_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_systolic_OS_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_systolic_OS_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_weight_router_U0',");
                        end
                    end
                    10: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0.acc_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0.layer_mode_c26_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    endcase
                end
                10 : begin
                    case(index2)
                    9: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0.acc_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.acc_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0.layer_mode_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.systolic_OS_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c26_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    7: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0.conv_param_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_param_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_conv_postprocess_U0_U.if_empty_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_conv_postprocess_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_conv_postprocess_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_conv_param_router_U0',");
                        end
                    end
                    11: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0.grp_conv_postprocess_Pipeline_VITIS_LOOP_128_4_fu_634.post_output_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0.layer_mode_c25_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_maxpool_U0_U.if_full_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0.ap_start & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0.real_start & (trans_in_cnt_4 == trans_out_cnt_4) & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_maxpool_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_maxpool_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0',");
                        end
                    end
                    endcase
                end
                11 : begin
                    case(index2)
                    10: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0.grp_maxpool_Pipeline_VITIS_LOOP_156_3_fu_274.post_output_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.post_output_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0.layer_mode_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c25_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_maxpool_U0_U.if_empty_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_maxpool_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_maxpool_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.conv_postprocess_U0',");
                        end
                    end
                    12: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0.grp_maxpool_Pipeline_VITIS_LOOP_210_6_fu_337.pool_output_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0.layer_mode_c24_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_output_router_U0_U.if_full_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0.ap_start & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0.real_start & (trans_in_cnt_5 == trans_out_cnt_5) & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_output_router_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_output_router_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0',");
                        end
                    end
                    endcase
                end
                12 : begin
                    case(index2)
                    11: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0.grp_p_anonymous_namespace_output_router_Pipeline_VITIS_LOOP_125_1_fu_64.pool_output_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool_output_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0.layer_mode_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c24_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_output_router_U0_U.if_empty_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_output_router_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_output_router_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.maxpool_U0',");
                        end
                    end
                    13: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0.grp_p_anonymous_namespace_output_router_Pipeline_VITIS_LOOP_125_1_fu_64.fc_input_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0.layer_mode_c23_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fully_connected_guard_U0_U.if_full_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0.ap_start & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0.real_start & (trans_in_cnt_6 == trans_out_cnt_6) & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fully_connected_guard_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fully_connected_guard_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0',");
                        end
                    end
                    5: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0.grp_p_anonymous_namespace_output_router_Pipeline_VITIS_LOOP_125_1_fu_64.pool1_feature_fifo_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_input_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.pool1_feature_fifo_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    endcase
                end
                13 : begin
                    case(index2)
                    12: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0.grp_fully_connected_fu_50.fc_input_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_input_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0.layer_mode_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c23_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fully_connected_guard_U0_U.if_empty_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fully_connected_guard_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fully_connected_guard_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_output_router_U0',");
                        end
                    end
                    14: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0.grp_fully_connected_fu_50.fc_acc_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0.layer_mode_c_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fc_postprocess_guard_U0_U.if_full_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0.ap_start & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0.real_start & (trans_in_cnt_7 == trans_out_cnt_7) & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fc_postprocess_guard_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fc_postprocess_guard_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0',");
                        end
                    end
                    endcase
                end
                14 : begin
                    case(index2)
                    13: begin
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0.grp_fc_postprocess_fu_38.fc_acc_stream_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.fc_acc_stream_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0.layer_mode_blk_n) begin
                            if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.layer_mode_c_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fc_postprocess_guard_U0_U.if_empty_n & p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fc_postprocess_guard_U0.ap_idle & ~p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fc_postprocess_guard_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.start_for_p_anonymous_namespace_fc_postprocess_guard_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0.grp_p_anonymous_namespace_run_layer_phase_fu_50.p_anonymous_namespace_fully_connected_guard_U0',");
                        end
                    end
                    endcase
                end
                15 : begin
                    case(index2)
                    4: begin
                        if (~logits_split_U0.computed_blk_n) begin
                            if (~computed_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.computed_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.computed_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~computed_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.computed_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.computed_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~start_for_logits_split_U0_U.if_empty_n & logits_split_U0.ap_idle & ~start_for_logits_split_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.start_for_logits_split_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_compute_runner_U0',");
                        end
                    end
                    17: begin
                        if (~logits_split_U0.write_logits_blk_n) begin
                            if (~write_logits_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.write_logits_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_writer_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.write_logits_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~write_logits_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.write_logits_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_writer_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.write_logits_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    16: begin
                        if (~logits_split_U0.argmax_logits_blk_n) begin
                            if (~argmax_logits_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.argmax_logits_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_argmax_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.argmax_logits_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~argmax_logits_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.argmax_logits_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_argmax_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.argmax_logits_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~start_for_p_anonymous_namespace_argmax_runner_U0_U.if_full_n & logits_split_U0.ap_start & ~logits_split_U0.real_start & (trans_in_cnt_9 == trans_out_cnt_9) & ~start_for_p_anonymous_namespace_argmax_runner_U0_U.if_read) begin
                            $display("//      Blocked by full output start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.start_for_p_anonymous_namespace_argmax_runner_U0_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_argmax_runner_U0',");
                        end
                    end
                    endcase
                end
                16 : begin
                    case(index2)
                    15: begin
                        if (~p_anonymous_namespace_argmax_runner_U0.grp_logits_argmax_fu_16.argmax_logits_blk_n) begin
                            if (~argmax_logits_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.argmax_logits_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.argmax_logits_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~argmax_logits_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.argmax_logits_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.argmax_logits_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~start_for_p_anonymous_namespace_argmax_runner_U0_U.if_empty_n & p_anonymous_namespace_argmax_runner_U0.ap_idle & ~start_for_p_anonymous_namespace_argmax_runner_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.start_for_p_anonymous_namespace_argmax_runner_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0',");
                        end
                    end
                    17: begin
                        if (~p_anonymous_namespace_argmax_runner_U0.grp_logits_argmax_fu_16.classes1_blk_n) begin
                            if (~classes_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.classes_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_writer_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.classes_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~classes_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.classes_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_writer_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.classes_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    endcase
                end
                17 : begin
                    case(index2)
                    15: begin
                        if (~result_writer_U0.write_logits_blk_n) begin
                            if (~write_logits_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.write_logits_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.write_logits_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~write_logits_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.write_logits_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.logits_split_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.write_logits_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    16: begin
                        if (~result_writer_U0.classes1_blk_n) begin
                            if (~classes_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.classes_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_argmax_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.classes_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~classes_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.classes_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.p_anonymous_namespace_argmax_runner_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.classes_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                    end
                    0: begin
                        if (~result_writer_U0.result_blk_n) begin
                            if (~result_c_U.if_empty_n) begin
                                $display("//      Blocked by empty input FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_c_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.entry_proc_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.result_c_U");
                                $fdisplay(fp, "Dependence_Channel_status EMPTY");
                            end
                            else if (~result_c_U.if_full_n) begin
                                $display("//      Blocked by full output FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.result_c_U' read by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.entry_proc_U0'");
                                $fdisplay(fp, "Dependence_Channel_path cnn_accelerator_ddr_cnn_accelerator_ddr.result_c_U");
                                $fdisplay(fp, "Dependence_Channel_status FULL");
                            end
                        end
                        if (~start_for_result_writer_U0_U.if_empty_n & result_writer_U0.ap_idle & ~start_for_result_writer_U0_U.if_write) begin
                            $display("//      Blocked by missing 'ap_start' from start propagation FIFO 'cnn_accelerator_ddr_cnn_accelerator_ddr.start_for_result_writer_U0_U' written by process 'cnn_accelerator_ddr_cnn_accelerator_ddr.entry_proc_U0',");
                        end
                    end
                    endcase
                end
            endcase
        end
    endtask

    // report
    initial begin : report_deadlock
        integer cycle_id;
        integer cycle_comp_id;
        integer record_time;
        wait (dl_reset == 1);
        cycle_id = 1;
        record_time = 0;
        while (1) begin
            @ (negedge dl_clock);
            case (CS_fsm)
                ST_DL_DETECTED: begin
                    cycle_comp_id = 2;
                    if (dl_detect_reg != dl_done_reg) begin
                        if (dl_done_reg == 'b0) begin
                            print_dl_head;
                            record_time = $time;
                        end
                        print_cycle_start(proc_path(origin), cycle_id);
                        cycle_id = cycle_id + 1;
                    end
                    else begin
                        print_dl_end((cycle_id - 1),record_time);
                        @(negedge dl_clock);
                        @(negedge dl_clock);
                        $finish;
                    end
                end
                ST_DL_REPORT: begin
                    if ((|(dl_in_vec)) & ~(|(dl_in_vec & origin_reg))) begin
                        print_cycle_chan_comp(dl_in_vec_reg, dl_in_vec);
                        print_cycle_proc_comp(proc_path(dl_in_vec), cycle_comp_id);
                        cycle_comp_id = cycle_comp_id + 1;
                    end
                    else begin
                        print_cycle_chan_comp(dl_in_vec_reg, dl_in_vec);
                    end
                end
            endcase
        end
    end
 
