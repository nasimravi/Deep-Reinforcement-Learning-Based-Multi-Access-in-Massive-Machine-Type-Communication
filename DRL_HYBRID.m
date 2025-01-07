classdef DRL_HYBRID < rl.env.MATLABEnvironment

    properties
        % Define environment properties
        xi = 64 % Number of preambles
        n = 175 % Number of devices
        mu_new = 0.1 % Initial average arrival rate
        packets = zeros(1, 175) % Packet buffer for devices
        btry = 0.1 * ones(1, 175) % Initial battery levels for devices
        ThroughputData = [] % Store throughput data
        energyData = [] % Store energy efficiency data
        lifetime = 0 % Counter for lifetime of the simulation
        seed = 10 % Random seed for reproducibility
        Total_time 
    end

    %%
    properties (Access = protected)
        % Initialize internal flag to indicate episode termination
        isDone = false        
    end

    %%
    properties
        State = zeros(1, 1)
    end
%% 
    methods
        function this = DRL_HYBRID()
            numActions = 2;

            % Define the observation space using rlNumericSpec

            ObservationInfo = rlNumericSpec([3, 1]);
            ObservationInfo.Name = 'States';

            ActionInfo = rlFiniteSetSpec(1:numActions);
            ActionInfo.Name = 'action';
            
            % Initialize the MATLABEnvironment
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            this.energyData = 0;
        end

        function [InitialObservation, InitialState] = reset(this)
            % Reset environment properties at the start of an episode
            this.btry = 0.1 * ones(1, 175);
            this.ThroughputData = 0;
            this.energyData = 0;
            this.packets = zeros(1, 175);
            this.lifetime = 0;
            this.Total_time = 0;

            InitialState.State = [0; 0; 0];
            InitialObservation = InitialState.State;
            this.State = InitialObservation;
        end

        function [obs, reward, isDone] = step(this, action)
            % Environment step function
            % This function updates the state, calculates reward, and checks for episode termination

            % Define constants for simulation
            Rho_dbm = -90; % Receive power (dBm)
            Rho = 10^(Rho_dbm / 10); % Convert to linear scale

            noise_dbm = -100.4; % Noise power (dBm)
            noise = 10^(noise_dbm / 10); % Convert to linear scale

            gamma_db = -10; % SINR threshold (dB)
            gamma = 10^(gamma_db / 10); % Convert to linear scale

            ms = 10^-3; % Millisecond constant
            Slot = 0.5 * ms; % Slot time length
            N_RAR = 40; % RAR window size
            Pr = 80 * (10^-3); % Receive power consumption
            Ps = 15 * (10^-6); % Idle power consumption
            N_CRT = 48; % Contention resolution time

            % Update random seed for reproducibility
            this.seed = this.seed + 1;
            rng(this.seed);

            % Generate random packet arrival rate
            arivval = rand;
            ir = 0 + (0.2 - 0) * arivval;

            % Update packet buffer for all devices
            this.packets = this.packets + random('poisson', ir, 1, 175);
            num_active = length(find(this.packets > 0));

            Energy = 1e-140;
            this.mu_new = 0.0175;
            this.xi = 64;
            Pt = 500 * (10^-3); % Transmission power
            preambles = randi(this.xi, [1, this.n]); % Assign random preambles
            channel_gain = exprnd(1, 1, this.n); % Generate random channel gains

            if action == 1
                % Grant-free random access protocol
                SINR = zeros(1, this.n);
                preambles(this.packets == 0) = 0; % Set preambles to 0 for inactive devices
                channel_gain(this.packets == 0) = 0; % Set channel gain to 0 for inactive devices

                % Update energy consumption and battery levels
                Energy = Energy + (Pt * 2 * Slot * num_active);
                this.btry(this.packets > 0) = this.btry(this.packets > 0) - (Pt * 2 * Slot);
                Time_s = 2 * Slot;

                % preamble collisions or not
                for xi_det = 1:this.xi
                    Index = find(preambles == xi_det);
                    if length(Index) > 1
                        preambles(preambles == xi_det) = 0; % Reset preambles for collided devices
                        this.btry(Index) = this.btry(Index) - (N_RAR * Pr * Slot);
                        Energy = Energy + length(Index) * N_RAR * Pr * Slot;
                        Time_s = Time_s + length(Index) * N_RAR * Slot;
                    end
                end

                % Calculate SINR for each device
                [sorted_channelgain, sorted_indices] = sort(channel_gain, 'descend');
                num_active_length = num_active + 1;
                sorted_indices(num_active_length:end) = 0; % Deactivate unused devices
                preamble_success = find(preambles > 0); % Devices with successful preambles
                num_preamble_success = length(preamble_success);

                for k = 1:num_active
                    interference = sum(sorted_channelgain(k + 1:num_active)) * Rho;
                    SINR(sorted_indices(k)) = channel_gain(sorted_indices(k)) * Rho / (interference + noise);
                end

                % Update battery levels and calculate energy for failed devices
                this.btry(SINR < gamma & SINR > 0) = this.btry(SINR < gamma & SINR > 0) - (N_RAR * Pr * Slot);
                SINR(preambles == 0) = 0;
                succ_index = find(SINR >= gamma); % Devices with successful transmissions

                Energy = Energy + (length(succ_index) * (N_RAR) * Slot * Pr) + ...
                         ((length(preamble_success) - length(succ_index)) * N_RAR * Pr * Slot);

                this.btry(succ_index) = this.btry(succ_index) - ((N_RAR ) * Slot * Pr);
                Time_s = Time_s + length(succ_index) * ((N_RAR) * Slot) + ...
                         ((length(preamble_success) - length(succ_index)) * N_RAR * Slot);
                
                % Update packets for successful transmissions
                this.packets(succ_index) = this.packets(succ_index) - 1;
                num_succ = length(succ_index);
                energy_Efficiency = num_succ / Energy; % Calculate energy efficiency
                Throughput_average_sum = num_succ / Time_s; % Calculate throughput

            else
                % Grant-based random access protocol

                preambles(this.packets == 0) = 0;
                Energy = Energy + (Pt * Slot * num_active);
                this.btry(this.packets > 0) = this.btry(this.packets > 0) - (Pt * Slot);
                Time_s = Slot;

                d = 0;
                for xi_det = 1:this.xi
                    Index = find(preambles == xi_det);
                    if length(Index) > 1
                        preambles(preambles == xi_det) = 0;
                        Energy = Energy + length(Index) * N_RAR * Pr * Slot;
                        this.btry(Index) = this.btry(Index) - (N_RAR * Pr * Slot);
                        Time_s = Time_s + length(Index) * N_RAR * Slot;
                    elseif isscalar(Index ...
                            )
                        %waiting time of preamble successful devices for sending msg3
                        d = d + 1;
                        %Eq (12)
                        Energy = Energy + ((N_RAR) * Slot * Pr + (d * Slot * Ps) + (Pt * Slot));
                        this.btry(Index) = this.btry(Index) - (((N_RAR) * Slot * Pr) + (d * Slot * Ps) + (Pt * Slot));
                        Time_s = Time_s + ((N_RAR) * Slot) + (d * Slot) + Slot;
                    end
                end

                succ_index = find(preambles > 0);
                num_succ = length(succ_index);
                Energy = Energy + length(succ_index) * (N_CRT) * (Slot * Pr);
                this.btry(succ_index) = this.btry(succ_index) - ((N_CRT) * (Slot * Pr));
                Time_s = Time_s + length(succ_index) * (N_CRT * Slot);

                this.packets(succ_index) = this.packets(succ_index) - 1;
                energy_Efficiency = num_succ / Energy;
                Throughput_average_sum = num_succ / Time_s;
            end

            if any(this.btry < 0)
                isDone = true;
            else
                isDone = false;
            end

            this.isDone = isDone;

            % Update rewards and episode status
            if ~this.isDone
                this.lifetime = this.lifetime + 1;
                this.Total_time = this.Total_time + Time_s;
                reward = Throughput_average_sum + energy_Efficiency;
            else
                reward = 0;
            end

            % Log throughput and energy data
            this.ThroughputData = [this.ThroughputData, Throughput_average_sum];
            this.energyData = [this.energyData, energy_Efficiency];

            faiulre=length(find(this.packets>0));


            % Update observation
            obs = [num_active;faiulre;Energy];
            

        end

    end  

end
