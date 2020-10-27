% Taku Ito
% 3/12/15

% This is a general script to import EDATs into MATLAB and produce stim files according to the type of stimulus files you want.
% Originally created to create stimulus files for auditory v visual visual task rule contrasts, i.e., constant/hi-pitch versus vertical/red sensory task rules
addpath('../');
% Subject numbers - this will need to incrementally edited as more subjects become preprocessed
%subjNums = [013 014 016 017 018 021 023 024 025 026 027 028 030 031 032 033 034 035 037 038 039 040 042 043 045 046 047 048 049 050 053 055 057 062]; % 063 066 067 068 069 070 072 074 075 077 081 086 088]; %041 

subjNums = [013 014 016 017 018 021 023 024 025 026 027 028 030 031 032 033 034 035 037 038 039 040 041 042 043 045 046 047 048 049 050 053 055 056 057 058 062 063 064 066 067 068 069 070 072 074 075 076 077 081 082 085 086 087 088 090 092 093 094 095 097 098 099 101 102 103 104 105 106 108 109 110 111 112 114 115 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 134 135 136 137 138 139 140 141];
% excluding subject 041
% Need to create this for ReffuncToAFNI function
subjNumStr = '013 014 016 017 018 021 023 024 025 026 027 028 030 031 032 033 034 035 037 038 039 040 041 042 043 045 046 047 048 049 050 053 055 056 057 058 062 063 064 066 067 068 069 070 072 074 075 076 077 081 082 085 086 087 088 090 092 093 094 095 097 098 099 101 102 103 104 105 106 108 109 110 111 112 114 115 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 134 135 136 137 138 139 140 141';
subjNumStr = strread(subjNumStr, '%s', 'delimiter', ' ');

% Import fMRI Behavioral EDAT files
EDATImport = taku_EDATImportBySubj_IndivRITL('_fMRI_CPRO.txt', subjNums);

% Run reference function, converting trial information to TR-by-TR info
reffunc = takuReffunc_SRActFlow_v1(EDATImport, 'BetaSeries');

% Put all subjects into a single vector
reffunc_vector = takuReffuncVector_IndivRITL(reffunc);

% ReffuncToAFNI - create stimulus timingfiles in the current directory
parpool(21)
parfor (block=1:84)

    [dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 8, [1:8], [581 581 581 581 581 581 581 581], .785,...
        {['Miniblock' num2str(block) '_Encoding'],
        ['Miniblock' num2str(block) '_Probe1'],
        ['Miniblock' num2str(block) '_Probe2'],
        ['Miniblock' num2str(block) '_Probe3']},...
        {['\w*_Task_Enc\w*_Miniblock' num2str(block) '_\w*'], 
        ['\w*_Task_Probe1_\w*_Miniblock' num2str(block) '_\w*'], 
        ['\w*_Task_Probe2_\w*_Miniblock' num2str(block) '_\w*'],
        ['\w*_Task_Probe3_\w*_Miniblock' num2str(block) '_\w*']},...
        1, 1, 'BetaSeries');
end

