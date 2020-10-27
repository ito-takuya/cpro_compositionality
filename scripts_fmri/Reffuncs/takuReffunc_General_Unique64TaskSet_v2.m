% Taku Ito
% 3/12/15

% This is a general script to import EDATs into MATLAB and produce stim files according to the type of stimulus files you want.
% Originally created to create stimulus files for auditory v visual visual task rule contrasts, i.e., constant/hi-pitch versus vertical/red sensory task rules
addpath('../scripts/');
% Subject numbers - this will need to incrementally edited as more subjects become preprocessed
subjNums = [013 014 016 017 018 021 023 024 025 026 027 028 030 031 032 033 034 035 037 038 039 040 041 042 043 045 046 047 048 049 050 053 055 056 057 058 062 063 064 066 067 068 069 070 072 074 075 076 077 081 082 085 086 087 088 090 092 093 094 095 097 098 099 101 102 103 104 105 106 108 109 110 111 112 114 115 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 134 135 136 137 138 139 140 141]
% excluding subject 041
% Need to create this for ReffuncToAFNI function
subjNumStr = '013 014 016 017 018 021 023 024 025 026 027 028 030 031 032 033 034 035 037 038 039 040 041 042 043 045 046 047 048 049 050 053 055 056 057 058 062 063 064 066 067 068 069 070 072 074 075 076 077 081 082 085 086 087 088 090 092 093 094 095 097 098 099 101 102 103 104 105 106 108 109 110 111 112 114 115 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 134 135 136 137 138 139 140 141';
subjNumStr = strread(subjNumStr, '%s', 'delimiter', ' ');

% Import fMRI Behavioral EDAT files
EDATImport = taku_EDATImportBySubj_IndivRITL('_fMRI_CPRO.txt', subjNums);

% Run reference function, converting trial information to TR-by-TR info
reffunc = takuReffunc_SRActFlow_v1(EDATImport, '64UniqueTask_Encoding');

% Put all subjects into a single vector
reffunc_vector = takuReffuncVector_IndivRITL(reffunc);

%%
% ReffuncToAFNI - create stimulus timingfiles in the current directory
[dmat, dLabels] = ReffuncToAFNI(reffunc_vector, subjNumStr, 8, [1:8], [581 581 581 581 581 581 581 581], .785,...
    {'Task1', ...
     'Task2', ...
     'Task3', ...
     'Task4', ...
     'Task5', ...
     'Task6', ...
     'Task7', ...
     'Task8', ...
     'Task9', ...
     'Task10', ...
     'Task11', ...
     'Task12', ...
     'Task13', ...
     'Task14', ...
     'Task15', ...
     'Task16', ...
     'Task17', ...
     'Task18', ...
     'Task19', ...
     'Task20', ...
     'Task21', ...
     'Task22', ...
     'Task23', ...
     'Task24', ...
     'Task25', ...
     'Task26', ...
     'Task27', ...
     'Task28', ...
     'Task29', ...
     'Task30', ...
     'Task31', ...
     'Task32', ...
     'Task33', ...
     'Task34', ...
     'Task35', ...
     'Task36', ...
     'Task37', ...
     'Task38', ...
     'Task39', ...
     'Task40', ...
     'Task41', ...
     'Task42', ...
     'Task43', ...
     'Task44', ...
     'Task45', ...
     'Task46', ...
     'Task47', ...
     'Task48', ...
     'Task49', ...
     'Task50', ...
     'Task51', ...
     'Task52', ...
     'Task53', ...
     'Task54', ...
     'Task55', ...
     'Task56', ...
     'Task57', ...
     'Task58', ...
     'Task59', ...
     'Task60', ...
     'Task61', ...
     'Task62', ...
     'Task63', ...
     'Task64'},...
    {'\w*_Enc\w*_TaskNum1_\w*', ...
     '\w*_Enc\w*_TaskNum2_\w*', ...
     '\w*_Enc\w*_TaskNum3_\w*', ...
     '\w*_Enc\w*_TaskNum4_\w*', ...
     '\w*_Enc\w*_TaskNum5_\w*', ...
     '\w*_Enc\w*_TaskNum6_\w*', ...
     '\w*_Enc\w*_TaskNum7_\w*', ...
     '\w*_Enc\w*_TaskNum8_\w*', ...
     '\w*_Enc\w*_TaskNum9_\w*', ...
     '\w*_Enc\w*_TaskNum10_\w*', ...
     '\w*_Enc\w*_TaskNum11_\w*', ...
     '\w*_Enc\w*_TaskNum12_\w*', ...
     '\w*_Enc\w*_TaskNum13_\w*', ...
     '\w*_Enc\w*_TaskNum14_\w*', ...
     '\w*_Enc\w*_TaskNum15_\w*', ...
     '\w*_Enc\w*_TaskNum16_\w*', ...
     '\w*_Enc\w*_TaskNum17_\w*', ...
     '\w*_Enc\w*_TaskNum18_\w*', ...
     '\w*_Enc\w*_TaskNum19_\w*', ...
     '\w*_Enc\w*_TaskNum20_\w*', ...
     '\w*_Enc\w*_TaskNum21_\w*', ...
     '\w*_Enc\w*_TaskNum22_\w*', ...
     '\w*_Enc\w*_TaskNum23_\w*', ...
     '\w*_Enc\w*_TaskNum24_\w*', ...
     '\w*_Enc\w*_TaskNum25_\w*', ...
     '\w*_Enc\w*_TaskNum26_\w*', ...
     '\w*_Enc\w*_TaskNum27_\w*', ...
     '\w*_Enc\w*_TaskNum28_\w*', ...
     '\w*_Enc\w*_TaskNum29_\w*', ...
     '\w*_Enc\w*_TaskNum30_\w*', ...
     '\w*_Enc\w*_TaskNum31_\w*', ...
     '\w*_Enc\w*_TaskNum32_\w*', ...
     '\w*_Enc\w*_TaskNum33_\w*', ...
     '\w*_Enc\w*_TaskNum34_\w*', ...
     '\w*_Enc\w*_TaskNum35_\w*', ...
     '\w*_Enc\w*_TaskNum36_\w*', ...
     '\w*_Enc\w*_TaskNum37_\w*', ...
     '\w*_Enc\w*_TaskNum38_\w*', ...
     '\w*_Enc\w*_TaskNum39_\w*', ...
     '\w*_Enc\w*_TaskNum40_\w*', ...
     '\w*_Enc\w*_TaskNum41_\w*', ...
     '\w*_Enc\w*_TaskNum42_\w*', ...
     '\w*_Enc\w*_TaskNum43_\w*', ...
     '\w*_Enc\w*_TaskNum44_\w*', ...
     '\w*_Enc\w*_TaskNum45_\w*', ...
     '\w*_Enc\w*_TaskNum46_\w*', ...
     '\w*_Enc\w*_TaskNum47_\w*', ...
     '\w*_Enc\w*_TaskNum48_\w*', ...
     '\w*_Enc\w*_TaskNum49_\w*', ...
     '\w*_Enc\w*_TaskNum50_\w*', ...
     '\w*_Enc\w*_TaskNum51_\w*', ...
     '\w*_Enc\w*_TaskNum52_\w*', ...
     '\w*_Enc\w*_TaskNum53_\w*', ...
     '\w*_Enc\w*_TaskNum54_\w*', ...
     '\w*_Enc\w*_TaskNum55_\w*', ...
     '\w*_Enc\w*_TaskNum56_\w*', ...
     '\w*_Enc\w*_TaskNum57_\w*', ...
     '\w*_Enc\w*_TaskNum58_\w*', ...
     '\w*_Enc\w*_TaskNum59_\w*', ...
     '\w*_Enc\w*_TaskNum60_\w*', ...
     '\w*_Enc\w*_TaskNum61_\w*', ...
     '\w*_Enc\w*_TaskNum62_\w*', ...
     '\w*_Enc\w*_TaskNum63_\w*', ...
     '\w*_Enc\w*_TaskNum64_\w*'},...
     1, 1, 'Unique64TaskSet_IndivRITL')

