function zref_mid_frame_acquire(animalID,ref_name,ch)
if ~exist('animalID')
    error('No animal ID')
end
reg_only = false;
fileID = '';
ch_active = 2; % change to hSI.hChannels.channelsActive once working
%ch = 1; % 1 = green, 2 = red
% prompt for desired spacing of planes in reference stack
plane_spacing = 2; %um
total_planes = 41; %41;
total_range = plane_spacing * (total_planes-1);
% prompt for desired frames per depth
frames_per_depth = 30;

global hSI;


% filename root
stack_filename_stem = datestr(now);
stack_filename_stem = strrep(stack_filename_stem,':','_');
stack_filename_stem = [ref_name,'_',stack_filename_stem];

rawDataDir = fullfile('V:\Local_Repository',animalID,'refz',stack_filename_stem,'raw');
[ ~, ~ ] = mkdir(rawDataDir);
x = tic;

logInitialState = hSI.hChannels.loggingEnable;
hSI.hMotionManager.enable=false;
hSI.hChannels.loggingEnable=true;
% start with the fast z profile open that you want to use with the scope
% in the zoom / zposition that you want to use
% channel to use


hSI.hScan2D.logFilePath = rawDataDir;

% position fast z actuator at zero
% hSI.hFastZ.positionAbsolute = 0;
% zeros all motor positions
hSI.hMotors.setRelativeZero; % this line may change with SI version

% get z spacing of fast z planes
fast_z_step = hSI.hStackManager.stackZStepSize;
fast_z_slices = hSI.hStackManager.numSlices;
% calculate reference stack z positions
num_ref_planes = length(0:plane_spacing:fast_z_step);
ref_stack_z = -(total_range/2):plane_spacing:(total_range/2);
% set SI volumes to frames_per_depth
numVolumes_original = hSI.hStackManager.numVolumes;
hSI.hStackManager.numVolumes = frames_per_depth;
all_filenames = [];
for iDepth = 1:length(ref_stack_z)
    disp(['Acquiring volume ',num2str(iDepth),' of ',num2str(length(ref_stack_z))]);
    % set z position
    hSI.hMotors.moveSample([0 0 ref_stack_z(iDepth)]);

    % set filename
    % hSI.hScan2D.logFileStem = ['z_ref_',stack_filename_stem,'_',sprintf('%06d', ref_stack_z(iDepth)),'_'];
    hSI.hScan2D.logFileStem = ['z_ref_',stack_filename_stem,'_',sprintf('%06d', iDepth),'_'];

    hSI.startGrab;
    % drawnow;
    % pause(3);
    % wait for grab to be complete
    while strcmp(hSI.acqState,'grab')
        drawnow();
    end
end


disp(['Time taken = ',num2str(toc(x))]);


% put settings back to start state
hSI.hStackManager.numVolumes = 99999999;
hSI.hChannels.loggingEnable=logInitialState;
hSI.hScan2D.logFileStem = '';
hSI.hScan2D.logFilePath = '';
% set motor to where it was
hSI.hMotors.moveSample([0 0 0]);

