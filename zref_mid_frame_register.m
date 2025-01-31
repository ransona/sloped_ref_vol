
function zref_mid_frame_register()
% filename root
stack_filename_stem = fileID;
% where to save final tiff
expDir = fullfile('V:\Local_Repository',animalID,'refz',[ref_name,stack_filename_stem,'.tif']);
expDirTemp = fullfile('V:\Local_Repository',animalID,'refz',stack_filename_stem);
fast_z_slices = 5;
fast_z_step = 50;

% register
% load the complete tif of one 5 level stack
imageFullFileName = dir(fullfile(expDirTemp,['*',stack_filename_stem,'*']));
[~,idx] = sort([imageFullFileName.datenum]);
imageFullFileName = imageFullFileName(idx);
imageFullFileName = {imageFullFileName.name};
% allocate space to store the registered frames from each depth and the
% depth value
all_registered_depths_frames = [];
all_registered_depths = [];
total_depths = 0;
for iFile = 1:length(imageFullFileName)
    disp(['Processing file ',num2str(iFile),'/',num2str(length(imageFullFileName))]);
    info = imfinfo(fullfile(expDirTemp,imageFullFileName{iFile}));
    numberOfPages = length(info);
    all_pages = [];
    current_page = 0;
    % load all pages of tif
    for page_num = ch:ch_active:numberOfPages
        current_page = current_page + 1;
        % Read the kth image in this multipage tiff file.
        all_pages(:,:,current_page) = imread(fullfile(expDirTemp,imageFullFileName{iFile}), page_num);
        % Now process thisPage somehow...
    end

    % register pages from same depth of fast z volume
    for iDepth = 3 % only do the middle depth
        disp(['Processing depth ',num2str(iDepth),'/',num2str(fast_z_slices)]);
        slice_frames = all_pages(:,:,iDepth:fast_z_slices:end);
        % test reg function
        % frame1 = double(imread('coins.png'));
        % allfr = cat(3,frame1,circshift(frame1,20,1));
        % regMovie = mean(rapidRegNonPar(allfr,squeeze(allfr(:,:,1))),3);
        % make ref for registration by registering 5 frames - start
        % with 2nd to avoid bad first frame
        slice_frames_ref = mean(rapidRegNonPar(slice_frames(:,:,round(linspace(2,size(slice_frames,3),5))),slice_frames(:,:,round(size(slice_frames,3)/2))),3);
        % register the rest of the frames
        slice_frames_reg = rapidRegNonPar(slice_frames,slice_frames_ref);
        % average the registered frames and allocate them to their depth
        total_depths = total_depths +1;
        all_registered_depths_frames(:,:,total_depths) = mean(slice_frames_reg(:,:,2:end),3);
        % calc depth using fast z slice spacing and file number
        all_registered_depths(total_depths) = ((iDepth - 1)*fast_z_step)+((iFile-1)*plane_spacing);
    end
end

% sort registered depth frame order depth
[all_registered_depths_sorted,i] = sort(all_registered_depths);
all_registered_depths_frames_sorted = all_registered_depths_frames(:,:,i);

% register each depth to the one above it and z score
% to avoid stange alignment because of the top sections being noise
% align inside out:
% from middle to bottom
for iDepth = floor(size(all_registered_depths_frames_sorted,3)/2):size(all_registered_depths_frames_sorted,3)
    if iDepth > floor(size(all_registered_depths_frames_sorted,3)/2)
        all_registered_depths_frames_sorted(:,:,iDepth) = rapidRegNonPar(all_registered_depths_frames_sorted(:,:,iDepth),all_registered_depths_frames_sorted(:,:,iDepth-1));
    end
    depth_frame = all_registered_depths_frames_sorted(:,:,iDepth);
    depth_frame = depth_frame - min(depth_frame(:));
    depth_frame = depth_frame / max(depth_frame(:));
    depth_frame = depth_frame * 255;
    all_registered_depths_frames_sorted(:,:,iDepth) = depth_frame;
end
% from middle to the top
for iDepth = 1:floor(size(all_registered_depths_frames_sorted,3)/2)
    if iDepth > 1
        all_registered_depths_frames_sorted(:,:,iDepth) = rapidRegNonPar(all_registered_depths_frames_sorted(:,:,iDepth),all_registered_depths_frames_sorted(:,:,iDepth-1));
    end
    depth_frame = all_registered_depths_frames_sorted(:,:,iDepth);
    depth_frame = depth_frame - min(depth_frame(:));
    depth_frame = depth_frame / max(depth_frame(:));
    depth_frame = depth_frame * 255;
    all_registered_depths_frames_sorted(:,:,iDepth) = depth_frame;
end

f = figure;
while true
    for iDepth = 1:size(all_registered_depths_frames_sorted,3)
        imagesc(all_registered_depths_frames_sorted(:,:,iDepth),[0 256])
        colorbar
        drawnow
        pause(0.2)
    end
    if ~strcmp(questdlg('Replay stack?'),'Yes')
        close(f)
        break
    end
end

all_registered_depths_frames_sorted = uint16(all_registered_depths_frames_sorted);
% output as a registration object for SI online motion correction
imwrite(squeeze(all_registered_depths_frames_sorted(:,:,1)), expDir);
for iFrame = 2:size(all_registered_depths_frames_sorted,3)
    imwrite(squeeze(all_registered_depths_frames_sorted(:,:,iFrame)), expDir, 'WriteMode', 'append');
end
disp(['Time taken = ',num2str(toc(x))]);

