% load in timeseries post fsl feat
datadir = '/home/rowanhuxley/Documents/Data_Various/NotBinRiv/CHRONO/func';
subID = '/'; %'VOL90';
tmpfilt = load_untouch_nii(strcat(datadir,subID,'fMRI_CHRONO_fMRI_CHRONO_1000ms_20240618124933_14.nii.gz'));
% tmpfilt = load_untouch_nii(strcat(datadir,subID,'/test_filtered_func_data3.nii'));
filttimecourse = double(tmpfilt.img);
filttimecourse = squeeze(filttimecourse);

% load the roi (irrespective of distance here)
restricted = load_untouch_nii(strcat(datadir,subID,'fMRI_CHRONO_fMRI_CHRONO_1000ms_20240618124933_14_mask.nii.gz'));
restrictedAngle = double(restricted.img);

% get x,y,z coords for ROI
[xCoord,yCoord,zCoord] = ind2sub(size(restrictedAngle),find(restrictedAngle));

% now extract the timecourse for these voxels only
for ivoxel = 1:size(xCoord)
    voxelData(ivoxel,:) = squeeze(filttimecourse(xCoord(ivoxel),yCoord(ivoxel),:)); %,zCoord(ivoxel),:);
end

% average across trials for each voxel (so matrix should be [nVoxels x
% 90ms]
for ivoxel = 1:size(voxelData,1)
    % Numbers for Chrono
    ttt = voxelData(ivoxel,580:end);
    ttt = detrend(ttt(1:2800));
    ttt = reshape(ttt,280,10); 
    
    % Numbers for VOL90
    % ttt = voxelData(ivoxel,280:end);
    % ttt = detrend(ttt(1:3060));
    % ttt = reshape(ttt,90,34); 

    % remove the trials where no wave was reported 
    %run1: [2.9.12.13.14.15.20.21.26.29.30.32.34]
    % run2: [4,5,9,11,12,13,16,18,19,21,23,25,26,28,32]
    % run3: [6,7,11,12,15,21,22,25,27,28,29,30,31]
    % ttt = ttt(:,[1,3:8,10:11,16:19,22:25,27,28,31,33]);
    % ttt = ttt(:,[1:3,6:8,10,14,15,17,20,22,24,27,29:31,33,34]);
    % ttt = ttt(:,[1:5,8:10,13:14,16:20,23:24,26,32:34]);
    ttt = mean(ttt,2); %mean across trials per voxel
    ttt = sgolayfilt(ttt,1,11);
    % ttt = ttt./max(ttt);
    voxelMean(ivoxel,:) = ttt;
end

voxelMean2 = voxelMean./(max(max(voxelMean)));

tim=0.1:0.1:28; %Chrono Time
% tim=0.1:0.1:9; % TW time
figure, plot(tim-2,voxelMean2')
xlabel('Time (s) from stim onset')
ylabel('Normalised BOLD resoponse')
title('Averaged BOLD responses from each voxel')
ylim([-0.8 1.2])

% Find the min and max and remove values before and after
for CurRunNum = 1:size(voxelMean2)
    CurRun = voxelMean2(CurRunNum,:);
    Max = max(CurRun);
    MaxLocation = find(CurRun==Max);
    CurRun(MaxLocation+5:max(size(CurRun))) = NaN;
    Min = min(CurRun);
    MinLocation = find(CurRun==Min);
    CurRun(1:MinLocation-1) = NaN;
    NewArray(CurRunNum,:) = CurRun;
    MinMax(CurRunNum,1) = MinLocation;
    MinMax(CurRunNum,2) = MaxLocation;
end

figure, plot(tim-2,NewArray')
xlabel('Time (s) from stim onset')
ylabel('Normalised BOLD resoponse')
title('Onset for each voxel')
ylim([-0.8 1.2])

Diff = diff(MinMax,1,2);
Percent = Diff/100;
% Reduce to 20% and 70% of remaining time course
for CurVal = 1:size(Diff)
    Vector = NewArray(CurVal,:);
    Start = round(Percent(CurVal)*20);
    End = round(Percent(CurVal)*80);
    Start = Start + MinMax(CurVal, 1);
    End = End +MinMax(CurVal, 1);
    Vector(1:Start-1) = NaN;
    Vector(End+1:max(size(Vector))) = NaN;
    FinalArray(CurVal,:) = Vector;
end
figure, plot(tim-2,FinalArray')
xlabel('Time (s) from stim onset')
ylabel('Normalised BOLD resoponse')
title('Onset for each voxel post reduction')
ylim([-0.8 1.2])


% fit model to time course and extract intercepts
for CurVal = 1:size(FinalArray)
    mdl = fitlm(tim, FinalArray(CurVal,:));
    Intercepts(CurVal) = table2array(mdl.Coefficients(2,1));
    % h=figure
    % mdl.plot
    % waitfor(h)
end

 InterMean = mean(Intercepts);
 Intercepts(37) = InterMean;

% Create and display heat map for visualisation purposes
HeatMap = zeros(size(restrictedAngle));
for x = 1:size(Intercepts')
    HeatMap(xCoord(x),yCoord(x)) = Intercepts(x);
end
imagesc(HeatMap)
title('Delay in BOLD onset for the chosen ROI')

% Average LH/RH
for x = 1:size(Intercepts')
    if Intercepts(x) ~= 0 && xCoord(x) < 45
        % Append to a vector
        LHData(x) = Intercepts(x);
    else if Intercepts(x) ~= 0
        RHData(x) = Intercepts(x);
    end
    end
end

%Check that both the left and right hemispheres could be parsed
ExistLH = exist(LHData);
ExistRH = exist(RHData);
% if not dont run the calculations
if ExistLH ~= 0 && ExistRH ~= 1
    LHDataN0 = nonzeros(LHData');
    RHDataN0 =nonzeros(RHData');

    % Compare average response 
    [TTest, PVal] = ttest2(RHDataN0, LHDataN0)
    RHMean = mean(RHDataN0)
    LHMean = mean(LHDataN0)

    DescXTTest = {"RightHemiAv", "LeftHemiAv", "TTest", "PVal" ; RHMean, LHMean, TTest, PVal};

    filename = './data/VOL90Run1.xlsx';
    writecell(DescXTTest,filename,'Sheet',1)
end