clear; close all;

addpath('.\src\');

%----------------------------- 供调试使用 ---------------------------------
fid = fopen('log2.txt', 'w');
%-------------------------------------------------------------------------


%--------------------------------------------------------------------------
in_dir = '.\data\Set12\';
%--------------------------------------------------------------------------


%----------------------------- 其他参数 -----------------------------------
MIN = 0;
MAX = 255;

cnt = 1;
%--------------------------------------------------------------------------

files = dir([in_dir, '*.tif']);

for i = 1 : 1: 3%length(files)
    
    %------------------------ 读取视频帧 #Xenics ----------------------------
    FRAME = imread([in_dir, files(i).name], 'tif');
    FRAME = double(FRAME(3:end-2,3:end-2,1));
    
    FRAME = FRAME - min(FRAME(:));
    FRAME = FRAME / max(FRAME(:)) * 255;
    
    [H, W] = size(FRAME);
    %----------------------------------------------------------------------
    
    ORG   = FRAME + repmat(randn(1, W) * 30, [H,1]) + randn(H,W) * 10;
    
    figure; imshow(ORG, []);      
    
    %----------------------------------------------------------------------
    [NUC, mu] = DeStrip_wiener(ORG, fid, FRAME);
    %----------------------------------------------------------------------

    
    %--------------------- 以下代码仅作显示使用 --ss--------------------------
    I = sort(ORG(:));
    MIN = I(1);
    MAX = I(end - 1);
    
    NUC_Dis = NUC - MIN; 
    NUC_Dis(NUC_Dis < 0) = 0;
    NUC_Dis(NUC_Dis > MAX - MIN) = MAX - MIN;
    NUC_Dis = NUC_Dis * 255 / (MAX - MIN);

    ORG_Dis = ORG - MIN;
    ORG_Dis(ORG_Dis < 0) = 0;
    ORG_Dis(ORG_Dis > MAX - MIN) = MAX - MIN;
    ORG_Dis = ORG_Dis * 255 / (MAX - MIN);
   
    
    figure; imshow([ORG_Dis, NUC_Dis], [0, 255]);
    
    name = files(i).name;
    imwrite(uint8(NUC_Dis), ['.\Results\Set12\', name(1:end-4), '_ours_mNL_sigma15.png']);
       
    fprintf('frame %2d. %s \n', cnt, name(1:end-4));
    
    cnt = cnt + 1;
    
end

fclose(fid);

p1 = load('log2.txt');

len = size(p1, 1);
figure; plot(1:len, p1(:,1), 1:len, p1(:,2), 1:len, p1(:,3), 1:len, p1(:,4), 1:len, p1(:,5), 'linewidth', 2);
xlabel('Iter'); ylabel('Bias'); title('Convergence Issues');

