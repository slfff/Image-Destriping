function [Z, mu] = DeStrip_wiener(Y, fid, gt)

[H, W] = size(Y);


%--------------------------- 列噪声初始值 --------------------------------
MaxIter = 30;

L = 7;
ker = fspecial('gaussian', [L, L], (L - 1)/4);

pad_len    = (L - 1) / 2;
pad_Y      = padarray(Y, [pad_len, pad_len], 'symmetric');
local_mean = conv2(pad_Y, ker, 'valid');

res = Y - local_mean;

mu = median(res) * 0;          % 初始值

fprintf(fid, '%f %f %f %f %f\n', mu(1, floor(W/6)), mu(1, floor(W/3)), mu(1, floor(W/2)), mu(1, floor(W/1.5)), mu(1, floor(W/1.2)));
%-------------------------------------------------------------------------

%------------------------ 协同列噪声估计和图像去噪 ------------------------
psnr = 10 * log10(255^2 / mean(mean((Y - gt).^2)));
fprintf('Iter: %d,  psnr: %f\n', 0, psnr);

for k = 1 : MaxIter
    tmp = Y - repmat(mu, [H,1]);      % 减去上一轮估计得到的列噪声的均值
    Z   = wiener(tmp);     % 去噪
    res = Y - Z;      % 减去去噪后的图像得到新的残差
    mu  = mean(res);    % 计算每一列残差的均值以更新列噪声均值的估计
    
    fprintf(fid, '%f %f %f %f %f\n', mu(floor(W/6)), mu(floor(W/3)), mu(floor(W/2)), mu(floor(W/1.5)), mu(floor(W/1.2)));
    
    psnr = 10 * log10(255^2 / mean(mean((Z - gt).^2)));
    
    fprintf('Iter: %d,  psnr: %f\n', k, psnr);
    
end
%-------------------------------------------------------------------------


%--------------------------------------------------------------------------
function [Z] = wiener(Y)

L = 11;

pad_len = (L - 1) / 2;
pad_Y   = padarray(Y, [pad_len, pad_len], 'symmetric');

ker        = fspecial('gaussian', [L, L], (L - 1)/4);
local_mean = conv2(pad_Y, ker, 'valid');

residual = Y - local_mean;  % 图像局部残差

pad_len = (L - 1) / 2;
pad_residual_abs = padarray(abs(residual), [pad_len, pad_len], 'symmetric');

sigmaV = conv2(pad_residual_abs, ker, 'valid');

sigmaV_square = sigmaV .^ 2;
%%----------------------------- 噪声方差 -----------------------------------
med           = median(sigmaV_square(:));
tmp           = sigmaV_square(sigmaV_square > 0.3*med & sigmaV_square < med);
his           = histogram(tmp);
[~,idx]       = max(his.Values);
sigmaN_square = his.BinEdges(idx+1);

sigmaU_square = max(sigmaV_square - sigmaN_square, 0);  % 减去列噪声方差
%%-------------------------------------------------------------------------

Z = sigmaU_square ./ (sigmaV_square) .* residual + local_mean;


