function [Z, mu] = DeStrip_tv(Y, fid, gt)

[H, W] = size(Y);


%--------------------------- 列噪声初始值 --------------------------------
MaxIter = 10;

L = 7;
ker = fspecial('gaussian', [L, L], (L - 1)/4);

pad_len    = (L - 1) / 2;
pad_Y      = padarray(Y, [pad_len, pad_len], 'symmetric');
local_mean = conv2(pad_Y, ker, 'valid');

res = Y - local_mean;

mu = median(res);          % 初始值

fprintf(fid, '%f %f %f %f %f\n', mu(1, floor(W/6)), mu(1, floor(W/3)), mu(1, floor(W/2)), mu(1, floor(W/1.5)), mu(1, floor(W/1.2)));
%-------------------------------------------------------------------------

%------------------------ 协同列噪声估计和图像去噪 ------------------------
psnr = 10 * log10(255^2 / mean(mean((Y - gt).^2)));
fprintf('Iter: %d,  psnr: %f\n', 0, psnr);

for k = 1 : MaxIter
    tmp = Y - repmat(mu, [H,1]);      % 减去上一轮估计得到的列噪声的均值
    Z   = tv_denoise(tmp);     % 去噪
    res = Y - Z;      % 减去去噪后的图像得到新的残差
    mu  = mean(res);    % 计算每一列残差的均值以更新列噪声均值的估计
    
    fprintf(fid, '%f %f %f %f %f\n', mu(floor(W/6)), mu(floor(W/3)), mu(floor(W/2)), mu(floor(W/1.5)), mu(floor(W/1.2)));
    
    psnr = 10 * log10(255^2 / mean(mean((Z - gt).^2)));
    
    fprintf('Iter: %d,  psnr: %f\n', k, psnr);
    
end
%-------------------------------------------------------------------------


%------------------------------------------------------------------------
function [img_est] = tv_denoise(Y)

%====================== Lingfei Song 2022.3.30 =========================
% (non-isotropic) Total-Variation denoise
%========================================================================

%------------------------------ 计算噪声方差 -----------------------------
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

med           = median(sigmaV_square(:));
tmp           = sigmaV_square(sigmaV_square > 0.3*med & sigmaV_square < med);
his           = histogram(tmp);
[~,idx]       = max(his.Values);
sigmaN_square = his.BinEdges(idx+1);

%------------------------------------------------------------------------

psf = ones(1,1);

[M,N] = size(Y);
MAX_ITER = 15;

d1 = [1 -1];    % horizontal difference
d2 = [1 -1]';   % vertical difference

D1 = fft2(d1,M,N);
D2 = fft2(d2,M,N);
IMG_BLUR = fft2(Y);
PSF = fft2(psf,M,N);
CNJ_D1 = conj(D1);
CNJ_D2 = conj(D2);
CNJ_PSF = conj(PSF);

rho = 0.1;
mu1 = zeros(M,N);
mu2 = zeros(M,N);
z1  = ifft2(D1.*IMG_BLUR);
z2  = ifft2(D2.*IMG_BLUR);

%------------------------- 计算梯度的绝对值偏差 -----------------------------
sigma = median(abs(z1(:)));
%---------------------------------------------------------------------------

lmd = sigmaN_square / sigma;

loss = zeros(MAX_ITER, 1);
img_est0 = zeros(M,N);
for i = 1:MAX_ITER
    IMG_EST = (CNJ_PSF.*IMG_BLUR + CNJ_D1.*fft2(mu1)/2 + CNJ_D2.*fft2(mu2)/2 + rho*CNJ_D1.*fft2(z1) + rho*CNJ_D2.*fft2(z2))...
            ./ (CNJ_PSF.*PSF + rho*CNJ_D1.*D1 + rho*CNJ_D2.*D2);
    img_est = ifft2(IMG_EST);
    
    TV1 = ifft2(D1.*IMG_EST);
    TV2 = ifft2(D2.*IMG_EST);
    z1 = soft(TV1, mu1, lmd, rho);
    z2 = soft(TV2, mu2, lmd, rho);
    mu1 = mu1 + 2 * rho * (z1 - TV1);
    mu2 = mu2 + 2 * rho * (z2 - TV2);
    rho = rho * 1.05;
    
    loss(i) = sum(sum(((ifft2(PSF.*IMG_EST) - Y).^2))) + lmd * sum(sum(abs(TV1))) ...
                                                                 + lmd * sum(sum(abs(TV2)));

    if norm(img_est(:)-img_est0(:))/norm(img_est(:)) < 1E-4
        rho = rho * 2;
    end
    
    if norm(img_est(:)-img_est0(:))/norm(img_est(:)) < 1E-5
        break;
    end
    
    img_est0 = img_est;
end

plot(loss(1:i)); title('loss');


function [z] = soft(TV, mu, lmd, rho)

%---------------------------------------------------------------
% z = d * img_est - mu / (2 * rho) - lmd / (2 * rho), 
%     if (d * img_est - mu / (2 * rho) - lmd / (2 * rho)) > 0;
% z = d * img_est - mu / (2 * rho) + lmd / (2 * rho),
%     if (d * img_est - mu / (2 * rho) + lmd / (2 * rho)) < 0;
% z = 0, otherwise.
%---------------------------------------------------------------
[M,N] = size(TV);

z1 = TV - mu / (2 * rho) - lmd / (2 * rho);
z2 = TV - mu / (2 * rho) + lmd / (2 * rho);

z = zeros(M,N);
idx = z1 > 0;
z(idx) = z1(idx);
idx = z2 < 0;
z(idx) = z2(idx);

