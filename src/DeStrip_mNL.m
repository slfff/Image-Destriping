function [Z, mu] = DeStrip_mNL(Y, fid, gt)

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
% mu = zeros(1, W);

fprintf(fid, '%f %f %f %f %f\n', mu(1, floor(W/6)), mu(1, floor(W/3)), mu(1, floor(W/2)), mu(1, floor(W/1.5)), mu(1, floor(W/1.2)));
%-------------------------------------------------------------------------

%------------------------ 协同列噪声估计和图像去噪 ------------------------
psnr = 10 * log10(255^2 / mean(mean((Y - gt).^2)));
fprintf('Iter: %d,  psnr: %f\n', 0, psnr);

for k = 1 : MaxIter
    tmp = Y - repmat(mu, [H,1]);      % 减去上一轮估计得到的列噪声的均值
    Z   = NLM(tmp);     % 去噪
    res = Y - Z;      % 减去去噪后的图像得到新的残差
    mu  = mean(res);    % 计算每一列残差的均值以更新列噪声均值的估计
    
    fprintf(fid, '%f %f %f %f %f\n', mu(floor(W/6)), mu(floor(W/3)), mu(floor(W/2)), mu(floor(W/1.5)), mu(floor(W/1.2)));
    
    pre_psnr = psnr;
    
%     sigma = std(mu);
%     mu(mu >  1 * sigma) =  1 * sigma;
%     mu(mu < -1 * sigma) = -1 * sigma;
    
    psnr = 10 * log10(255^2 / mean(mean((Z - gt).^2)));
    
    fprintf('Iter: %d,  psnr: %f\n', k, psnr);
    
%     if psnr - pre_psnr < 0.01
%         break;
%     end
end
%-------------------------------------------------------------------------



function [z] = NLM(y)

m = mean(y(:));

y = y - m;

[H, W] = size(y);
L = 9;

%----------------------------- 计算噪声方差 -------------------------------
ker = fspecial('gaussian', [7, 7], (7 - 1)/4);

pad_len    = (7 - 1) / 2;
pad_y      = padarray(y, [pad_len, pad_len], 'symmetric');
local_mean = conv2(pad_y, ker, 'valid');

res = y - local_mean;

sigmaV = conv2(abs(res), ker, 'valid');

sigmaV_square = sigmaV .^ 2;

med     = median(sigmaV_square(:));
tmp     = sigmaV_square(sigmaV_square > 0.3*med & sigmaV_square < med);
his     = histogram(tmp);
[~,idx] = max(his.Values);

sigma = sqrt(his.BinEdges(idx+1));
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
pad_len = (L - 1) / 2;
pad_y   = padarray(y, [pad_len, pad_len], 'symmetric');
% 扩充后方便窗口操作
%--------------------------------------------------------------------------

z = zeros(H, W);

h_square = L * L * 2 * sigma^2;

% weight = zeros(H,W);
weight111 = zeros(H,W);

rad = max(round(H / 14), 36);

for i = 1 : H
    for j = 1 : W
        patch_main = pad_y(i:i+L-1, j:j+L-1);
        
        min_u = max(i-rad, 1);
        max_u = min(i+rad, H);
        min_v = max(j-rad, 1);
        max_v = min(j+rad, W);
        
%         weight(min_u : max_u, min_v : max_v) = 0;
%         weight(i,j) = 1;
        
        block = pad_y(min_u : max_u + L -1, min_v : max_v + L - 1);
        
        weight111(min_u : max_u, min_v : max_v) = ...
            exp(- (sum(sum(patch_main.^2)) + conv2(block.^2, ones(L,L), 'valid') - 2 * filter2(patch_main, block, 'valid')) / h_square);
        
        weight111(:,j) = 0; weight111(i,j) = 1;
        tmp_weight111 = weight111(min_u : max_u, min_v : max_v) / sum(sum(weight111(min_u : max_u, min_v : max_v)));
        
%         for u = min_u : max_u
%             for v = min_v : max_v
%                 if v ~= j
%                     patch = pad_y(u:u+L-1, v:v+L-1);
%                     weight(u,v) = exp(- sum((patch(:) - patch_main(:)).^2) / h_square);
%                 end
%             end
%         end
% 
%         tmp_weight = weight(min_u : max_u, min_v : max_v) / sum(sum(weight(min_u : max_u, min_v : max_v)));
        
        tmp = sum(sum(y(min_u : max_u, min_v : max_v) .* tmp_weight111));

        z(i,j) = tmp;
    end
end

z = z + m;



