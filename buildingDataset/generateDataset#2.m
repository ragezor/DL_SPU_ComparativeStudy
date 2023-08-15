%% Code of generating dataset #2

%image size
rows = 256; cols = 256;
% wrap-count numbers
num_order = 16;
%Image numbers
num_img = 1000;
for i_img = 1:num_img
    z_final = zeros(rows*cols, 1);
    %peaks numbers
    num_peaks = randi(20, 1);
    for j = 1:num_peaks
        rng('shuffle'); mu = randi(rows, 1, 2);
        % t1 = randi(rows/4, 2, 2)
        % sigma = t1' * t1 + 0.01 * eye(2);
        rng('shuffle'); t1 = 25 * randi(rows, 1, 1);
        sigma = [t1(1) 0;0 t1(1)];
        [X, Y] = meshgrid(1:cols, 1:rows);
        x = reshape(X, rows*cols, 1);
        y = reshape(Y, rows*cols, 1);
        p=[x, y];
        z = mvnpdf(p, mu, sigma);
        z = z - min(z);
        z = z ./ max(z);    
        rng('shuffle'); operation= rand(1);
        if operation > 0.5
            z_final = z_final + z;
        else
            z_final = z_final - z;
        end        
    end   
    z_final = z_final - min(z_final);
    z_final = z_final ./ max(z_final);    
    z_final = num_order * 2 * pi * z_final;
    Z = reshape(z_final, rows, cols);  
    x=1:256;
    y=1:256;
    %figure(1);
   % [x,y]=meshgrid(x,y);
   % mesh(Z);
   % grid on£»
    %order_num = zeros(1, 15);
    
    %wrap-count map
    order = floor(Z/(pi+pi));
    %wrapped phase
    wrapped = Z -  order * (pi + pi)-pi;
    phaseW_png = (wrapped-(-pi))/(2*pi)*254+1;
    %order_png=mat2gray(order);
   %imwrite(uint8(phaseW_png),'res1.png');
   order_png=uint8(order);
   %i_img_new=i_img+1400;
   i_img_new=i_img;
   %imwrite(order_png,'order1997.png');
   %writeNPY(wrapped, strcat('E:\work\guas_data\input\eval\','wrappedd_',num2str(i_img),'.npy'));
  %a\ writeNPY(order, strcat('E:\work\guas_data\label\eval\','order_',num2str(i_img),'.npy'));
  fid=fopen('train.txt','a+'); cd
  fprintf(fid,'%s\r\n',num2str(i_img_new,'%05d'));
  fclose(fid);
   imwrite(uint8(phaseW_png), strcat('D:\2022_1_5_work\gaus\input\test\','warpped_',num2str(i_img_new,'%05d'),'.png'));
   imwrite(order_png, strcat('D:\2022_1_5_work\gaus\label\test\','order_',num2str(i_img_new,'%05d'),'.png'));
    close all;
end
