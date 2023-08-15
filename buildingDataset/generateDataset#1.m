%% Code of generating dataset #1

%Display the first 10 Zernike functions
num=0
for i_count=1:1
     x = -1:0.0039:1;
    [X,Y]=meshgrid(x,x);
    [theta,r] = cart2pol(X,Y);
    idx = r<=1;
    z = zeros(size(X));
    z_all = zeros(size(X));
    lim=randi([2,30],1,1);
    add=floor(lim/2)-1;
    n=randi([-lim,lim],1,30);
    i_add_ran=randi([1,10],1,1);
    subs=randi([0,add],1,30);
    subs=subs*2;
    m=n-subs;
    y = zernfun(n,m,r(idx),theta(idx));
    a = 1:15;
    n_new=randi([5,15],1,1);% random 5 z +
    b=a(randperm(numel(a),n_new));
      for i=1:n_new
        z(idx) = y(:,b(i));
        z_all = z_all + z;
      end
    phi = z_all(1:512,1:512);
    phi=phi(128:383,128:383);
    phi_min = min(min(phi));
    phi_max = max(max(phi));
    phi = (phi-phi_min)/(phi_max-phi_min);
        order_number =16;
        phi_new = phi*2*pi*order_number;
        order = floor(phi_new/2/pi);
        phaseW = phi_new - order*2*pi-pi;
                phaseW_png = (phaseW-(-pi))/(2*pi)*254+1;
                phaseW_png_int=uint8(phaseW_png);
                if sum(sum( phaseW_png_int))==0
                    i=i-1;
                    continue;
                end;
                 num = num+1;               
                imwrite( phaseW_png_int, strcat('D:/2022_1_5_work/16zer/input/eval/','wrapped_',num2str(num,'%05d'),'.png'));
                imwrite(uint8(order), strcat('D:/2022_1_5_work/16zer/label/eval/','order_',num2str(num,'%05d'),'.png'));
end



