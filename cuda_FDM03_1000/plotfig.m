 T = readtable('ap1');
    area=table2array(T);
    imagesc(1:1000,1:1000,area(1:1000,1:1000))%eval: Execute MATLAB expression in text
    axis([1 1000 1 1000])
    colorbar()