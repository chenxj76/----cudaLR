writerObj=VideoWriter('test.avi');  %// 定义一个视频文件用来存动画  
writerObj.FrameRate=3;
open(writerObj);                    %// 打开该视频文件  
for i = 0:1:50  
    f1 = figure(1);
    hold on
%     colormap(gray(256));
    colormap(jet);
    ap=['ap', num2str(i)];
    T = readtable(ap);
    area=table2array(T);
    imagesc(1:1000,1:1000,area(1:1000,1:1000))%eval: Execute MATLAB expression in text
    axis([1 1000 1 1000])
    colorbar()
    if(i==0)
        title('t=0.02ms');
    else
        title(['t=' num2str(i*10) 'ms']);
    end
    pause(0.1)
    frame = getframe(f1);            %// 把图像存入视频文件中  
    writeVideo(writerObj,frame); %// 将帧写入视频  
end
close(writerObj); %// 关闭视频文件句柄  